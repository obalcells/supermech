# %% Importing things
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, GPTNeoXForCausalLM
from fastchat.conversation import Conversation, get_conv_template
from livelossplot import PlotLosses
from jaxtyping import Int, Float
from typing import Tuple, List, Optional, Dict
import einops

# Terminology:
# `input_ids`: the tokenizer input (including the instruction, the adversarial string, and the target and padding tokens)
# `batch_size`: number of different candidates we explore at the same time
# Slices: `adv_suffix_slice`, (the adversarial string),
#         `target_slice` (only target string we want to elicit)
#         `input_slice` (everything except for the target string)

# %% Base Suffix Attack class
class SuffixAttack:
    def __init__(self, model, tokenizer, conv_template: Conversation, instruction: str, target: str, initial_adv_suffix: str):
        self.model = model
        self.tokenizer = tokenizer

        self.instruction = instruction
        self.target = target
        self.adv_suffix = initial_adv_suffix # the initial adversarial string

        self.allow_non_ascii = True
        self.batch_size = 16 
        self.topk = 30 
        self.temp = 1

        self.device = 'cpu'

        # self.user_role_slice = None 
        # self.goal_slice = None
        # self.assistant_role_slice = None

        self.conv_template = conv_template
        # self.input_slice = None
        # self.target_slice = None
        # self.loss_slice = None

    def run(self):
        # get token ids for the initial adversarial string
        input_ids, suffix_slice, target_slice = self.get_input_ids_for_suffix(
            adv_suffix=self.adv_suffix,
        )

        # get the initial loss
        with torch.no_grad():
            batch = input_ids.unsqueeze(0)
            logits = self.model(input_ids=batch).logits
            loss = self.get_loss(input_ids=batch, logits=logits, target_slice=target_slice)
            print(f"Original adv suffix is {self.adv_suffix} with loss {loss[0].item()}")
        
        token_gradients = self.token_gradients(input_ids, suffix_slice, target_slice)

        candidate_strings, batch_of_candidates = self.get_batch_of_suffix_candidates(
            input_ids,
            suffix_slice,
            token_gradients,
            random_seed=0
        )

        del token_gradients

        print(f"Done getting the candidate strings! Candidate strings are {candidate_strings}")
        print(f"Batch is (shape {batch.shape}): {batch}")

        with torch.no_grad():
            attention_mask = (batch_of_candidates != self.tokenizer.pad_token_id).type(batch_of_candidates.dtype)
            logits = self.model(input_ids=batch_of_candidates, attention_mask=attention_mask).logits
            print(f"Logits shape {logits.shape}")
            # we have to update the target slice because the batch of candidates has different length
            target_len = target_slice.stop - target_slice.start
            target_slice = slice(batch_of_candidates.shape[1] - target_len, batch_of_candidates.shape[1])
            print(f"Last tokens in the batch of cands {batch_of_candidates[:,target_slice]}")
            loss = self.get_loss(input_ids=batch_of_candidates, logits=logits, target_slice=target_slice)
            del logits
            best_new_adv_suffix_idx = loss.argmin()
            best_new_adv_suffix_string = candidate_strings[best_new_adv_suffix_idx]
            print(f"New adv string is {best_new_adv_suffix_string} with loss {loss[best_new_adv_suffix_idx].item()}")

            fed_tokens = batch_of_candidates[best_new_adv_suffix_idx]
            print(f"Tokens that were fed are {batch_of_candidates[best_new_adv_suffix_idx]}")

        fed_logits = self.model(input_ids=fed_tokens.unsqueeze(0)).logits
        fed_loss = self.get_loss(input_ids=fed_tokens.unsqueeze(0), logits=fed_logits, target_slice=target_slice)
        print(f"Fed loss is {fed_loss[0].item()}")

        self.adv_suffix = best_new_adv_suffix_string

        # get token ids for the initial adversarial string
        input_ids, suffix_slice, target_slice = self.get_input_ids_for_suffix(
            adv_suffix=self.adv_suffix,
        )
        print(f"Tokens that will be feed now are {input_ids}")

        # get the initial loss
        with torch.no_grad():
            batch = input_ids.unsqueeze(0)
            logits = self.model(input_ids=batch).logits
            loss = self.get_loss(input_ids=batch, logits=logits, target_slice=target_slice)
            print(f"New adv suffix is {self.adv_suffix} with loss {loss[0].item()}")

    # returns the input tokens for the given suffix string
    # and also the slice within the tokens array containing the suffix and the target strings
    def get_input_ids_for_suffix(self, adv_suffix=None) -> Tuple[Int[Tensor, "seq_len"], slice, slice]:
        # if no suffix is given we just use the main one we got in the class
        if adv_suffix is None:
            adv_suffix = self.adv_suffix

        self.conv_template.messages = []

        self.conv_template.append_message(self.conv_template.roles[0], f"{self.instruction} {adv_suffix}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()

        # now we have to figure out which tokens correspond to the suffix and the target
        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids

        self.conv_template.messages = []

        self.conv_template.append_message(self.conv_template.roles[0], None)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        user_role_slice = slice(None, len(toks))

        self.conv_template.update_last_message(f"{self.instruction}")
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        goal_slice = slice(user_role_slice.stop, max(user_role_slice.stop, len(toks)-1))

        separator = ' ' if self.instruction else ''
        self.conv_template.update_last_message(f"{self.instruction}{separator}{adv_suffix}")
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        suffix_slice = slice(goal_slice.stop, len(toks)-1)

        self.conv_template.append_message(self.conv_template.roles[1], None)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        assistant_role_slice = slice(suffix_slice.stop, len(toks))

        self.conv_template.update_last_message(f"{self.target}")
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        target_slice = slice(assistant_role_slice.stop, len(toks)-1)

        self.conv_template.messages = []

        input_ids = torch.tensor(self.tokenizer(prompt).input_ids[:target_slice.stop])

        return input_ids, suffix_slice, target_slice 
    
    def cross_entropy_loss(
        self,
        *,
        input_ids: Int[Tensor, "batch_size seq_len"],
        logits: Float[Tensor, "batch_size seq_len d_vocab"],
        target_slice: slice=None,
    ) -> Float[Tensor, "batch_size"]:
        assert len(input_ids.shape) == 2 , "Input ids must be 2d tensor here (everywhere else it's 1d)"
        assert len(logits.shape) == 3, "Logits must be a 3d tensor" 

        # the loss slice is shifted by 1 to the left
        loss_slice = slice(target_slice.start-1, target_slice.stop-1)

        # we want the shape of the labels to match the first two dimensions of the batch
        # to be able to apply the gather operation
        assert target_slice.stop == input_ids.shape[1], "The target slice must be within the input ids"
        labels = einops.repeat(input_ids[:,target_slice], "batch_size seq_len -> batch_size seq_len 1", batch_size=logits.shape[0])

        target_logprobs = torch.gather(
            logits[:,loss_slice].log_softmax(dim=-1),
            dim=-1,
            index=labels
        ).squeeze(-1) # we have one extra dimension at the end we don't need

        #print(f"Logprobs has shape {target_logprobs.shape}")

        return target_logprobs.mean(dim=-1)

    # returns the loss for each of the adv suffixes we want to test
    # at the moment this is just the cross entropy loss on the target slice
    def get_loss(
        self,
        *,
        input_ids: Int[Tensor, "batch_size seq_len"],
        logits: Float[Tensor, "batch_size seq_len d_vocab"],
        target_slice: slice=None,
    ) -> Float[Tensor, "batch_size"]:
        assert target_slice is not None, "We must pass a target slice to calculate the loss" # in the future we might not need the target slice to calculate the loss
        return self.cross_entropy_loss(input_ids=input_ids, logits=logits, target_slice=target_slice)

    # computes the gradient on the one-hot input token matrix with respect to the loss
    def token_gradients(
        self,
        input_ids: Int[Tensor, "seq_len"],
        suffix_slice: slice,
        target_slice: slice
    ) -> Float[Tensor, "batch_size seq_len d_vocab"]:
        assert len(input_ids.shape) == 1, "At the moment `input_ids` must be a 1d tensor when passed to token gradients"

        if isinstance(self.model, LlamaForCausalLM):
            embed_weights = self.model.model.embed_tokens.weight
            embeds = self.model.model.embed_tokens(input_ids).detach()
        elif isinstance(self.model, GPTNeoXForCausalLM):
            embed_weights = self.model.base_model.embed_in.weight
            embeds = self.model.base_model.embed_in(input_ids.unsqueeze(0).to(self.device)).detach()
        else:
            assert False, "Only Llama and GPTNeoX models are supported"

        one_hot = torch.zeros(
            input_ids[suffix_slice].shape[0],
            embed_weights.shape[0],
            dtype=embed_weights.dtype
        ).to(self.device)

        # maybe use scatter here?
        one_hot[torch.arange(input_ids[suffix_slice].shape[0]),input_ids[suffix_slice]] = 1.0

        # this returns an error when run on MPS
        #one_hot = torch.zeros(
        #    input_ids[input_slice].shape[0],
        #    embed_weights.shape[0],
        #    device='cpu',
        #    dtype=embed_weights.dtype
        #)
        #one_hot = one_hot_cpu.scatter(
        #    1,
        #    input_ids[input_slice].unsqueeze(1),
        #    torch.ones(one_hot_cpu.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
        #)

        #print(f"One-hot shape is {one_hot.shape}")
        #print(f"Embed matrix has shape {embed_weights.shape}")

        one_hot.requires_grad_()
        #input_embeds = (one_hot @ embed_weights).unsqueeze(0)

        #print(f"Input embeds were computed")

        # now stitch it together with the rest of the embeddings
        #embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()


        full_embeds = torch.cat(
            [
                embeds[:,:suffix_slice.start,:],
                (one_hot @ embed_weights).unsqueeze(0),
                embeds[:,suffix_slice.stop:,:]
            ],
            dim=1
        ).to(self.device)

        #print(f"Stiched embeddings")

        #print(f"Stiched embeddings have type {full_embeds.dtype}")
        #print(f"Should have type {embeds.dtype}")

        #print(f"Shape of full embeds is: {full_embeds.shape}")

        # what else is in here?
        logits = self.model(inputs_embeds=full_embeds).logits

        #print("Compute the forward pass")

        #logits = model(inputs_embeds=full_embeds).logits

        #print(f"Gotten logits {logits.shape}")

        # targets = input_ids[target_slice].to(device)

        # we add one extra dimension at the beginning to the input token ids
        batch = input_ids.unsqueeze(0)
        loss = self.get_loss(input_ids=batch, logits=logits, target_slice=target_slice)
        # loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)

        #print(f"Forward pass")

        loss.backward()

        grad = one_hot.grad.clone()
        grad = grad / grad.norm(dim=-1, keepdim=True)

        # should we do this?
        self.model.zero_grad()

        return grad


    # selects k tokens within the topk (allowed) tokens, replaces them and returns a batch
    # we can feed to the model
    def get_batch_of_suffix_candidates(
        self,
        input_ids: Int[Tensor, "seq_len"],
        adv_suffix_slice: slice,
        grad: Float[Tensor, "suffix_len d_vocab"], # what does this batch_size represent?
        topk=1024,
        temp=1,
        filter_by_size=False,
        # not_allowed_tokens=None
        random_seed=None
    ) -> Tuple[List[str], Int[Tensor, "n_cands seq_len"]]:
        adv_suffix_token_ids = input_ids[adv_suffix_slice]

        assert len(adv_suffix_token_ids.shape) == 1, "The adv suffix array must be 1-dimensional at the moment"
        assert len(adv_suffix_token_ids) == grad.shape[0], "The length of the suffix (in tokens) must match the length of the gradient"

        # TODO: Add support for not_allowed_tokens
        # if not_allowed_tokens is not None:
        #     grad[:, not_allowed_tokens.to(grad.device)] = np.infty

        top_indices = (-grad).topk(topk, dim=1).indices
        adv_suffix_token_ids = adv_suffix_token_ids.to(grad.device)

        original_control_toks = adv_suffix_token_ids.repeat(self.batch_size, 1)

        new_token_pos = torch.arange(
            0,
            len(adv_suffix_token_ids),
            len(adv_suffix_token_ids) / self.batch_size,
            device=grad.device
        ).type(torch.int64)

        if random_seed is not None:
            random_seed = torch.manual_seed(random_seed)

        new_token_val = torch.gather(
            top_indices[new_token_pos], 1,
            torch.randint(0, topk, (self.batch_size, 1),
            device=grad.device)
        )

        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

        new_adv_strings = [self.tokenizer.decode(toks, skip_special_tokens=True) for toks in new_control_toks]

        # now we filter the suffix strings which 
        valid_candidate_strings = []
        valid_candidate_input_ids = []
        max_len_token_ids = 0
        prev_target_slice = None

        for new_adv_suffix in new_adv_strings:
            input_ids_for_this_suffix, _, _ = self.get_input_ids_for_suffix(adv_suffix=new_adv_suffix)
            # we could add extra conditions here
            # we could check that the tokens representing the instruction haven't changed
            # this would be useful to use caching for the prefix which never changes 
            if filter_by_size == True and len(input_ids_with_for_this_suffix) != len(input_ids):
                print(f"Warning: {adv_string} was not a valid candidate its token ids with len {len(input_ids_for_this_suffix)} are {input_ids_for_this_suffix}")
                continue

            valid_candidate_strings.append(new_adv_suffix)
            valid_candidate_input_ids.append(input_ids_for_this_suffix)
            max_len_token_ids = max(max_len_token_ids, len(input_ids_for_this_suffix)) 

        assert len(valid_candidate_strings) != 0, "All candidate strings were invalid"

        batch_size = len(valid_candidate_strings)
        batch = torch.full((batch_size, max_len_token_ids), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)

        # make sure that the target string is located in the same place for all candidates
        for i, candidate_input_ids in enumerate(valid_candidate_input_ids):
            # Pad each candidate input ids from the length to match the maximum length
            # print(f"Candidate input ids: {valid_candidate_input_ids}")
            # print("")
            batch[i, -len(candidate_input_ids):] = candidate_input_ids

        return valid_candidate_strings, batch 

# %% Quick tests
def test_shape(model, tokenizer, conv_template):
    test_attack = get_sample_attack(model, tokenizer, conv_template)

    input_ids, suffix_slice, target_slice = test_attack.get_input_ids(return_batch_dim=True)
    token_gradients = test_attack.token_gradients(input_ids, suffix_slice, target_slice)

    assert len(input_ids.shape) == 1, "Input ids must be a 1d tensor"
    assert len(token_gradients.shape) == 2, "Token gradients must be a 2d tensor (suffix_len, d_vocab))"

def test_candidate_sampling(model, tokenizer, conv_template):
    test_attack = get_sample_attack(model, tokenizer, conv_template)

    input_ids, suffix_slice, target_slice = test_attack.get_input_ids_for_suffix(
        adv_suffix=test_attack.adv_suffix
    )

    l = test_attack.get_loss(
        input_ids=input_ids.unsqueeze(0),
        logits=torch.zeros((1, len(input_ids), tokenizer.vocab_size)),
        target_slice=target_slice
    )

    token_gradients = test_attack.token_gradients(input_ids, suffix_slice, target_slice)

    candidate_strings, batch = test_attack.get_batch_of_suffix_candidates(
        input_ids,
        suffix_slice,
        token_gradients,
        topk=test_attack.topk,
        random_seed=0
    )

    print(f"Suffix candidate strings are: {candidate_strings}")

    og_token_replacement_candidates = original_sample_control(
        input_ids[suffix_slice],
        token_gradients,
        batch_size=test_attack.batch_size,
        topk=test_attack.topk,
        temp=test_attack.temp,
        not_allowed_tokens=None,
        random_seed=0
    ) 

    og_new_adv_suffix = original_get_filtered_cands(
        tokenizer,
        og_token_replacement_candidates,
        filter_cand=False,
        curr_control=test_attack.adv_suffix
    )
    
    print(f"Batch shape {batch.shape}")
    print(f"Custom suffix candidate strings: {candidate_strings}")
    print(f"Original suffix candidate strings: {og_new_adv_suffix}")

    assert sorted(og_new_adv_suffix) == sorted(candidate_strings), "Generated suffix candidates do not match original implementation suffix candidates"
    print("\033[92mSampling tests passed!\033[0m")

def test_loss():
    pass


# %% Return a sample attack to test stuff
def get_model_and_tokenizer(model_path="EleutherAI/pythia-1.4b", tokenizer_path="EleutherAI/pythia-1.4b", **kwargs):
    precision = torch.float32
    device = 'mps'
    auth_token = "" 

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=precision,
        trust_remote_code=True,
        use_auth_token=auth_token
    ).to(device)

    if 'llama-2' in model_path:
        use_fast = False
    elif 'pythia' in model_path:
        use_fast = True

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=use_fast
    )

    # I don't really know why we do this
    if 'llama-2' in model_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    elif 'pythia' in model_path:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# 'oasst_pythia' - for pythia
# 'llama-2' - for llama
def load_conversation_template(template_name):
    conv_template = get_conv_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()

    return conv_template


def get_sample_attack(model, tokenizer, conv_template):
    # model, tokenizer = get_model_and_tokenizer(model_path="../models/small_pythia", tokenizer_path="../models/eleuther_tokenizer")
    # conv_template = load_conversation_template("oasst_pythia")

    sample_attack = SuffixAttack(
        model=model,
        tokenizer=tokenizer,
        conv_template=conv_template,
        instruction="Hey dude, what is your name?",
        target="My name is John Cena!",
        initial_adv_suffix="! ! ! ! !"
    )

    return sample_attack

def original_sample_control(
    control_toks,
    grad,
    batch_size=4,
    topk=256,
    temp=1,
    not_allowed_tokens=None,
    random_seed=None
):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices

    print(f"OG top indices are: {top_indices}")

    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0,
        len(control_toks),
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)

    if random_seed is not None:
        random_seed = torch.manual_seed(random_seed)

    new_token_val = torch.gather(
        top_indices[new_token_pos], 1,
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks

def original_get_filtered_cands(
    tokenizer,
    control_cand: Int[Tensor, "batch_size"],
    filter_cand=True,
    curr_control=None
) -> List[str]:
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    #print(f"Cands are {cands}")
    #print(f"Counts are {count}")

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
    return cands

def print_tokens(
    tokens: Int[Tensor, "seq_len"],
    tokenizer
):
    print(tokenizer.decode(tokens, skip_special_tokens=True))



# %% other things...



""""
Now the string utils code
"""


"""
# 'oasst_pythia' - for pythia
# 'llama-2' - for llama
def load_conversation_template(template_name):
    conv_template = get_conv_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()

    return conv_template



def generate(model,
             tokenizer,
             prompt: str,
             max_new_tokens: int = 25,
             generation_config = None
) -> str:
    if generation_config is None:
        generation_config = model.generation_config
        generation_config.max_new_tokens = max_new_tokens

    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = model.generate(**inputs, generation_config=generation_config)
    return tokenizer.decode(tokens[0])
"""



# %%
