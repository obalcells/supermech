# %% Importing things
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, GPTNeoXForCausalLM, GenerationConfig
from fastchat.conversation import get_conv_template
from livelossplot import PlotLosses
from jaxtyping import Int, Float
from typing import Tuple, List, Optional, Dict
import einops
import gc
from IPython.display import display, HTML


# %% Base Suffix Attack class
class SuffixAttack:
    def __init__(
        self,
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer,
        instruction: str,
        target: str,
        initial_adv_suffix: str,
        batch_size: int=1024,
        topk: int=256,
        temp: float=1.0
    ):
        self.model = model
        self.tokenizer = tokenizer

        if isinstance(self.model, LlamaForCausalLM):
            conv_template_name = "llama-2"
            self.tokenizer.pad_token = tokenizer.unk_token
            self.tokenizer.padding_side = 'left'
        elif isinstance(self.model, GPTNeoXForCausalLM):
            conv_template_name = 'oasst_pythia'
            self.tokenizer.padding_side = 'left'
        else:
            assert False, "Only Llama and GPTNeoX models are supported"        

        self.conv_template = get_conv_template(conv_template_name)
        assert self.conv_template.name in ["llama-2", "oasst_pythia"], "Only two conversation templates are supported at the moment"

        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()
    
        self.instruction = instruction
        self.target = target
        self.adv_suffix = initial_adv_suffix # the initial adversarial string

        # self.allow_non_ascii = True # not supported at the moment
        self.batch_size = batch_size 
        self.topk = topk 
        self.temp = temp 

        self.device = self.model.device 

    def step(self):
        # get token ids for the initial adversarial string
        input_ids, suffix_slice, target_slice = self.get_input_ids_for_suffix(
            adv_suffix=self.adv_suffix,
        )
        
        token_gradients = self.token_gradients(input_ids, suffix_slice, target_slice)

        candidate_strings, batch_of_candidates = self.get_batch_of_suffix_candidates(
            input_ids,
            suffix_slice,
            token_gradients
        )

        del token_gradients; gc.collect()

        with torch.no_grad():
            attention_mask = (batch_of_candidates != self.tokenizer.pad_token_id).type(batch_of_candidates.dtype)
            logits = self.model(input_ids=batch_of_candidates, attention_mask=attention_mask).logits
            target_len = target_slice.stop - target_slice.start
            target_slice = slice(batch_of_candidates.shape[1] - target_len, batch_of_candidates.shape[1])
            candidate_losses = self.get_loss(input_ids=batch_of_candidates, logits=logits, target_slice=target_slice)
            best_new_adv_suffix_idx = candidate_losses.argmin()
            self.adv_suffix = candidate_strings[best_new_adv_suffix_idx]

        del logits, batch_of_candidates, attention_mask; gc.collect()

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

        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids

        if self.conv_template.name == 'llama-2':
            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            goal_slice = slice(user_role_slice.stop, max(user_role_slice.stop, len(toks)))

            separator = ' ' if self.instruction else ''
            self.conv_template.update_last_message(f"{self.instruction}{separator}{adv_suffix}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            suffix_slice = slice(goal_slice.stop, len(toks))

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            assistant_role_slice = slice(suffix_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            target_slice = slice(assistant_role_slice.stop, len(toks)-2)

        elif self.conv_template.name == "oasst_pythia":
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
        
        else:
            assert False, "Conversation template not supported"

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
        # we want the shape of the labels to match the first two dimensions of the batch
        # to be able to apply the gather operation
        assert target_slice.stop == input_ids.shape[1], "The target slice must be at the end of the prompt"

        # the loss slice is shifted by 1 to the left
        loss_slice = slice(target_slice.start-1, target_slice.stop-1)

        labels = einops.repeat(input_ids[:,target_slice], "batch_size seq_len -> batch_size seq_len 1", batch_size=logits.shape[0])

        target_logprobs = torch.gather(
            logits[:,loss_slice].log_softmax(dim=-1),
            dim=-1,
            index=labels
        ).squeeze(-1) # we have one extra dimension at the end we don't need

        return -target_logprobs.mean(dim=-1)

    # returns the loss for each of the adv suffixes we want to test
    # at the moment this is just the cross entropy loss on the target slice
    def get_loss(
        self,
        *,
        input_ids: Int[Tensor, "batch_size seq_len"],
        logits: Float[Tensor, "batch_size seq_len d_vocab"],
        target_slice: slice=None,
    ) -> Float[Tensor, "batch_size"]:
        assert target_slice is not None, "A target slice must be passed to calculate the loss" # in the future we might not need the target slice to calculate the loss
        return self.cross_entropy_loss(input_ids=input_ids, logits=logits, target_slice=target_slice)

    def get_loss_for_suffix(self, adv_suffix: str=None) -> Float:
        if adv_suffix is None:
            adv_suffix = self.adv_suffix
        input_ids, suffix_slice, target_slice = self.get_input_ids_for_suffix(adv_suffix=adv_suffix)
        batch = input_ids.unsqueeze(0)
        attention_mask = (batch != self.tokenizer.pad_token_id).type(batch.dtype)
        logits = self.model(input_ids=batch, attention_mask=attention_mask).logits
        return self.get_loss(input_ids=batch, logits=logits, target_slice=target_slice)[0].item()

    def get_loss_for_input_ids(self, input_ids: Int[Tensor, "seq_len"], target_slice: slice) -> Float:
        batch = input_ids.unsqueeze(0)
        attention_mask = (batch != self.tokenizer.pad_token_id).type(batch.dtype)
        logits = self.model(input_ids=batch, attention_mask=attention_mask).logits
        loss = self.get_loss(input_ids=input_ids.unsqueeze(0), logits=logits, target_slice=target_slice)[0].item()
        return loss

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

        one_hot.requires_grad_()
        #input_embeds = (one_hot @ embed_weights).unsqueeze(0)

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

        # is this needed here too?
        self.model.zero_grad()

        logits = self.model(inputs_embeds=full_embeds).logits

        # we add one extra dimension at the beginning to the input token ids
        batch = input_ids.unsqueeze(0)
        loss = self.get_loss(input_ids=batch, logits=logits, target_slice=target_slice)

        loss.backward()

        grad = one_hot.grad.clone()
        grad = grad / grad.norm(dim=-1, keepdim=True) # why do this?

        # is this necessary?
        self.model.zero_grad()
        del logits, full_embeds, one_hot, embeds, embed_weights; gc.collect()

        return grad


    # selects k tokens within the topk (allowed) tokens, replaces them and returns a batch
    # we can feed to the model
    def get_batch_of_suffix_candidates(
        self,
        input_ids: Int[Tensor, "seq_len"],
        adv_suffix_slice: slice,
        grad: Float[Tensor, "suffix_len d_vocab"], # what does this batch_size represent?
        filter_by_size=True,
        # not_allowed_tokens=None
        random_seed=None
    ) -> Tuple[List[str], Int[Tensor, "n_cands seq_len"]]:
        adv_suffix_token_ids = input_ids[adv_suffix_slice]

        assert len(adv_suffix_token_ids.shape) == 1, "The adv suffix array must be 1-dimensional at the moment"
        assert len(adv_suffix_token_ids) == grad.shape[0], "The length of the suffix (in tokens) must match the length of the gradient"

        # TODO: Add support for not_allowed_tokens
        # if not_allowed_tokens is not None:
        #     grad[:, not_allowed_tokens.to(grad.device)] = np.infty

        top_indices = (-grad).topk(self.topk, dim=1).indices
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
            torch.randint(0, self.topk, (self.batch_size, 1),
            device=grad.device)
        )

        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

        # print(f"Prev suffix toks are: {adv_suffix_token_ids}")
        # print(f"First new control toks are: {new_control_toks[0]}")

        # print(f"Decoded prev suffix string is: {self.tokenizer.decode(adv_suffix_token_ids, skip_special_tokens=True)}")
        # print(f"Decoded first new control string is: {self.tokenizer.decode(new_control_toks[0], skip_special_tokens=True)}")

        # cur_string = self.tokenizer.decode(adv_suffix_token_ids, skip_special_tokens=True)
        # new_string = self.tokenizer.decode(new_control_toks[0], skip_special_tokens=True)
        # print(f"Old string is {cur_string}")
        # print(f"New string is {new_string}")

        # old_input_ids, _, _ = self.get_input_ids_for_suffix(adv_suffix=cur_string)
        # new_input_ids, _, _ = self.get_input_ids_for_suffix(adv_suffix=new_string)
        # print(f"Old input ids are: {old_input_ids}")
        # print(f"New input ids are: {new_input_ids}")
        # assert False

        # the problem is here??
        new_adv_strings = [self.tokenizer.decode(toks, skip_special_tokens=True) for toks in new_control_toks]

        # now we filter the suffix strings which 
        valid_candidate_strings = []
        valid_candidate_input_ids = []
        max_len_token_ids = 0
        # prev_target_slice = None
        # cnt = 0
        # cnt2 = 0

        # - we changed the suffix to ! ! ! ! ! ! ! ! ! and now the previous other assert is triggered?
        # - Let's first investigate for "! ! ! ! ! ! ! !" 
        #   - The previous assert is being triggered length difference of 3 vs 11?
        #   - Is everything due to add_special_tokens=False? No
        #   - Is it just a coincidence and most of the cands will be ok? No
        #   - Is it a problem of the suffix "! ! ! ! ! ! !" being very fragile? 
        #       - Yes it seems so. With a more or less pure noise suffix we don't have the same problem
        #   - Was `add_special_tokens=True` messing up before? No, it's not just that (checked before already, pretty sure now)
        #   - Manual check? (quick!)
        #       - mmmm, very sussss. Notice that we print the left side without spaces and the right side with spaces
        #       - See this: "Suffix is !!!!!!!!!!Based vs prev ! ! ! ! ! ! ! ! ! ! !"
        #       - See this: decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True) in the llm-attacks code
        #       - Let's try to manually decode the main adv suffix and see what we get
        #       - Why are the spaces skipped? Are we missing something in the tokenizer?
        # Let's move on

        for i, new_adv_suffix in enumerate(new_adv_strings):
            input_ids_for_this_suffix, _, _ = self.get_input_ids_for_suffix(adv_suffix=new_adv_suffix)

            # should we set a size limit too?  

            if filter_by_size == True and len(self.tokenizer(new_adv_suffix, add_special_tokens=False).input_ids) != len(adv_suffix_token_ids):
            # if filter_by_size == True and len(input_ids_for_this_suffix) != len(input_ids):
                continue
                # print("A)")
                # print(f"i is {i}")
                # print(f"Length difference {len(self.tokenizer(new_adv_suffix, add_special_tokens=False).input_ids)} vs {len(adv_suffix_token_ids)}")
                # print(f"Suffix is {new_adv_suffix} vs prev {self.adv_suffix}")
                # print(f"Suffix tokens are {self.tokenizer(new_adv_suffix, add_special_tokens=False).input_ids}")
                # print(f"Control toks are are {new_control_toks[i]}")
                # print(f"Decoding the control toks differently: {self.tokenizer.decode(new_control_toks[i])}")
                # print(f"")
                # cnt2 += 1

            # about half of the candidates are filtered if we change the condition to be:
            # len(input_ids_for_this_suffix) == len(input_ids
            # 
            # if len(input_ids_for_this_suffix) != len(input_ids):
            #     print("B)")
            #     print(f"i is {i}")
            #     print(f"Ids Length difference {len(input_ids_for_this_suffix)} vs {len(input_ids)}")
            #     print(f"Input Ids {input_ids_for_this_suffix} vs {input_ids}")
            #     print(f"Tokenizer Length difference {len(self.tokenizer(new_adv_suffix, add_special_tokens=False).input_ids)} vs {len(adv_suffix_token_ids)}")
            #     print(f"Suffix is {new_adv_suffix} vs prev {self.adv_suffix}")
            #     print(f"Tokenizer Input Ids {self.tokenizer(new_adv_suffix, add_special_tokens=False).input_ids}")
            #     print(f"Control toks are are {new_control_toks[i]}")
            #     cnt += 1
            # assert len(input_ids_for_this_suffix) == len(input_ids), "The length of the suffix must match the length of the original suffix"

            valid_candidate_strings.append(new_adv_suffix)
            valid_candidate_input_ids.append(input_ids_for_this_suffix)
            max_len_token_ids = max(max_len_token_ids, len(input_ids_for_this_suffix)) 

        # print(f"Adv suffix tokens are {adv_suffix_token_ids}")
        # print(f"cnt is {cnt}")
        # print(f"cnt2 is {cnt2}")

        assert len(valid_candidate_strings) > 0, "All candidate strings were invalid"

        batch_size = len(valid_candidate_strings)
        batch = torch.full((batch_size, max_len_token_ids), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)

        # make sure that the target string is located in the same place for all candidates
        for idx, candidate_input_ids in enumerate(valid_candidate_input_ids):
            # Pad each candidate input ids from the length to match the maximum length
            # print(f"Candidate input ids: {valid_candidate_input_ids}")
            # print("")
            batch[idx, -len(candidate_input_ids):] = candidate_input_ids

        return valid_candidate_strings, batch 

    def generate(self,
                prompt: str = None,
                max_new_tokens: int = 25,
                generation_config: GenerationConfig = None
    ) -> str:
        if prompt is None:
            prompt = f"{self.instruction} {self.adv_suffix}"

        if generation_config is None:
            generation_config = self.model.generation_config
            generation_config.max_new_tokens = max_new_tokens

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        attn_masks = torch.ones_like(input_ids).to(self.device)
        tokens = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids).to(self.device),
            generation_config=generation_config,
            pad_token_id=self.tokenizer.pad_token_id
        )[0]
        return self.tokenizer.decode(tokens[input_ids.shape[1]:])
