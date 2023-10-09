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
        batch_size: int=512,
        topk: int=256,
        temp: float=1.0,
        allow_non_ascii: bool=False,
    ):
        self.model = model
        self.tokenizer = tokenizer

        if isinstance(self.model, LlamaForCausalLM):
            conv_template_name = "llama-2"
            self.tokenizer.pad_token = tokenizer.unk_token
            self.tokenizer.padding_side = 'left'
        elif isinstance(self.model, GPTNeoXForCausalLM):
            conv_template_name = 'oasst_pythia'
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

        self.not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(self.tokenizer)

        self.device = self.model.device 
        print(f"Model device is {self.device}")

    def step(self):
        # let's check how much memory the current tensors are taking up
        assert not self.model.training, "Model must be in eval mode"

        # print_tensors_memory()
        # print(f"1 - Beginning of step {torch.cuda.memory_allocated(self.device)/1024/1024} MB")

        # get token ids for the initial adversarial string
        input_ids, suffix_slice, target_slice = self.get_input_ids_for_suffix(
            adv_suffix=self.adv_suffix,
        )

        input_ids = input_ids.to(self.device)

        # FIX 2) Are the input ids the same

        their_suffix_manager = SuffixManager(
            tokenizer=self.tokenizer,
            conv_template=self.conv_template,
            instruction=self.instruction,
            target=self.target,
            adv_string=self.adv_suffix
        )

        # print("")
        # print("FIX 2")
        # print(f"MY input ids: {input_ids}")
        # their_input_ids = their_suffix_manager.get_input_ids(adv_string=self.adv_suffix).to(self.device)
        # print(f"THEIR input ids: {their_input_ids}")
        # print(f"MY decoded input ids: {self.tokenizer.decode(input_ids, skip_special_tokens=False)}")
        # print(f"THEIR decoded input ids: {self.tokenizer.decode(their_input_ids, skip_special_tokens=False)}")

        assert target_slice.stop == len(input_ids), "Target slice test"

        token_gradients_ = self.token_gradients(input_ids, suffix_slice, target_slice)

        # FIX 3) Are the token gradients the same?

        their_token_gradients = token_gradients(
                        self.model,
                        their_input_ids,
                        their_suffix_manager._control_slice,
                        their_suffix_manager._target_slice,
                        their_suffix_manager._loss_slice
                    )

        # check if the tensors match
        same_grads =  torch.allclose(token_gradients_, their_token_gradients, atol=1e-4), "Token gradients test"
        # print("")
        # print("FIX 3")
        # print(f"Token gradients the same? {same_grads}")
        # print(f"MY token gradients shape {token_gradients_.shape}")
        # print(f"THEIR token gradients shape {their_token_gradients.shape}")
        # print(f"MY token gradients: {token_gradients_[0][:10]}")
        # print(f"THEIR token gradients: {their_token_gradients[0][:10]}")

        # print(f"8 - After token gradients {torch.cuda.memory_allocated(self.device)//1024//1024} MB")

        candidate_strings, batch_of_candidates = self.get_batch_of_suffix_candidates(
            input_ids,
            suffix_slice,
            token_gradients_,
            filter_by_size=True,
            random_seed=0 # to ensure reproducibility for the randint call
        )

        print("")
        print("FIX 4")
        print(f"MY candidate strings {candidate_strings[-10:]}")
        print(f"MY candidate strings {batch_of_candidates[-10:]}")

        del token_gradients_
        gc.collect()
        torch.cuda.empty_cache()

        # LLM-attacks code
        # Step 3.1 Slice the input to locate the adversarial suffix.
        adv_suffix_tokens = input_ids[their_suffix_manager._control_slice].to(self.device)

        # Step 3.2 Randomly sample a batch of replacements.
        new_adv_suffix_toks = sample_control(adv_suffix_tokens,
                    their_token_gradients,
                    self.batch_size,
                    topk=self.topk,
                    temp=self.temp,
                    not_allowed_tokens=self.not_allowed_tokens)

        print(f"THEIR New adv suffix toks before filtering {new_adv_suffix_toks[-10:]}")

        # Step 3.3 This step ensures all adversarial candidates have the same number of tokens.
        # This step is necessary because tokenizers are not invertible
        # so Encode(Decode(tokens)) may produce a different tokenization.
        # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
        new_adv_suffix = get_filtered_cands(self.tokenizer,
                                            new_adv_suffix_toks,
                                            filter_cand=False, # works with false
                                            curr_control=self.adv_suffix)

        # FIX 4) Are the candidate strings the same?
        print(f"THEIR num of new candidates {len(new_adv_suffix)}")
        print(f"MY candidate suffixes {candidate_strings[:10]}")        
        print(f"THEIR candidate suffixes {new_adv_suffix[:10]}")        
        
        assert new_adv_suffix == candidate_strings, "Candidate strings test"

        with torch.no_grad():
            # FIX 5) Are the logits the same?
            their_logits, their_ids = get_logits(model=model,
                                    tokenizer=tokenizer,
                                    input_ids=input_ids,
                                    control_slice=suffix_manager._control_slice,
                                    test_controls=new_adv_suffix,
                                    return_ids=True)

            losses = target_loss(their_logits, their_ids, their_suffix_manager._target_slice)

            print(f"MY first ids are {batch_of_candidates[0]}")
            print(f"THEIR first ids are {their_ids[0]}")

            # print(f"12 - After generating batch of candidates {torch.cuda.memory_allocated(self.device)//1024//1024} MB")

            # print(f"Shape of batch of candidates {batch_of_candidates.shape}")

            # let's check how much memory the current tensors are taking up
            # print_tensors_memory()

            # print(f"13 - After deleting token gradients {torch.cuda.memory_allocated(self.device)//1024//1024} MB")

            attention_mask = (batch_of_candidates != self.tokenizer.pad_token_id).type(batch_of_candidates.dtype)
            # print(f"14 - After creating attn mask {torch.cuda.memory_allocated(self.device)//1024//1024} MB")
            logits = self.model(input_ids=batch_of_candidates, attention_mask=attention_mask).logits

            same_logits = torch.allclose(logits, their_logits, atol=1e-4), "Logits test"

            print(f"Logits same? {same_logits}")
            print(f"MY logits first toks {logits[0][:10]}")
            print(f"THEIR logits first toks {their_logits[0][:10]}")

            # print(f"15 - After forward pass {torch.cuda.memory_allocated(self.device)//1024//1024} MB")
            target_len = target_slice.stop - target_slice.start
            target_slice = slice(batch_of_candidates.shape[1] - target_len, batch_of_candidates.shape[1])
            candidate_losses = self.get_loss(input_ids=batch_of_candidates, logits=logits, target_slice=target_slice)

            print(f"MY losses are {candidate_losses}")
            print(f"THEIR losses are {losses}")

            assert same_logits, "Logits must be the same"

            best_new_adv_suffix_idx = candidate_losses.argmin()
            self.adv_suffix = candidate_strings[best_new_adv_suffix_idx]

        del logits, batch_of_candidates, attention_mask
        gc.collect()
        torch.cuda.empty_cache()

    # returns the input tokens for the given suffix string
    # and also the slice within the tokens array containing the suffix and the target strings
    def get_input_ids_for_suffix(self, adv_suffix=None) -> Tuple[Int[Tensor, "seq_len"], slice, slice]:
        # if no suffix is given we just use the main one we got in the class
        if adv_suffix is None:
            adv_suffix = self.adv_suffix

        self.conv_template.messages = []

        self.conv_template.append_message(self.conv_template.roles[0], f"{self.instruction} {self.adv_suffix}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()

        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids

        self.conv_template.messages = []

        if self.conv_template.name == 'llama-2':

            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            goal_slice = slice(user_role_slice.stop, max(user_role_slice.stop, len(toks)))

            separator = ' ' if self.instruction else ''
            self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_suffix}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            suffix_slice = slice(goal_slice.stop, len(toks))

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            assistant_role_slice = slice(suffix_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            target_slice = slice(assistant_role_slice.stop, len(toks)-2)

        elif self.conv_template.name == "oasst_pythia":
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

        # labels = einops.repeat(
        #     input_ids[:,target_slice],
        #     "batch_size seq_len -> batch_size seq_len 1",
        #     batch_size=logits.shape[0]
        # ).to(self.device)

        # if logits.device != self.model.device:
        #     print(f"Logits device {logits.device}")
        #     print(f"Labels device {labels.device}")
        #     print(f"Model device {self.model.device}")

        # target_logprobs = torch.gather(
        #     logits[:,loss_slice].log_softmax(dim=-1),
        #     dim=-1,
        #     index=labels
        # ).squeeze(-1) # we have one extra dimension at the end we don't need

        # return -target_logprobs.mean(dim=-1)

        batch_size = logits.shape[0]

        loss = nn.CrossEntropyLoss(reduction='none')(
            einops.rearrange(logits[:,loss_slice], "batch_size seq_len d_vocab -> (batch_size seq_len) d_vocab"),
            einops.rearrange(input_ids[:,target_slice], "batch_size seq_len -> (batch_size seq_len)")
        )

        loss = einops.reduce(loss, "(batch_size seq_len) -> batch_size", batch_size=batch_size, reduction='mean')

        return loss

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
        batch = input_ids.unsqueeze(0).to(self.device)
        attention_mask = (batch != self.tokenizer.pad_token_id).type(batch.dtype).to(self.device)
        logits = self.model(input_ids=batch, attention_mask=attention_mask).logits.to(self.device)
        return self.get_loss(input_ids=batch, logits=logits, target_slice=target_slice)[0].item()

    def get_loss_for_input_ids(self, input_ids: Int[Tensor, "seq_len"], target_slice: slice) -> Float:
        batch = input_ids.unsqueeze(0).to(self.device)
        attention_mask = (batch != self.tokenizer.pad_token_id).type(batch.dtype).to(self.device)
        logits = self.model(input_ids=batch, attention_mask=attention_mask).logits.to(self.device)
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
            embeds = self.model.model.embed_tokens(input_ids.unsqueeze(0).to(self.device)).detach()
        elif isinstance(self.model, GPTNeoXForCausalLM):
            embed_weights = self.model.base_model.embed_in.weight
            embeds = self.model.base_model.embed_in(input_ids.unsqueeze(0).to(self.device)).detach()
        else:
            assert False, "Only Llama and GPTNeoX models are supported"

        # print(f"2 - After computing embeddings {torch.cuda.memory_allocated(self.device)//1024//1024} MB")

        one_hot = torch.zeros(
            input_ids[suffix_slice].shape[0],
            embed_weights.shape[0],
            dtype=embed_weights.dtype
        ).to(self.device)

        # print(f"3 - After creating one_hot {torch.cuda.memory_allocated(self.device)//1024//1024} MB")

        # maybe use scatter here?
        one_hot[torch.arange(input_ids[suffix_slice].shape[0]),input_ids[suffix_slice]] = 1.0

        # this returns an error when run on MPS
        #one_hot = torch.zeros(
        #    input_ids[input_slice].shape[0],
        #    embed_weights.shape[0],
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

        # print(f"3 - After creating one_hot {torch.cuda.memory_allocated(self.device)//1024//1024} MB")

        logits = self.model(inputs_embeds=full_embeds).logits.to(self.device)

        # print(f"4 - After the forward pass {torch.cuda.memory_allocated(self.device)//1024//1024} MB")

        # we add one extra dimension at the beginning to the input token ids
        batch = input_ids.unsqueeze(0).to(self.device)
        loss = self.get_loss(input_ids=batch, logits=logits, target_slice=target_slice)

        loss.backward()

        # print(f"5 - After backward pass {torch.cuda.memory_allocated(self.device)//1024//1024} MB")

        grad = one_hot.grad.clone()
        grad = grad / grad.norm(dim=-1, keepdim=True) # why do this?

        # print(f"6 - After cloning grad {torch.cuda.memory_allocated(self.device)//1024//1024} MB")

        self.model.zero_grad()
        del logits, batch, full_embeds, one_hot, embeds, embed_weights, loss; gc.collect()
        torch.cuda.empty_cache()

        # print(f"7 - After deleting many things {torch.cuda.memory_allocated(self.device)//1024//1024} MB")

        return grad


    # selects k tokens within the topk (allowed) tokens, replaces them and returns a batch
    # we can feed to the model
    def get_batch_of_suffix_candidates(
        self,
        input_ids: Int[Tensor, "seq_len"],
        adv_suffix_slice: slice,
        grad: Float[Tensor, "suffix_len d_vocab"], # what does this batch_size represent?
        filter_by_size=True,
        random_seed=None
    ) -> Tuple[List[str], Int[Tensor, "n_cands seq_len"]]:
        adv_suffix_token_ids = input_ids[adv_suffix_slice]

        assert len(adv_suffix_token_ids.shape) == 1, "The adv suffix array must be 1-dimensional at the moment"
        assert len(adv_suffix_token_ids) == grad.shape[0], "The length of the suffix (in tokens) must match the length of the gradient"

        if self.not_allowed_tokens is not None:
            grad[:, self.not_allowed_tokens.to(grad.device)] = np.infty

        top_indices = (-grad).topk(self.topk, dim=1).indices
        adv_suffix_token_ids = adv_suffix_token_ids.to(grad.device)

        original_control_toks = adv_suffix_token_ids.repeat(self.batch_size, 1)
        print(f"Shape of original control toks {original_control_toks.shape}")

        # print(f"9 - After creating original control toks {torch.cuda.memory_allocated(self.device)//1024//1024} MB")

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

        print(f"MY New control toks {new_control_toks[-10:]}")

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

        print(f"MY New adv strings {new_adv_strings}")

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
                continue
            # if filter_by_size == True and len(input_ids_for_this_suffix) != len(input_ids):
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

        # print(f"10 - After filtering candidate strings {torch.cuda.memory_allocated(self.device)//1024//1024} MB")

        batch_size = len(valid_candidate_strings)
        assert batch_size == self.batch_size, "Number of candidates equals the batch size"
        batch = torch.full((batch_size, max_len_token_ids), self.tokenizer.pad_token, dtype=torch.long, device=self.device)
        

        # print(f"11 - After creating batch {torch.cuda.memory_allocated(self.device)//1024//1024} MB")

        # make sure that the target string is located in the same place for all candidates
        for idx, candidate_input_ids in enumerate(valid_candidate_input_ids):
            # Pad each candidate input ids from the length to match the maximum length
            batch[idx, -len(candidate_input_ids):] = candidate_input_ids

        return valid_candidate_strings, batch 

