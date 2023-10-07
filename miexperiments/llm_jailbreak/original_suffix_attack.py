# %% Importing things
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, GPTNeoXForCausalLM, GenerationConfig
from fastchat.conversation import Conversation, get_conv_template
from livelossplot import PlotLosses
from jaxtyping import Int, Float
from typing import Tuple, List, Optional, Dict
import einops
import gc
from IPython.display import display, HTML

# %% Things in the original suffix attack code
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

# %% Tests
def test_shape(model, tokenizer, conv_template):
    assert False, "Test has to be updated"
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
