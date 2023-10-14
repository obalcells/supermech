import torch
import torch.nn as nn
import numpy as np
import os
import json
from IPython.display import display, HTML
from torch import Tensor
from jaxtyping import Int, Float
from typing import Tuple, List, Optional, Dict

# things for the suffix attack
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, GPTNeoXForCausalLM, GenerationConfig
from fastchat.conversation import get_conv_template
import einops
import gc
import time

from llama_multiprompt_jailbreak import AttackPrompt, PromptManager

class PromptManager():
    def batch_loader(self, control_cand):
        # this is the same across all the prompts more or less
        max_len = self._control_slice.stop - self._control_slice.start

        test_ids = [
            torch.tensor(self.tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
            for control in test_controls
        ]

        pad_tok = 0
        while pad_tok in self.input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1

        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))

        locs = torch.arange(self._control_slice.start, self._control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
        ids = torch.scatter(
            self.input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
            1,
            locs,
            test_ids
        )
        if pad_tok >= 0:
            attn_mask = (ids != pad_tok).type(ids.dtype)
        else:
            attn_mask = None



# tests that our custom batching technique works
def test_batching():

    self.prompt = PromptManager(
        goals,
        targets,
        self.tokenizer,
        self.conv_template,
        control_init,
        test_prefixes,
        # managers
    )
