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
