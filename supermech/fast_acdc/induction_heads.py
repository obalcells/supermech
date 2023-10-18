# %% Importing stuff
import os
import sys
import plotly.express as px
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import functools
from tqdm import tqdm
from IPython.display import display
import webbrowser
import gdown
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import circuitsvis as cv
import supermech
from supermech.utils.utils import imshow, hist, plot_comp_scores, plot_logit_attribution, plot_loss_difference

# this doesn't seem to work??
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# the path where all the files are stored for doing experiments
ROOT_PATH = Path("/users/oscarbalcells/Desktop/mech_int_experiments/supermech").resolve()

MAIN = __name__ == "__main__"

# %%
def get_induction_heads_model(weights_path: Path=Path(f"{ROOT_PATH}/models/attn_only_2L_half.pth")):
    print(f"Weights path is {str(weights_path)}")

    if not weights_path.exists():
        print("Downloading weights from google drive")
        url = "https://drive.google.com/uc?id=1vcZLJnJoYKQs-2KOjkd6LvHZrkSdoxhu"
        output = str(weights_path)
        gdown.download(url, output)

    assert weights_path.exists(), "Weights path still doesn't exist"

    cfg = HookedTransformerConfig(
            d_model=768,
            d_head=64,
            n_heads=12,
            n_layers=2,
            n_ctx=2048,
            d_vocab=50278,
            attention_dir="causal",
            attn_only=True, # defaults to False
            tokenizer_name="EleutherAI/gpt-neox-20b", 
            seed=398,
            use_attn_result=True,
            normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
            positional_embedding_type="shortformer"
    )

    model = HookedTransformer(cfg).to(device)
    pretrained_weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(pretrained_weights)

    return model

# get the model and test it out on a prompt
model = get_induction_heads_model()
text = "We think that powerful, significantly superhuman machine "
with torch.no_grad():
    output_logits = model(text, return_type="logits")
    print(model.generate(text, max_new_tokens=10))

# %% Generate repeated token dataset

# the "clean" dataset
def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
    '''
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    prefix = (torch.ones(batch, 1, device=device) * model.tokenizer.bos_token_id).long()
    rep_tokens = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), device=device, dtype=torch.long)
    rep_tokens = torch.cat([prefix, rep_tokens, rep_tokens], dim=-1)
    print(f"Rep tokens shape {rep_tokens.shape}")
    return rep_tokens

seq_len = 20
batch = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch)

# %%

seq_len = 50
batch = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch)


# %%

# We make a tensor to store the induction score for each head.
# We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)


def induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    '''
    Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
    '''
    
    # other way:
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
    induction_scores = einops.reduce(induction_stripe, "batch head_index position -> head_index", "mean")
    layer = hook.layer()
    induction_score_store[layer, :] = induction_scores

pattern_hook_names_filter = lambda name: name.endswith("pattern")

# Run with hooks (this is where we write to the `induction_score_store` tensor`)
model.run_with_hooks(
    rep_tokens_10, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        pattern_hook_names_filter,
        induction_score_hook
    )]
)

# Plot the induction scores for each head in each layer
imshow(
    induction_score_store, 
    labels={"x": "Head", "y": "Layer"}, 
    title="Induction Score by Head", 
    text_auto=".2f",
    width=900, height=400
)