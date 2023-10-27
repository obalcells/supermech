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
import transformer_lens as tl
import circuitsvis as cv
import supermech
from supermech.utils.plotly_utils import imshow, hist, plot_comp_scores, plot_logit_attribution, plot_loss_difference
import matplotlib.pyplot as plt
import networkx as nx
from torchtyping import TensorType as TT
from typing import List, Union, Optional, Callable
import gc

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = '1' 


# %% Loading the model

# weights_path = "../models/attn_only_2L_half.pth"
# weights_url = "https://drive.google.com/uc?id=1vcZLJnJoYKQs-2KOjkd6LvHZrkSdoxhu"

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained('redwood_attn_2l', map_location=device)
model = model.to('cpu')
model.set_use_attn_result(True)
model.set_use_hook_mlp_in(True)
# model = HookedTransformer(cfg).to(device)
# pretrained_weights = torch.load(weights_path, map_location=device)
# model.load_state_dict(pretrained_weights)

with torch.no_grad():
    print(model.generate("ABCAB", max_new_tokens=10))

# %% Generate repeated token dataset

# we generate the "clean" and the "corrupted" datasets 
def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
    # full_seq_len = 1 + 2 * seq_len
    prefix = (torch.ones(batch, 1, device=device) * model.tokenizer.bos_token_id).long()
    rep_tokens = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), device=device, dtype=torch.long)
    rep_tokens = torch.cat([prefix, rep_tokens, rep_tokens], dim=-1)
    print(f"Rep tokens shape {rep_tokens.shape}")
    return rep_tokens

def generate_corrupted_tokens(
    model: HookedTransformer, clean_tokens: Int[Tensor, "batch full_seq_len"],
) -> Int[Tensor, "batch full_seq_len"]:
    # full_seq_len = 1 + 2 * seq_len
    batch = clean_tokens.shape[0]
    seq_len = (clean_tokens.shape[1] - 1) // 2
    random_tokens = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), device=device, dtype=torch.long)
    corrupted_tokens = rep_tokens.clone()
    corrupted_tokens[:, 1+seq_len:] = random_tokens 
    return corrupted_tokens 

seq_len = 50
batch = 10
clean_tokens = generate_repeated_tokens(model, seq_len, batch)
# corrupted_tokens = generate_corrupted_tokens(model, clean_tokens)
corrupted_tokens = generate_repeated_tokens(model, seq_len, batch)


# %% Computing the loss for the different token positions

def show_loss(model, tokens, labels):
    with torch.no_grad():
        logits = model(tokens, return_type="logits")
        logprobs = logits.to(torch.float64).log_softmax(dim=-1)
        pred_logprobs = torch.gather(
            logprobs,
            index=labels,
            # index=tokens[:, 1:].unsqueeze(-1),
            dim=-1
        )
        loss = -torch.mean(pred_logprobs, dim=-1)
        imshow(loss)

# labels = clean_tokens[:, 1:].unsqueeze(-1)
labels = corrupted_tokens[:, 1:].unsqueeze(-1)
show_loss(model, clean_tokens, labels)
show_loss(model, corrupted_tokens, labels)

# %% Induction scores

induction_score_store = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
def induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
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

# %% What are the heads that attend to the previous token?

with torch.no_grad():
    logits = model(rep_tokens, return_type="logits")
    logprobs = logits.to(torch.float64).log_softmax(dim=-1)
    print(rep_tokens[:, 1:].unsqueeze(-1).shape)
    pred_logprobs = torch.gather(
        logprobs,
        index=rep_tokens[:, 1:].unsqueeze(-1),
        dim=-1
    )
    print(pred_logprobs[0])
    print(pred_logprobs.shape)
    print(pred_logprobs.shape)
    loss = torch.mean(pred_logprobs, dim=-1)
    imshow(loss)

# %% Induction scores

prev_token_score = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
def prev_token_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    layer = hook.layer()
    prev_token_score[layer,:] = pattern.diagonal(dim1=-2, dim2=-1, offset=-1).mean(dim=-1).mean(dim=0)

pattern_hook_names_filter = lambda name: name.endswith("pattern")

# Run with hooks (this is where we write to the `induction_score_store` tensor`)
model.run_with_hooks(
    rep_tokens, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        pattern_hook_names_filter,
        prev_token_hook 
    )]
)

# Plot the induction scores for each head in each layer
imshow(
    prev_token_score, 
    labels={"x": "Head", "y": "Layer"}, 
    title="Previous token score by Head", 
    text_auto=".2f",
    width=900, height=400
)

# %%

G=nx.Graph()
# Add nodes and edges
G.add_edge("L 0 H 0", "L 1 H 5")
G.add_edge("L 0 H 0", "L 1 H 6")
nx.draw(G, with_labels=True)
plt.savefig('labels.png')

# %%

logits, cache = model.run_with_cache(
    rep_tokens,
    return_type="logits"
)

# %% Creating the adjacency matrix

# Creating the adjacency matrix and filling it
str2index = {}
index2str = []
pos2index = []

for layer in range(model.cfg.n_layers):
    pos2index.append([])
    for head in range(model.cfg.n_heads):
        str2index[f"L {layer} H {head}"] = len(index2str)
        pos2index[-1].append(len(index2str))
        index2str.append(f"L {layer} H {head}")

n_nodes = len(index2str)
adj_matrix = np.zeros((n_nodes, n_nodes))
# torch.zeros((n_nodes, n_nodes), device=model.cfg.device)

for layer_from in range(model.cfg.n_layers):
    for head_from in range(model.cfg.n_heads):
        index_from = pos2index[layer_from][head_from]
        for layer_to in range(model.cfg.n_layers)[layer_from:]:
            for head_to in range(model.cfg.n_heads):
                index_to = pos2index[layer_to][head_to]
                adj_matrix[layer_from][layer_to] = 1

# %%

def get_logit_diff(logits, tokens_1=clean_tokens, tokens_2=corrupted_tokens):
    correct_token_logits = torch.gather(
        logits,
        index=tokens_1[:, 1:].unsqueeze(-1),
        dim=-1
    )

    corrupted_token_logits = torch.gather(
        logits,
        index=tokens_2[:, 1:].unsqueeze(-1),
        dim=-1
    )

    return (correct_token_logits[:,-seq_len:,0] - corrupted_token_logits[:,-seq_len:,0]).mean(dim=-1).mean(dim=0)

clean_logits, clean_cache = model.run_with_cache(clean_tokens, return_type="logits")
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens, return_type="logits")

CLEAN_BASELINE = get_logit_diff(clean_logits, tokens_1=clean_tokens, tokens_2=corrupted_tokens)
CORRUPTED_BASELINE = get_logit_diff(corrupted_logits, tokens_1=clean_tokens, tokens_2=corrupted_tokens)

# %%

print(f"Clean baseline: {CLEAN_BASELINE}")
print(f"Corrupted baseline: {CORRUPTED_BASELINE}")

# %%

Metric = Callable[[TT["batch_and_pos_dims", "d_model"]], float]

def induction_metric(logits: TT["batch_and_pos_dims", "d_model"], answer_tokens=clean_tokens):
    rep_token_logits_sum = torch.gather(
        logits,
        index=answer_tokens[:, 1:].unsqueeze(-1),
        dim=-1
    )[:,-seq_len:,0].mean(dim=-1).mean(dim=0)

    metric_score = (rep_token_logits_sum - CORRUPTED_BASELINE) / (CLEAN_BASELINE - CORRUPTED_BASELINE)
    return metric_score

# %%

model.zero_grad()
model.reset_hooks()
# del clean_value, clean_cache, clean_grad_cache
gc.collect()
torch.cpu.empty_cache()

# %%

CORRUPTED_BASELINE = CORRUPTED_BASELINE.detach()
CLEAN_BASELINE = CLEAN_BASELINE.detach()

# %% Cache

# filter_not_qkv_input = lambda name: "_input" not in name
hook_filter = lambda name: True or name.endswith("ln1.hook_normalized") or name.endswith("attn.hook_result")
def get_cache_fwd_and_bwd(model, tokens, metric):
    model.reset_hooks()
    cache = {}
    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()
    model.add_hook(hook_filter, forward_cache_hook, "fwd")

    grad_cache = {}
    def backward_cache_hook(act, hook):
        print(f"A In backward hook {hook.name}")
        grad_cache[hook.name] = act.detach()
        print(f"B In backward hook {hook.name}")
    model.add_hook(hook_filter, backward_cache_hook, "bwd")

    model.zero_grad()
    value = metric(model(tokens))
    value.backward()
    model.zero_grad()
    cache = ActivationCache({}, model)
    grad_cache = ActivationCache({}, model)
    return value.item(), cache, grad_cache

# %% Clean cache

clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(model, clean_tokens, induction_metric)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(clean_cache))

    # %% Corrupted cache

corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(model, corrupted_tokens, induction_metric)
print("Corrupted Value:", corrupted_value)
print("Corrupted Activations Cached:", len(corrupted_cache))
print("Corrupted Gradients Cached:", len(corrupted_grad_cache))
print("Clean Gradients Cached:", len(clean_grad_cache))

# %% Corrupted cache

corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(model, corrupted_tokens, induction_metric)
print("Corrupted Value:", corrupted_value)
print("Corrupted Activations Cached:", len(corrupted_cache))
print("Corrupted Gradients Cached:", len(corrupted_grad_cache))

# %%
