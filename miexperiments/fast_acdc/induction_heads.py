from transformer_lens import utils, HookedTransformer, HookedTransformerConfig
import torch as t
def get_model(weights_dir="./models/attn_only_2L_half.pth"):
    model = HookedTransformer(cfg)
    pretrained_weights = t.load(weights_dir, map_location=device)
    model.load_state_dict(pretrained_weights)
    return model