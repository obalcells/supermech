import torch
from torch import Tensor
from jaxtyping import Int, Float
from typing import Tuple, List, Optional, Dict
import os

class ActivationVectors:
    def __init__(self):
        self.vectors_path = os.path.dirname(os.path.abspath(__file__))

        self.refusal_vector_layers = list(range(10, 20))
        self.resid_stream_refusal_vectors = {} 

        for layer in self.refusal_vector_layers:
            self.resid_stream_refusal_vectors[layer] = torch.load(f"{self.vectors_path}/nina_refusal_vectors/vec_layer_{layer}.pt")

    def get_resid_stream_refusal_vector(self, layer:int) -> Float[Tensor, "d_model"]:
        assert layer in self.refusal_vector_layers, f"Refusal vector for layer {layer} isn't available"
        return self.resid_stream_refusal_vectors[layer]

