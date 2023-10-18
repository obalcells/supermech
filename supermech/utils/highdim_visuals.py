import torch as t
from torch import Tensor
from typing import List, Union, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from transformer_lens import HookedTransformer
from transformer_lens.utils import to_numpy
from typing import Dict
import pandas as pd
from jaxtyping import Float, Int
import einops
from typing import Tuple, List, Optional, Dict
from torch.nn.functional import normalize
from typing import Tuple, List, Optional, Dict
from jaxtyping import Float, Int
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

def save_activation_projection_tsne(
    activations1,
    activations2,
    fname,
    title,
    label1="Positive Examples",
    label2="Negative Examples",
):
    """
    activations1: n_samples x vector dim tensor
    activations2: n_samples x vector dim tensor

    projects to n_samples x 2 dim tensor using t-SNE (over the full dataset of both activations 1 and 2) and saves visualization.
    Colors projected activations1 as blue and projected activations2 as red.
    """
    plt.clf()
    activations = torch.cat([activations1, activations2], dim=0)
    activations_np = activations.cpu().numpy()

    # t-SNE transformation
    tsne = TSNE(n_components=2)
    projected_activations = tsne.fit_transform(activations_np)

    # Splitting back into activations1 and activations2
    activations1_projected = projected_activations[: activations1.shape[0]]
    activations2_projected = projected_activations[activations1.shape[0] :]

    # Visualization
    for x, y in activations1_projected:
        plt.scatter(x, y, color="blue", marker="o", alpha=0.4)

    for x, y in activations2_projected:
        plt.scatter(x, y, color="red", marker="o", alpha=0.4)

    # Adding the legend
    scatter1 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="blue",
        markersize=10,
        label=label1,
    )
    scatter2 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="red",
        markersize=10,
        label=label2,
    )

    plt.legend(handles=[scatter1, scatter2])
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig(fname)