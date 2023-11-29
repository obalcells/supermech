import torch
import gc
import numpy as np
from transformer_lens import HookedTransformer, ActivationCache
from jaxtyping import Float, Bool
from typing import Callable, Tuple, Union, Dict, Optional
import einops

class ModelWrapper:
    def __init__(self, model):
        self.model = model

        self.model.set_use_hook_mlp_in(True)
        self.model.set_use_split_qkv_input(True)
        self.model.set_use_attn_result(True)

        self.n_layers = self.model.cfg.n_layers
        self.n_heads = self.model.cfg.n_heads
        self.d_model = self.model.cfg.d_model
        self.dtype = self.model.cfg.dtype
        self.device = self.model.cfg.device

        self.valid_upstream_node_types = ["resid_pre", "mlp", "head"]
        self.valid_downstream_node_types = ["resid_post", "mlp", "head"]

        self.valid_upstream_hook_types = ["hook_result", "hook_resid_pre", "hook_mlp_out"]
        self.valid_downstream_hook_types = ["hook_q_input", "hook_k_input", "hook_v_input", "hook_mlp_in", "hook_resid_post"]

        self.upstream_nodes = []
        self.downstream_nodes = []

        self.upstream_hook_index: Dict[str, slice] = {} 
        self.downstream_hook_index: Dict[str, slice] = {}
        self.upstream_before_layer: Dict[int, slice] = {}

        self.upstream_node_name_to_index = {}
        self.downstream_node_name_to_index = {}
        
    def setup_edges(self, upstream_nodes=None, downstream_nodes=None):

        # we first reset all the graph-related attributes
        self.upstream_hook_index: Dict[str, slice] = {} 
        self.downstream_hook_index: Dict[str, slice] = {}
        self.upstream_before_layer: Dict[int, slice] = {}

        self.upstream_node_name_to_index = {}
        self.downstream_node_name_to_index = {}

        self.n_upstream_nodes = 0
        self.n_downstream_nodes = 0 

        self.upstream_nodes = []
        self.downstream_nodes = []

        # if no nodes are specified, we assume that all of them will be used
        if upstream_nodes is None:
            upstream_nodes = self.valid_upstream_node_types.copy()
        
        if downstream_nodes is None:
            downstream_nodes = self.valid_downstream_node_types.copy()

        # we can assume that the two lists of hooks are sorted by layer number
        self.upstream_hooks, self.downstream_hooks = self.get_hooks_from_nodes(upstream_nodes, downstream_nodes)

        upstream_node_index = 0

        for hook_name in self.upstream_hooks:
            layer = int(hook_name.split(".")[1])
            hook_type = hook_name.split(".")[-1]
            
            # we store the slice of all upstream hooks previous to this layer
            if layer not in self.upstream_before_layer:
                # we must check previous layers too because we might have skipped some
                # if our upstream nodes are layer specific
                for earlier_layer in range(0, layer + 1):
                    if earlier_layer not in self.upstream_before_layer:
                        self.upstream_before_layer[earlier_layer] = slice(0, upstream_node_index)

            if hook_type == "hook_result":
                for head_idx in range(self.model.cfg.n_heads):
                    self.upstream_nodes.append(f"head.{layer}.{head_idx}")
                    self.upstream_node_name_to_index[f"head.{layer}.{head_idx}"] = upstream_node_index + head_idx
                self.upstream_hook_index[hook_name] = slice(upstream_node_index, upstream_node_index + self.n_heads)
                upstream_node_index += self.n_heads 
            elif hook_type == "hook_resid_pre":
                self.upstream_nodes.append(f"resid_pre.{layer}")
                self.upstream_node_name_to_index[f"resid_pre.{layer}"] = upstream_node_index
                self.upstream_hook_index[hook_name] = slice(upstream_node_index, upstream_node_index + 1)
                upstream_node_index += 1
            elif hook_type == "hook_mlp_out":
                self.upstream_nodes.append(f"mlp.{layer}")
                self.upstream_node_name_to_index[f"mlp.{layer}"] = upstream_node_index
                self.upstream_hook_index[hook_name] = slice(upstream_node_index, upstream_node_index + 1)
                upstream_node_index += 1
            else:
                raise NotImplementedError("Invalid upstream hook type")

        # we store the slice of all upstream hooks indices previous to the last layer
        for layer in range(0, self.n_layers):
            if layer not in self.upstream_before_layer:
                self.upstream_before_layer[layer] = slice(0, upstream_node_index)

        downstream_node_index = 0

        for hook_name in self.downstream_hooks:
            layer = int(hook_name.split(".")[1])
            hook_type = hook_name.split(".")[-1]

            if hook_type == "hook_q_input" or hook_type == "hook_k_input" or hook_type == "hook_v_input":
                letter = hook_type.split("_")[1].lower()
                for head_idx in range(self.model.cfg.n_heads):
                    self.downstream_nodes.append(f"head.{layer}.{head_idx}.{letter}")
                    self.downstream_node_name_to_index[f"head.{layer}.{head_idx}.{letter}"] = downstream_node_index + head_idx
                self.downstream_hook_index[hook_name] = slice(downstream_node_index, downstream_node_index + self.n_heads)
                downstream_node_index += self.n_heads 
            elif hook_type == "hook_resid_post":
                self.downstream_nodes.append(f"resid_post.{layer}")
                self.downstream_node_name_to_index[f"resid_post.{layer}"] = downstream_node_index
                self.downstream_hook_index[hook_name] = slice(downstream_node_index, downstream_node_index + 1)
                downstream_node_index += 1
            elif hook_type == "hook_mlp_in":
                self.downstream_nodes.append(f"mlp.{layer}")
                self.downstream_node_name_to_index[f"mlp.{layer}"] = downstream_node_index
                self.downstream_hook_index[hook_name] = slice(downstream_node_index, downstream_node_index + 1)
                downstream_node_index += 1
            else:
                raise NotImplementedError("Invalid upstream hook type")

        self.n_upstream_nodes = len(self.upstream_nodes)
        self.n_downstream_nodes = len(self.downstream_nodes)

        activations_tensor_in_gb = self.n_upstream_nodes * self.d_model * self.dtype.itemsize / 2**30 
        print(f"Saving activations requires {activations_tensor_in_gb:.4f} GB of memory per token")

    def run_eap(self, clean_tokens, corrupted_tokens, metric, upstream_nodes=None, downstream_nodes=None):
        self.setup_edges(upstream_nodes, downstream_nodes)
        
        self.eap_scores = torch.zeros((self.n_upstream_nodes, self.n_downstream_nodes), device=self.device)

        seq_len = clean_tokens.shape[1]
        batch_size = clean_tokens.shape[0]

        assert seq_len == corrupted_tokens.shape[1], "Sequence length mismatch"
        assert batch_size == corrupted_tokens.shape[0], "Batch size mismatch"

        activations_diff_in_gb = batch_size * seq_len * self.n_upstream_nodes * self.d_model * self.dtype.itemsize / 2**30 
        memory_allocated_in_gb = torch.cuda.memory_allocated() / 2**30
        total_memory_in_gb = torch.cuda.get_device_properties(self.device).total_memory / 2**30
        print(f"Saving activation differences requires {activations_diff_in_gb:.2f} GB of memory")
        assert activations_diff_in_gb < total_memory_in_gb - memory_allocated_in_gb, "Not enough memory to store activation differences"

        self.model.reset_hooks()
        gc.collect()
        torch.cuda.empty_cache()

        upstream_activations_difference = torch.zeros((batch_size, seq_len, self.n_upstream_nodes, self.d_model), device=self.device, dtype=self.dtype, requires_grad=False)

        memory_allocated_in_gb = torch.cuda.memory_allocated() / 2**30
        print(f"Total memory allocated after creating activation differences tensor is {memory_allocated_in_gb:.2f} GB out of {total_memory_in_gb:.2f} GB")

        def corrupted_forward_hook(activations, hook):
            hook_slice = self.upstream_hook_index[hook.name]
            if activations.ndim == 3:
                # we are in the case of a residual layer or MLP
                # activations have shape [batch_size, seq_len, d_model]
                # we need to add a dimension to make it [batch_size, seq_len, 1, d_model]
                # the hook slice is a slice of size 1
                upstream_activations_difference[:, :, hook_slice, :] = -activations.unsqueeze(-2)
            elif activations.ndim == 4:
                # we are in the case of an attention layer
                # activations have shape [batch_size, seq_len, n_heads, d_model]
                upstream_activations_difference[:, :, hook_slice, :] = -activations

        def clean_forward_hook(activations, hook):
            hook_slice = self.upstream_hook_index[hook.name]
            if activations.ndim == 3:
                upstream_activations_difference[:, :, hook_slice, :] += activations.unsqueeze(-2)
            elif activations.ndim == 4:
                upstream_activations_difference[:, :, hook_slice, :] += activations

        def backward_hook(grad, hook):
            downstream_hook_slice = self.downstream_hook_index[hook.name]

            # we get the slice of all upstream activation differences previous to this layer
            layer = hook.layer()
            earlier_layer_hooks_slice = self.upstream_before_layer[layer]

            # grad has shape [batch_size, seq_len, n_heads, d_model] or [batch_size, seq_len, d_model]
            # we want to multiply it by the upstream activations difference
            if grad.ndim == 3:
                grad_expanded = grad.unsqueeze(-2)  # Shape: [batch_size, seq_len, 1, d_model]
            else:
                grad_expanded = grad  # Shape: [batch_size, seq_len, n_heads, d_model]

            # we compute the mean over the batch_size and seq_len dimensions
            result = torch.matmul(
                upstream_activations_difference[:, :, earlier_layer_hooks_slice],
                grad_expanded.transpose(-1, -2)
            ).sum(dim=0).sum(dim=0) # we sum over the batch_size and seq_len dimensions

            self.eap_scores[earlier_layer_hooks_slice, downstream_hook_slice] = result 

        upstream_hook_filter = lambda name: name.endswith(tuple(self.upstream_hooks))
        downstream_hook_filter = lambda name: name.endswith(tuple(self.downstream_hooks))

        # we first perform a forward pass on the corrupted input 
        self.model.add_hook(upstream_hook_filter, corrupted_forward_hook, "fwd")

        # we don't need gradients for this forward pass
        # we'll take the gradients when we perform the forward pass on the clean input
        with torch.no_grad(): 
            corrupted_tokens = corrupted_tokens.to(self.device)
            self.model(corrupted_tokens, return_type=None)        

        # now we perform a forward and backward pass on the clean input
        self.model.reset_hooks()
        self.model.add_hook(upstream_hook_filter, clean_forward_hook, "fwd")
        self.model.add_hook(downstream_hook_filter, backward_hook, "bwd")

        clean_tokens = clean_tokens.to(self.device)
        torch.cuda.empty_cache()
        value = metric(self.model(clean_tokens, return_type="logits"))
        value.backward()
        
        # we delete the activations difference tensor to free up memory
        self.model.zero_grad()
        del upstream_activations_difference
        gc.collect()
        torch.cuda.empty_cache()

        self.eap_scores = self.eap_scores.cpu()

        return self.eap_scores

    def top_edges(self, eap_scores=None, n=10, abs=True):
        if eap_scores is None:
            eap_scores = self.eap_scores

        # get indices of maximum values in 2d tensor
        if abs:
            top_scores, top_indices = torch.topk(eap_scores.flatten().abs(), k=n, dim=0)
        else:
            top_scores, top_indices = torch.topk(eap_scores.flatten(), k=n, dim=0)

        top_edges = []
        print("\nTop edges:")
        for i, (abs_score, index) in enumerate(zip(top_scores, top_indices)):
            upstream_node_idx, downstream_node_idx = np.unravel_index(index, eap_scores.shape)
            score = eap_scores[upstream_node_idx, downstream_node_idx]
            top_edges.append((self.upstream_nodes[upstream_node_idx], self.downstream_nodes[downstream_node_idx], score.item()))
            if i <= 10: 
                # we only print the first 10
                print(f"{score:.4f}\t{self.upstream_nodes[upstream_node_idx]} -> {self.downstream_nodes[downstream_node_idx]}")

        return top_edges
    
    # given a set of upstream nodes and downstream nodes, this function returns the corresponding hooks
    # to access the activations of these nodes
    # we return the list of hooks sorted by layer number
    def get_hooks_from_nodes(self, upstream_nodes, downstream_nodes):

        # we first check that the types of the nodes passed are valid
        for node in upstream_nodes:
            node_type = node.split(".")[0] # 'resid_pre', 'mlp' or 'head'
            assert node_type in self.valid_upstream_node_types, "Invalid upstream node"

        for node in downstream_nodes:
            node_type = node.split(".")[0] # 'resid_post', 'mlp' or 'head'
            assert node_type in self.valid_downstream_node_types, "Invalid downstream node"

        upstream_hooks = []
        downstream_hooks = []

        for node in upstream_nodes:
            node_is_layer_specific = (len(node.split(".")) > 1)
            if not node_is_layer_specific:
                # we are in the case of a global node that applies to all layers
                hook_type = "hook_resid_pre" if node == "resid_pre" else "hook_mlp_out" if node == "mlp" else "attn.hook_result"
                for layer in range(self.n_layers):
                    upstream_hooks.append(f"blocks.{layer}.{hook_type}")
            else:
                assert node.split(".")[1].isdigit(), "Layer number must be an integer"
                layer = int(node.split(".")[1])

                hook_type = "hook_resid_pre" if node == "resid_pre" else "hook_mlp_out" if node == "mlp" else "attn.hook_result"
                upstream_hooks.append(f"blocks.{layer}.{hook_type}")

        for node in downstream_nodes:
            node_is_layer_specific = (len(node.split(".")) > 1)
            if not node_is_layer_specific:
                # we are in the case of a global node that applies to all layers
                if node == "head":
                    for layer in range(self.n_layers):
                        for letter in "qkv":
                            downstream_hooks.append(f"blocks.{layer}.hook_{letter}_input")
                elif node == "resid_post" or node == "mlp":
                    hook_type = "hook_resid_post" if node == "resid_post" else "hook_mlp_in"
                    for layer in range(self.n_layers):
                        downstream_hooks.append(f"blocks.{layer}.{hook_type}")
                else:
                    raise NotImplementedError("Invalid downstream node")
            else:
                # we are in the case of a node specified for a single layer
                assert node.split(".")[1].isdigit(), "Layer number must be an integer"
                layer = int(node.split(".")[1])

                if node.startswith("resid_post") or node.startswith("mlp"):
                    hook_type = "hook_resid_post" if node.startswith("resid_post") else "hook_mlp_in"
                    downstream_hooks.append(f"blocks.{layer}.{hook_type}")
                elif node.startswith("head"):
                    # in this case a specific head is specified with (optionally) a corresponding input channel (q, k or v)
                    head_idx = int(node.split(".")[2])
                    letters = ["q", "k", "v"]
                    if len(node.split(".")) == 4:
                        # a specific input channel is specified so we modify the hook name accordingly
                        letter_specified = node.split(".")[3]
                        assert letter_specified in letters, "Invalid letter specified"
                        letters = [letter_specified]
                    for letter in letters:
                        downstream_hooks.append(f"blocks.{layer}.hook_{letter}_input")
                else:
                    raise NotImplementedError("Invalid downstream node")

        upstream_hooks = list(set(upstream_hooks))
        downstream_hooks = list(set(downstream_hooks))

        # we sort the hooks by layer number
        upstream_hooks = sorted(upstream_hooks, key=lambda hook_name: int(hook_name.split(".")[1]))
        downstream_hooks = sorted(downstream_hooks, key=lambda hook_name: int(hook_name.split(".")[1]))

        return upstream_hooks, downstream_hooks

    def forward_with_patching(self, clean_tokens, corrupted_tokens, edges):
        upstream_nodes = [edges[i][0] for i in range(len(edges))]
        downstream_nodes = [edges[i][1] for i in range(len(edges))]

        self.setup_edges(upstream_nodes, downstream_nodes)

        adj_matrix = torch.zeros((self.n_upstream_nodes, self.n_downstream_nodes), device=self.device, dtype=self.dtype)

        print(f"Number of upstream nodes is {self.n_upstream_nodes}")
        print(f"Number of downstream nodes is {self.n_downstream_nodes}")

        upstream_node_index = 0
        downstream_node_index = 0
        upstream_node_indexes = []

        for upstream_node, downstream_node in zip(upstream_nodes, downstream_nodes): 
            upstream_node_index = self.upstream_node_name_to_index[upstream_node]
            downstream_node_index = self.downstream_node_name_to_index[downstream_node]
            upstream_node_indexes.append(upstream_node_index)

            adj_matrix[upstream_node_index, downstream_node_index] = 1.0

        seq_len = clean_tokens.shape[1]
        batch_size = clean_tokens.shape[0]
        assert seq_len == corrupted_tokens.shape[1], "Sequence length mismatch"
        assert batch_size == corrupted_tokens.shape[0], "Batch size mismatch"

        activations_diff_in_gb = batch_size * seq_len * self.n_upstream_nodes * self.d_model * self.dtype.itemsize / 2**30 
        memory_allocated_in_gb = torch.cuda.memory_allocated() / 2**30
        total_memory_in_gb = torch.cuda.get_device_properties(self.device).total_memory / 2**30
        print(f"Saving activation differences requires {activations_diff_in_gb:.2f} GB of memory")
        assert activations_diff_in_gb < total_memory_in_gb - memory_allocated_in_gb, "Not enough memory to store activation differences"

        self.model.reset_hooks()
        gc.collect()
        torch.cuda.empty_cache()

        upstream_activations_difference = torch.zeros((batch_size, seq_len, self.n_upstream_nodes, self.d_model), device=self.device, requires_grad=False)

        memory_allocated_in_gb = torch.cuda.memory_allocated() / 2**30
        print(f"Total memory allocated after creating activation differences tensor is {memory_allocated_in_gb:.2f} GB out of {total_memory_in_gb:.2f} GB")
        
        def corrupted_upstream_hook(activations, hook):
            hook_slice = self.upstream_hook_index[hook.name]
            if activations.ndim == 3:
                upstream_activations_difference[:, :, hook_slice, :] = -activations.unsqueeze(-2)
            elif activations.ndim == 4:
                upstream_activations_difference[:, :, hook_slice, :] = -activations

        def clean_upstream_hook(activations, hook):
            hook_slice = self.upstream_hook_index[hook.name]
            if activations.ndim == 3:
                upstream_activations_difference[:, :, hook_slice, :] += activations.unsqueeze(-2)
            elif activations.ndim == 4:
                upstream_activations_difference[:, :, hook_slice, :] += activations

        def clean_downstream_hook(activations, hook):
            downstream_hook_slice = self.downstream_hook_index[hook.name]

            # we get the slice of all upstream activation differences previous to this layer
            layer = hook.layer()
            earlier_layer_hooks_slice = self.upstream_before_layer[layer]

            # shape of adj_matrix_expanded: [1, 1, n_upstream_nodes_at_earlier_layers, n_downstream_nodes_at_hook, 1]
            # shape of upstream_activations_difference: [batch_size, seq_len, n_upstream_nodes_at_earlier_layers, d_model]
            # shape of patch_difference: [batch_size, seq_len, n_downstream_nodes_at_hook, d_model]
            # the tensor 'patch_difference' represents the sum of all upstream activation differences that are connected to this downstream node

            # For some reason this doesn't work: patch_difference = (adj_matrix_expanded[:, :, earlier_layer_hooks_slice, downstream_hook_slice, :] * upstream_activations_difference[:, :, earlier_layer_hooks_slice, :]).sum(dim=-2)
            patch_difference = einops.einsum(
                adj_matrix[earlier_layer_hooks_slice, downstream_hook_slice],
                upstream_activations_difference[:, :, earlier_layer_hooks_slice, :],
                "n_upstream n_downstream_at_hook, batch_size seq_len n_upstream d_model -> batch_size seq_len n_downstream_at_hook d_model"
            )

            # alternatively, it might be faster to
            # 1) gather the activation differences for every non-zero element in adj_matrix[:, hook_slice] and sum them
            # 2) use torch.sparse.mm to multiply the adj_matrix with the activation differences

            if activations.ndim == 3:
                assert patch_difference.shape[-2] == 1, "Number of downstream nodes should be 1 for this type of hook" 
                activations = activations - patch_difference.squeeze(-2)
            elif activations.ndim == 4:
                activations = activations - patch_difference
            
            return activations

        upstream_hook_filter = lambda name: name.endswith(tuple(self.upstream_hooks))
        downstream_hook_filter = lambda name: name.endswith(tuple(self.downstream_hooks))

        # we first perform a forward pass on the corrupted input 
        self.model.add_hook(upstream_hook_filter, corrupted_upstream_hook, "fwd")

        with torch.no_grad():
            corrupted_tokens = corrupted_tokens.to(self.device)
            self.model(corrupted_tokens, return_type=None)

        self.model.reset_hooks()
        torch.cuda.empty_cache()

        self.model.add_hook(upstream_hook_filter, clean_upstream_hook, "fwd")
        self.model.add_hook(downstream_hook_filter, clean_downstream_hook, "fwd")

        with torch.no_grad():
            clean_tokens = clean_tokens.to(self.device)
            logits = self.model(clean_tokens, return_type="logits")

        self.model.reset_hooks()
        del upstream_activations_difference
        torch.cuda.empty_cache()
        gc.collect()

        return logits 
