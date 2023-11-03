import os
import sys
import torch
from jaxtyping import Int, Float
from torch import Tensor
import einops
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter

def print_tokens(tokens: Int[Tensor, "seq_len"], tokenizer):
    if len(tokens.shape) == 2:
        # remove the batch dimension if present
        tokens = tokens[0]
    print([tokenizer.decode(tok) for tok in tokens])

# projects activations into direction
def project_onto_direction(activations: Float[Tensor, "n d_model"], direction: Float[Tensor, "d_model"]):
    mag = torch.norm(direction)
    assert not torch.isinf(mag), "The magnitude of the direction vector should not be infinity"

    # Project each vector in H onto the direction
    # Note: Here we assume H is a 2-D tensor where each row is a vector
    # that needs to be projected onto the direction.
    # The unsqueeze operation is necessary to align the dimensions for broadcasting
    # during the dot product.
    projection = torch.matmul(activations, direction.unsqueeze(1)) / mag

    # The result will be a column tensor with the projection scalars.
    # If you want the projection result in the direction of the vector,
    # you need to multiply by the unit vector in the direction of 'direction'.
    unit_direction = direction / mag
    projection_as_vectors = projection * unit_direction.unsqueeze(0)
    
    return projection_as_vectors

def add_vector_after_position(matrix, vector, position_ids, after=None):
    after_id = after
    if after_id is None:
        after_id = position_ids.min().item() - 1
    mask = position_ids > after_id
    mask = mask.unsqueeze(-1)
    matrix += mask.float() * vector
    return matrix


def find_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m + 1):
        if torch.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1


def find_instruction_end_postion(tokens, end_str):
    end_pos = find_subtensor_position(tokens, end_str)
    return end_pos + len(end_str) - 1

# %% Model wrapper classes
class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.save_activations = False 
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        # if self.save_activations:
        #     self.activations = output[0]
        return output

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.after_position = -1 

        self.optimize_activations = False
        self.activations_callback = None 

        self.calc_dot_product = False
        self.save_activations = False
        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        activations: Float[Tensor, "batch_size seq_len d_model"] = output[0]

        if self.activations_callback is not None:
            self.activations_callback(activations)

        if self.calc_dot_product == True and self.calc_dot_product_with is not None:
            # last_token_activations = self.activations[0, -1, :]
            # decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            # top_token_id = torch.topk(decoded_activations, 1)[1][0]
            # top_token = self.tokenizer.decode(top_token_id)
            # dot_product = torch.dot(last_token_activations, self.calc_dot_product_with)
            # self.dot_products.append((top_token, dot_product.cpu().item()))

            # dot_products_all_tokens = einops.einsum(
            #     activations, self.calc_dot_product_with,
            #     "batch_size seq_len d_model, d_model -> batch_size seq_len"
            # )

            # dot_products_avg_all_tokens = einops.reduce(
            #     dot_products_all_tokens, "batch_size seq_len -> seq_len", 'mean'
            # )

            # print(f"Dot products across all tokens: {dot_products_across_all_tokens}")
            # print(f"Printing tensor of shape {dot_products_avg_all_tokens.shape}")
            # pretty_tensor(dot_products_avg_all_tokens)

            # cosim: Float[Tensor, "batch_size"] = einops.einsum(
            #     last_token_activations / last_token_activations.norm(dim=-1, keepdim=True),
            #     self.calc_dot_product_with,
            #     "batch_size d_model, d_model -> batch_size"
            # )

            multiplier = 1.0
            normalize = True

            last_token_activations = activations[:, self.after_position, :]
            norm_pre = torch.norm(last_token_activations, dim=-1, keepdim=True)
            target_activations = last_token_activations + self.calc_dot_product_with * multiplier
            if normalize:
                norm_post = torch.norm(target_activations, dim=-1, keepdim=True)
                target_activations = target_activations / norm_post * norm_pre
            l2_norm = torch.norm(last_token_activations - target_activations, p=2)

            self.dot_products.append(l2_norm)

        if self.add_activations is not None:
            # ADDSPOT
            # print(f"Position ids are {kwargs['position_ids']}")
            # print(f"After position is {self.after_position}")
            # print(f"Activations shape is {activations.shape}")
            normalize = True

            # Approach 1)
            norm_pre = activations[:, self.after_position, :].norm(dim=-1, keepdim=True)
            activations[:,self.after_position:,:] = activations[:,self.after_position:,:] + 1.0 * self.add_activations
            norm_post = activations[:, self.after_position, :].norm(dim=-1, keepdim=True)
            if normalize:
                norm_post = activations[:, self.after_position, :].norm(dim=-1, keepdim=True)
                activations[:,self.after_position:,:] = (activations[:,self.after_position:,:] / norm_post) * norm_pre

            # Approach 2) Same but projecting first
            # print(f"Approach 2")
            # norm_pre = activations[:, self.after_position, :].norm(dim=-1, keepdim=True)
            # proj_activations = project_onto_direction(activations[:, self.after_position, :], self.add_activations)
            # activations[:,self.after_position:,:] -= proj_activations
            # activations[:,self.after_position:,:] = activations[:,self.after_position:,:] + 1.0 * self.add_activations
            # if normalize:
            #     norm_post = activations[:, self.after_position, :].norm(dim=-1, keepdim=True)
            #     activations[:,self.after_position:,:] = (activations[:,self.after_position:,:] / norm_post) * norm_pre

            # augmented_output = add_vector_after_position(
            #     matrix=activations,
            #     vector=self.add_activations,
            #     position_ids=kwargs["position_ids"],
            #     after=self.after_position,
            # )
            # output = (augmented_output + self.add_activations,) + output[1:]

        if self.save_activations:
            self.activations = output[0] 

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def set_activations_callback(callback):
        self.activations_callback = callback

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        # self.calc_dot_product_with = None
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.after_position = -1 
        self.dot_products = []
        self.activations_callback = None

# Wrapper around the model to keep our stub class clean
class Llama2Wrapper:
    def __init__(self, model, tokenizer, system_prompt="You are a helpful, honest and concise assistant."):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model device is {self.device}")
        self.system_prompt = system_prompt
        self.wrapped_model = model
        self.tokenizer = tokenizer
        # the model is compiled to run on the GPU already
        self.END_STR = torch.tensor(self.tokenizer.encode("[/INST]")[1:]).to(
            self.device
        )

        for i, layer in enumerate(self.wrapped_model.model.layers):
            self.wrapped_model.model.layers[i] = BlockOutputWrapper(
                layer, self.wrapped_model.lm_head, self.wrapped_model.model.norm, self.tokenizer
            )

    def set_save_activations(self, value, layers=None):
        if layers is None:
            for layer in self.wrapped_model.model.layers:
                layer.save_activations = value
                layer.block.self_attn.save_activations = value
        elif isinstance(layers, int):
            self.wrapped_model.model.layers[layers].save_activations = value
            self.wrapped_model.model.layers[layers].block.self_attn.save_activations = value
        elif isinstance(layers[0], int):
            for layer in layers:
                self.wrapped_model.model.layers[layer].save_activations = value
                self.wrapped_model.model.layers[layer].block.self_attn.save_activations = value
        else:
            assert False, "Incorrect input arguments to set_save_activations function"

    def set_save_internal_decodings(self, value):
        for layer in self.wrapped_model.model.layers:
            layer.save_internal_decodings = value

    def set_after_positions(self, pos):
        for layer in self.wrapped_model.model.layers:
            layer.after_position = pos

    def set_activations_callback(self, layer, callback):
        self.wrapped_model.model.layers[layer].activations_callback = callback

    def prompt_to_tokens(
        self,
        instruction,
        model_output="",
        append_space_after_instruction=True,
        skip_sys_tags=True,
        batch_dim=True
    ) -> Int[Tensor, "batch_size seq_len"]:

        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        if len(self.system_prompt) == 0 and skip_sys_tags:
            # if there's no system prompt we don't add the <<SYS>> tags inside the instruction
            dialog_content = instruction.strip()
        else:
            dialog_content = B_SYS + self.system_prompt + E_SYS + instruction.strip()

        if len(model_output) == 0 and not append_space_after_instruction:
            prompt = f"{B_INST} {dialog_content.strip()} {E_INST}" 
        else:
            prompt = f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"

        if batch_dim:
            return self.tokenizer.encode(prompt, return_tensors="pt")
        else:
            return self.tokenizer.encode(prompt, return_tensors="pt")[0]

    def generate_text(self, prompt, max_new_tokens=16, **kwargs) -> str:
        input_ids = self.prompt_to_tokens(prompt).to(self.device)
        print(f"Printing tokens inside the `generate_text` function:")
        print_tokens(input_ids[0], self.tokenizer)
        output_ids = self.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, **kwargs)
        return self.tokenizer.batch_decode(output_ids)[0]

    def generate(self, *args, **kwargs):
        assert not (kwargs.get("input_ids") is None), "Must pass input_ids to generate function as a positional argument"
        instr_pos = find_instruction_end_postion(kwargs.get("input_ids")[0], self.END_STR)
        instr_pos -= 4 
        print("Instr pos is ", instr_pos)
        self.set_after_positions(instr_pos)
        return self.wrapped_model.generate(*args, **kwargs)

    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.wrapped_model(tokens).logits
            return logits

    def get_last_activations(self, layer):
        return self.wrapped_model.model.layers[layer].activations

    def set_add_activations(self, layer, activations):
        self.wrapped_model.model.layers[layer].add(activations)

    def set_calc_dot_product(self, value:bool, layer=None):
        if layer is None:
            for layer in self.wrapped_model.model.layers:
                layer.calc_dot_product = value 
        else:
            assert isinstance(layer, int), "Layer passed must be an integer"
            self.wrapped_model.model.layers[layer].calc_dot_product = value

    def set_calc_dot_product_with(self, layer, vector):
        self.wrapped_model.model.layers[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self.wrapped_model.model.layers[layer].dot_products

    def reset_all(self):
        for layer in self.wrapped_model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        data = self.get_activation_data(decoded_activations, topk)[0]
        print(label, data)

    def decode_all_layers(
        self,
        tokens,
        topk=10,
        print_attn_mech=True,
        print_intermediate_res=True,
        print_mlp=True,
        print_block=True,
    ):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        for i, layer in enumerate(self.wrapped_model.model.layers):
            print(f"Layer {i}: Decoded intermediate outputs")
            if print_attn_mech:
                self.print_decoded_activations(
                    layer.attn_out_unembedded, "Attention mechanism", topk=topk
                )
            if print_intermediate_res:
                self.print_decoded_activations(
                    layer.intermediate_resid_unembedded,
                    "Intermediate residual stream",
                    topk=topk,
                )
            if print_mlp:
                self.print_decoded_activations(
                    layer.mlp_out_unembedded, "MLP output", topk=topk
                )
            if print_block:
                self.print_decoded_activations(
                    layer.block_output_unembedded, "Block output", topk=topk
                )

    def plot_decoded_activations_for_layer(self, layer_number, tokens, topk=10):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        layer = self.wrapped_model.model.layers[layer_number]

        data = {}
        data["Attention mechanism"] = self.get_activation_data(
            layer.attn_out_unembedded, topk
        )[1]
        data["Intermediate residual stream"] = self.get_activation_data(
            layer.intermediate_resid_unembedded, topk
        )[1]
        data["MLP output"] = self.get_activation_data(layer.mlp_out_unembedded, topk)[1]
        data["Block output"] = self.get_activation_data(
            layer.block_output_unembedded, topk
        )[1]

        # Plotting
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        fig.suptitle(f"Layer {layer_number}: Decoded Intermediate Outputs", fontsize=21)

        for ax, (mechanism, values) in zip(axes.flatten(), data.items()):
            tokens, scores = zip(*values)
            ax.barh(tokens, scores, color="skyblue")
            ax.set_title(mechanism)
            ax.set_xlabel("Value")
            ax.set_ylabel("Token")

            # Set scientific notation for x-axis labels when numbers are small
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))

    @property
    def generation_config(self):
        return self.wrapped_model.generation_config

    # we set this property to match the way we interface with the HF Llama2 model
    # whenever we have a transformers llama2 model we call model.model to get the underlying
    # pytorch model
    @property
    def model(self):
        return self.wrapped_model.model

    @model.setter
    def model(self, model):
        self.wrapped_model = model

    def zero_grad(self):
        self.wrapped_model.zero_grad()

    def __call__(self, *args, **kwargs):
        if kwargs.get('after_position') is not None:
            # the position of the first token in the loss slice
            after_position = kwargs.get('after_position')
            self.set_after_positions(after_position)
        return self.wrapped_model(*args, **kwargs)

    def get_prompt_with_system(self, instruction, model_output="") -> str:
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        if self.system_prompt is None or len(self.system_prompt) == 0:
            dialog_content = instruction.strip()
        else:
            dialog_content = B_SYS + self.system_prompt + E_SYS + instruction.strip()

        if len(model_output) > 0:
            dialog_conent = f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
        else:
            dialog_content = f"{B_INST} {dialog_content.strip()} {E_INST}"

        return dialog_content

    def tokenize_and_batch(self, prompts, batch_size=512):
        prompts_with_system = [self.get_prompt_with_system(prompt) for prompt in prompts]
        num_batches = len(prompts) // batch_size + int(len(prompts) % batch_size != 0)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(len(prompts_with_system), start_idx + batch_size)

            yield self.tokenizer(
                prompts_with_system[start_idx:end_idx],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

    def show_tokens_dot_product(self, input_ids: Int[Tensor, "seq_len"], layers=None):
        chars = [self.tokenizer.decode(input_ids[i:i+1]) for i in range(len(input_ids))] 
        self.set_save_activations(True)
        self.wrapped_model(input_ids=input_ids.unsqueeze(0)) # we don't get the output

        for layer_idx, layer in enumerate(self.wrapped_model.model.layers):
            if not layer.calc_dot_product and layer.calc_dot_product_with is None:
                continue
            elif layer_idx not in layers:
                continue
            print(f"Dot product of layer {layer_idx} across tokens:")
            dot_product = einops.einsum(
                layer.activations[0], layer.calc_dot_product_with,
                "seq_len d_model, d_model -> seq_len"
            )
            print_pretty_tensor(dot_product, chars)

        self.reset_all()
        self.set_save_activations(False)


def download_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    HF_MODEL_DIR = "meta-llama/Llama-2-7b-chat-hf"

    AutoModelForCausalLM.from_pretrained(
        HF_MODEL_DIR,
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
    )
    AutoTokenizer.from_pretrained(
        HF_MODEL_DIR,
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
    )

if __name__ == "__main__":
    download_models()