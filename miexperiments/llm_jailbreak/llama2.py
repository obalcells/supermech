import os
import sys
import torch
from jaxtyping import Int, Float
from torch import Tensor
import einops
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
from rich.console import Console
from rich.style import Style
from rich.color import Color
from rich.text import Text


# interfaces with the model
# 1) model(input_ids=ids, attention_mask=attn_mask).logits
# 2) model(inputs_embeds=full_embeds).logits
# 3) model.zero_grad()
# 4) model.model.{something}
# 5) model.generate()

def print_pretty_tensor(vals, input_ids=None):
    if len(vals.shape) == 2:
        for row in vals:
            line = Text() 
            for val in row:
                if val < 0.5:
                    # Gradient from red to yellow for values < 0.5
                    green_intensity = 255
                    red_intensity = int(255 * (val * 2))
                else:
                    # Gradient from yellow to green for values >= 0.5
                    green_intensity = int(255 * ((1 - val) * 2))
                    red_intensity = 255

                color = Color.from_rgb(red_intensity, green_intensity, 0)
                style = Style(color=color)
                if input_ids is not None:
                    color_code = f"rgb({red_intensity},{green_intensity},0)"
                    line.append(f"{val:.2f} ", style=color_code)

        console.print(line)

    elif len(vals.shape) == 1:
        line = Text()
        for i, val_ in enumerate(vals):
            val = val_.item() 
            if val < 0.5:
                # Gradient from red to yellow for values < 0.5
                green_intensity = 255
                red_intensity = int(255 * (val * 2))
            else:
                # Gradient from yellow to green for values >= 0.5
                green_intensity = int(255 * ((1 - val) * 2))
                red_intensity = 255

            color_code = f"\033[38;2;{red_intensity};{green_intensity};0m"
            # Using string formatting to ensure each number occupies a width of 6 characters
            if input_ids is not None:
                if input_ids[i] == '\n':
                    formatted_val = f"\.n".rjust(2)
                else:
                    formatted_val = f"{input_ids[i]}".rjust(2)
            else:
                formatted_val = f"{val:.2f}".rjust(6)
            line += f"{color_code}{formatted_val} "
        line += "\033[0m"  # Reset color
        print(line)

    else:
        print(vals)

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
        if self.save_activations:
            self.activations = output[0]
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

        self.calc_dot_product = False
        self.save_activations = False
        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        activations: Float[Tensor, "batch_size seq_len d_model"] = output[0]

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

            last_token_activations = activations[:, self.after_position-30:self.after_position , :]
            dot_product: Float[Tensor, "batch_size"] = einops.einsum(
                last_token_activations, self.calc_dot_product_with,
                "batch_size seq_len d_model, d_model -> batch_size"
            )
            # divide by the number of tokens if we compute across more than one token
            dot_product /= 30
            self.dot_products.append(dot_product)
        if self.add_activations is not None:
            augmented_output = add_vector_after_position(
                matrix=output[0],
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                after=self.after_position,
            )
            output = (augmented_output + self.add_activations,) + output[1:]

        if self.save_activations:
            self.activations = activations

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

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        # self.calc_dot_product_with = None
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.after_position = -1 
        self.dot_products = []

# Wrapper around the model to keep our stub class clean
class Llama2Wrapper:
    def __init__(self, model, tokenizer, system_prompt="You are a helpful, honest and concise assistant."):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model device is {self.device}")
        self.system_prompt = system_prompt
        # the model and tokenizers are passed as arguments
        # self.wrapped_model, self.tokenizer = Llama7BChatHelper.get_llama_model_and_tokenizer()
        self.wrapped_model = model
        self.tokenizer = tokenizer
        # the model is compiled to run on the GPU already
        # self.wrapped_model = self.wrapped_model.to(self.device)
        self.END_STR = torch.tensor(self.tokenizer.encode("[/INST]")[1:]).to(
            self.device
        )

        print(f"Model has {len(self.wrapped_model.model.layers)} layers")

        for i, layer in enumerate(self.wrapped_model.model.layers):
            self.wrapped_model.model.layers[i] = BlockOutputWrapper(
                layer, self.wrapped_model.lm_head, self.wrapped_model.model.norm, self.tokenizer
            )

    def set_save_activations(self, value):
        for layer in self.wrapped_model.model.layers:
            layer.save_activations = value
            layer.block.self_attn.save_activations = value

    def set_save_internal_decodings(self, value):
        for layer in self.wrapped_model.model.layers:
            layer.save_internal_decodings = value

    def set_after_positions(self, pos):
        for layer in self.wrapped_model.model.layers:
            layer.after_position = pos

    """ How a prompt+response looks like:
    [INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nHello, World! [\INST]Hello there!
    """
    def prompt_to_tokens(self, instruction):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        dialog_content = B_SYS + self.system_prompt + E_SYS + instruction.strip()
        dialog_tokens = self.tokenizer.encode(
            f"{B_INST} {dialog_content.strip()} {E_INST}"
        )
        return torch.tensor(dialog_tokens).unsqueeze(0)

    def generate_text(self, prompt, max_new_tokens=50) -> str:
        input_ids = self.prompt_to_tokens(prompt).to(self.device)
        output_ids = self.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, top_k=1)
        return self.tokenizer.batch_decode(output_ids)[0]

    def generate(self, *args, **kwargs):
        # input_ids = kwargs.get('input_ids')
        # assert not (input_ids is None), "Must pass input_ids to generate function as a positional argument"
        # instr_pos = find_instruction_end_postion(input_ids[0], self.END_STR)
        # self.set_after_positions(instr_pos)
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

    def show_tokens_dot_product(self, input_ids: Int[Tensor, "seq_len"]):
        chars = [self.tokenizer.decode(input_ids[i:i+1]) for i in range(len(input_ids))] 
        self.set_after_positions(0)
        self.set_save_activations(True)
        self.wrapped_model(input_ids=input_ids.unsqueeze(0)) # we don't get the output

        for i, layer in enumerate(self.wrapped_model.model.layers):
            if not layer.calc_dot_product and layer.calc_dot_product_with is None:
                continue
            print(f"Dot product of layer {i} across tokens:")
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