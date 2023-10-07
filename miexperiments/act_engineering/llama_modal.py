# %% Imports
from modal import Image, Secret, Stub, gpu, method
import torch
import os
import json
from IPython.display import display, HTML
from torch import Tensor
from jaxtyping import Int, Float
from typing import Tuple, List, Optional, Dict

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib

# %% Classes for downloading the model in the container

HF_MODEL_DIR = "meta-llama/Llama-2-7b-chat-hf"

def download_models():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    AutoModelForCausalLM.from_pretrained(
        HF_MODEL_DIR,
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
    )
    AutoTokenizer.from_pretrained(
        HF_MODEL_DIR,
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
    )

# %% Helper functions/classes

class ComparisonDataset(torch.utils.data.Dataset):
    def __init__(self, data, system_prompt, tokenizer):
        self.data = data
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        pos_answer = item["answer_matching_behavior"]
        neg_answer = item["answer_not_matching_behavior"]
        pos_tokens = prompt_to_tokens(
            self.tokenizer, self.system_prompt, question, pos_answer
        )
        neg_tokens = prompt_to_tokens(
            self.tokenizer, self.system_prompt, question, neg_answer
        )
        return pos_tokens, neg_tokens


def prompt_to_tokens(tokenizer, system_prompt, instruction, model_output):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
    dialog_tokens = tokenizer.encode(
        f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
    )
    return torch.tensor(dialog_tokens).unsqueeze(0)

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

def value_to_color(value, cmap=plt.cm.RdBu, vmin=-25, vmax=25):
    # Convert value to a range between 0 and 1
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(value))
    return matplotlib.colors.to_hex(rgba)


def display_token_dot_products(data):
    html_content = ""
    vmin = min([x[1] for x in data])
    vmax = max([x[1] for x in data])
    for token, value in data:
        color = value_to_color(value, vmin=vmin, vmax=vmax)
        html_content += f"<span style='background-color: {color}; padding: 2px 5px; margin: 2px; border-radius: 3px;'>{token} ({value:.4f})</span>"
    display(HTML(html_content))
    
def display_token_dot_products_final_text(data, text, tokenizer):
    html_content = "<div>"
    vmin = min([x[1] for x in data])
    vmax = max([x[1] for x in data])
    tokens = tokenizer.encode(text)
    tokens = tokenizer.batch_decode(torch.tensor(tokens).unsqueeze(-1))
    for idx, (_, value) in enumerate(data):
        color = value_to_color(value, vmin=vmin, vmax=vmax)
        html_content += f"<span style='background-color: {color}; padding: 2px 5px; margin: 2px; border-radius: 3px;'>{tokens[idx].strip()} ({value:.4f})</span>"
    html_content += "</div>"
    display(HTML(html_content))

def display_tokens(input_ids: Int[Tensor, "batch_size seq_len"], tokenizer):
    background_color = "#5e5e5e"
    text_color = "#ffffff"
    html_content = "<div>"
    # we just display the first batch
    tokens = tokenizer.batch_decode(input_ids[0].unsqueeze(-1))
    for token in tokens:
        html_content += f"<span style='color: {text_color}; background-color: {background_color}; padding: 2px 5px; margin: 2px;'>{token} (0.0)</span>"
    html_content += "</div>"
    display(HTML(html_content))

# %% Model wrapper classes
class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
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
        self.after_position = None

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = torch.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = torch.dot(last_token_activations, self.calc_dot_product_with)
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations is not None:
            augmented_output = add_vector_after_position(
                matrix=output[0],
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                after=self.after_position,
            )
            output = (augmented_output + self.add_activations,) + output[1:]

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
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.after_position = None
        self.calc_dot_product_with = None
        self.dot_products = []

# Wrapper around the model to keep our stub class clean
class ModelWrapper:
    def __init__(self, model, tokenizer, system_prompt="You are a helpful, honest and concise assistant."):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model device is {self.device}")
        self.system_prompt = system_prompt
        # the model and tokenizers are passed as arguments
        # self.model, self.tokenizer = Llama7BChatHelper.get_llama_model_and_tokenizer()
        self.model = model
        self.tokenizer = tokenizer
        # the model is compiled to run on the GPU already
        # self.model = self.model.to(self.device)
        self.END_STR = torch.tensor(self.tokenizer.encode("[/INST]")[1:]).to(
            self.device
        )

        print(f"Model has {len(self.model.model.layers)} layers")

        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.model.norm, self.tokenizer
            )

    def set_save_internal_decodings(self, value):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def set_after_positions(self, pos):
        for layer in self.model.model.layers:
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

    def generate_text(self, prompt, max_new_tokens=50):
        tokens = self.prompt_to_tokens(prompt).to(self.device)
        return self.generate(tokens, max_new_tokens=max_new_tokens)

    def generate(self, tokens, max_new_tokens=50):
        instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
        self.set_after_positions(instr_pos)
        generated = self.model.generate(
            inputs=tokens, max_new_tokens=max_new_tokens, top_k=1
        )
        return self.tokenizer.batch_decode(generated)[0]

    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens).logits
            return logits

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add(activations)

    def set_calc_dot_product_with(self, layer, vector):
        self.model.model.layers[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self.model.model.layers[layer].dot_products

    def reset_all(self):
        for layer in self.model.model.layers:
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
        for i, layer in enumerate(self.model.model.layers):
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
        layer = self.model.model.layers[layer_number]

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

# %% Modal image specs

image = (
    # Python 3.11+ not yet supported for torch.compile
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate~=0.18.0",
        "transformers~=4.28.1",
        "torch~=2.0.0",
        "jaxtyping",
        "sentencepiece~=0.1.97",
        "typing-extensions==4.5.0",  # >=4.6 causes typing issues
        "matplotlib"
    )
    .run_function(
        download_models,
        secret=Secret.from_name("huggingface"),
    )
)

# %% Modal Llama2 class

stub = Stub(name="llama2", image=image)

@stub.cls(gpu=gpu.A100(), secret=Secret.from_name("huggingface"))
class Llama7BChatHelper:
    def __enter__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_DIR,
            use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
        )

        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_DIR,
            use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
            torch_dtype=torch.float16,
            device_map="auto",
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model.eval()
        # model = torch.compile(model)

        self.wrapped_model = ModelWrapper(model, tokenizer)

    @method()
    def generate(
        self,
        prompt: str
    ) -> str:
        print(f"Prompt is {prompt}")
        text = self.wrapped_model.generate_text(prompt, max_new_tokens=5)
        print(f"Generated text is {text}")
        return text 


    # Can I pass the pytorch dataset directly?
    # Can I mount my own dataset folder there?
    @method()
    def generate_steering_vectors(
        self, data, start_layer=0, end_layer=32, token_idx=-2
    ) -> Tuple[List[Float[Tensor, "d_model"]], List[Int]]:
        from tqdm import tqdm

        system_prompt = "You are a helpful, honest and concise assistant."
        dataset = ComparisonDataset(data, system_prompt, self.wrapped_model.tokenizer)
        print(f"Using {len(dataset)} samples")

        layers = list(range(start_layer, end_layer))
        positive_activations = dict([(layer, []) for layer in layers])
        negative_activations = dict([(layer, []) for layer in layers])
        self.wrapped_model.set_save_internal_decodings(False)
        self.wrapped_model.reset_all()
        for p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        # for p_tokens, n_tokens in dataset:
            p_tokens = p_tokens.to(self.wrapped_model.device)
            n_tokens = n_tokens.to(self.wrapped_model.device)
            self.wrapped_model.reset_all()
            self.wrapped_model.get_logits(p_tokens)
            for layer in layers:
                p_activations = self.wrapped_model.get_last_activations(layer)
                p_activations = p_activations[0, token_idx, :].detach().cpu()
                positive_activations[layer].append(p_activations)
            self.wrapped_model.reset_all()
            self.wrapped_model.get_logits(n_tokens)
            for layer in layers:
                n_activations = self.wrapped_model.get_last_activations(layer)
                n_activations = n_activations[0, token_idx, :].detach().cpu()
                negative_activations[layer].append(n_activations)
        steering_vectors = []
        for layer in layers:
            positive = torch.stack(positive_activations[layer])
            negative = torch.stack(negative_activations[layer])
            vec = (positive - negative).mean(dim=0)
            steering_vectors.append(vec)
        return steering_vectors, layers

# %% Modal entrypoint

@stub.local_entrypoint()
def generate_and_save_steering_vectors():
    model = Llama7BChatHelper()

    data_path = "../datasets/custom_harm_dataset/refusal_data_A_B.json"
    data = json.load(open(data_path))

    print(f"The first 3 data samples are: {data[:3]}")

    vectors_path = "./harmful_steering_vectors"
    os.makedirs(vectors_path, exist_ok=True)

    steering_vectors, layers = model.generate_steering_vectors.remote(data)

    for layer, vec in zip(layers, steering_vectors):
        torch.save(vec, f"{vectors_path}/vec_layer_{layer}.pt")
