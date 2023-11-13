# %%
import os
import gc
# os.environ["CUDA_VISIBLE_DEVICES"] = "5" # has to be before importing torch
from modal import Image, Secret, Stub, gpu, method
import torch
import torch.nn.functional as F
from torch import Tensor
import json
from IPython.display import display, HTML
from typing import Tuple, List, Optional, Dict
from tqdm import tqdm

# %% Helper functions
HF_MODEL_DIR = "meta-llama/Llama-2-7b-chat-hf"

def download_models():
    from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaTokenizerFast
    from datasets import load_dataset

    AutoModelForCausalLM.from_pretrained(
        HF_MODEL_DIR,
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
    )
    LlamaTokenizer.from_pretrained(
        HF_MODEL_DIR,
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
    )
    LlamaTokenizerFast.from_pretrained(
        "huggyllama/llama-7b",
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
    )

def clear_memory():
    # 1/0
    cuda_mem_before = torch.cuda.memory_allocated()//1024//1024
    torch.cuda.empty_cache()
    gc.collect()
    cuda_mem_now = torch.cuda.memory_allocated()//1024//1024
    print(f"Released: {cuda_mem_before-cuda_mem_now} MB")
    print(f"CUDA memory now: {cuda_mem_now} MB")

def instruction_to_prompt(
    instruction: str,
    system_prompt: str="",
    model_output: str="",
) -> str:
    """
    Converts an instruction to a prompt string structured for Llama2-chat.
    Note that, unless model_output is supplied, the prompt will (intentionally) end with a space.
    See details of Llama2-chat prompt structure here: here https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    """

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    if len(system_prompt) == 0:
        dialog_content = instruction.strip()
    else:
        dialog_content = B_SYS + system_prompt.strip() + E_SYS + instruction.strip()
    prompt = f"{B_INST} {dialog_content} {E_INST} {model_output.strip()}"
    return prompt

def find_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m + 1):
        if torch.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1

def calculate_perplexity(logits, input_ids, end_of_instruction_ids):
    """
    Calculate the perplexity for each sequence in a batch.

    :param logits: Tensor of shape [batch_size, seq_len, vocab_size] containing the logits.
    :param output_tokens: Tensor of shape [batch_size, seq_len] containing the generated token IDs.
    :param pad_token_id: The ID of the padding token.
    :return: A tensor of perplexities for each sequence in the batch.
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Calculate the cross-entropy loss
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fn(logits[:,:-1,:].transpose(1,2), input_ids[:,1:])
    
    # find the position where the instruction ends for each prompt in the batch
    end_of_instruction_positions = []
    end_of_instruction_ids = end_of_instruction_ids.to(logits.device)

    for i in range(batch_size):
        end_of_instruction_position = find_subtensor_position(input_ids[i], end_of_instruction_ids)
        assert end_of_instruction_position >= 1, "End of instruction token not found in prompt"
        end_of_instruction_positions.append(end_of_instruction_position)

    end_of_instruction_positions = torch.tensor(end_of_instruction_positions).reshape(batch_size, 1).to(logits.device)

    # Adjust the position_ids to match the original sequence length
    position_ids = torch.arange(seq_len, device=logits.device).unsqueeze(0)

    # Now, the comparison operation will yield a tensor of shape [batch_size, seq_len]
    output_tokens_mask = (position_ids >= end_of_instruction_positions).float()

    # Ensure the mask has the correct shape for the subsequent operations
    output_tokens_mask = output_tokens_mask[:, 1:].to(logits.device)

    # Reshape loss back to [batch_size, seq_len]
    loss = loss.view(batch_size, seq_len-1)

    # Calculate the average loss per sequence, taking into account the different lengths
    loss_per_sequence = torch.sum(loss * output_tokens_mask, dim=1) / torch.sum(output_tokens_mask, dim=1)

    # Calculate the perplexity for each sequence
    perplexity = torch.exp(loss_per_sequence)

    return perplexity

# %% Modal image
image = (
    # Python 3.11+ not yet supported for torch.compile
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate~=0.18.0",
        "transformers~=4.28.1",
        "torch~=2.0.0",
        "jaxtyping",
        "sentencepiece~=0.1.97",
        "einops",
        "datasets",
        "tqdm"
    )
    .run_function(
        download_models,
        secret=Secret.from_name("huggingface"),
    )
)

# %% Modal class 
stub = Stub(name="llama2-test-steering", image=image)

@stub.cls(gpu=gpu.A100(), secret=Secret.from_name("huggingface"))
class Benchmark:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast

        assert torch.cuda.is_available(), "CUDA must be available"

        self.tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_DIR,
            use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
        )

        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = 'left'

        self.fast_tokenizer = LlamaTokenizerFast.from_pretrained(
            "huggyllama/llama-7b",
            truncation_side="left",
            use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
        )

        self.fast_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.hf_model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_DIR,
            use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.hf_model.eval()


    @method()
    def generation(
        self,
        model_wrapper,
        data,
        batch_size=64,
        max_new_tokens=256,
        system_prompt=""
    ):
        from datasets import load_dataset

        if isinstance(data, str):
            if data == "advbench":
                dataset = load_dataset('obalcells/advbench', split="train")
                data = [
                    {
                        "instruction": dataset[i]["goal"],
                        "output": dataset[i]["target"],
                    }
                    for i in range(len(dataset))
                ]

            elif data == "alpaca":
                dataset = load_dataset('tatsu-lab/alpaca', split='train')
                # we discard all the entries with input context
                data = [
                    {
                        "instruction": dataset[i]["instruction"],
                        "output": dataset[i]["output"],
                    }
                    for i in range(len(dataset)) if len(dataset[i]["input"]) == 0
                ]
                
            else:
                assert False, "Unknown dataset"
        else:
            assert isinstance(data, List), "Invalid format"

        model_wrapper.wrap_model(self.hf_model, self.tokenizer)

        instructions = [sample["instruction"] for sample in data]
        prompts = [instruction_to_prompt(instruction, system_prompt=system_prompt, model_output="") for instruction in instructions]

        generations = []

        clear_memory()

        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            batch = self.tokenizer(batch_prompts, padding=True, return_tensors="pt")
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].half().cuda()
            num_input_tokens = input_ids.shape[1]

            clear_memory()

            output_tokens = model_wrapper.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )

            generation = self.tokenizer.batch_decode(output_tokens[:, num_input_tokens:], skip_special_tokens=True)
            generations.extend(generation)

            del input_ids, attention_mask, output_tokens
            gc.collect()
            clear_memory()

        results = []

        for i in range(len(data)):
            results.append({
                "instruction": instructions[i],
                "generation": generations[i]
            })

        return results 

    @method()
    def perplexity(
        self,
        model_wrapper,
        data=None,
        return_baseline_ratio=True,
        return_mean=False,
        system_prompt=""
    ):
        from datasets import load_dataset

        if isinstance(data, str):
            # text dataset
            # we just return the average perplexity across the text
            if data == "openwebtext10k":
                dataset = load_dataset('stas/openwebtext-10k', split='train')
                text = dataset["text"][:100]
                model_wrapper.wrap_model(self.hf_model, self.tokenizer)
                perplexity = self.perplexity_text(model_wrapper, text)
                print(f"Perplexity: {perplexity}")
                return perplexity

            # instruction dataset
            if data == "advbench":
                dataset = load_dataset('obalcells/advbench', split="train")
                data = [
                    {
                        "instruction": dataset[i]["goal"],
                        "output": dataset[i]["target"],
                    }
                    for i in range(len(dataset))
                ]

            # instruction dataset
            elif data == "alpaca":
                dataset = load_dataset('tatsu-lab/alpaca', split='train')
                # we discard all the entries with input context
                data = [sample for sample in dataset if len(sample["input"]) == 0] 
                
            else:
                assert False, "Unknown dataset"

        if return_baseline_ratio:
            baseline_perplexities = self.perplexity_instructions(model_wrapper, data, system_prompt=system_prompt)

        model_wrapper.wrap_model(self.hf_model, self.tokenizer)
        perplexities = self.perplexity_instructions(model_wrapper, data, system_prompt=system_prompt)

        if return_mean:
            if return_baseline_ratio:
                ratios = [perplexity / baseline_perplexity for perplexity, baseline_perplexity in zip(perplexities, baseline_perplexities)]
                return sum(ratios) / len(ratios)
            else:
                return sum(perplexities) / len(perplexities)

        results = []

        for i in range(len(data)):
            if return_baseline_ratio:
                results.append({ 
                    "instruction": data[i]["instruction"],
                    "output": data[i]["output"],
                    "perplexity_ratio": perplexities[i] / baseline_perplexities[i],
                })
            else:
                results.append({ 
                    "instruction": data[i]["instruction"],
                    "output": data[i]["output"],
                    "perplexity": perplexities[i],
                })

        return results 

    def perplexity_text(
        self,
        model,
        text,
        stride=512,
        batch_size=64,
        seq_stride=50
    ) -> float:
        losses = []
        total_loss_len = 0
        max_len = 512 
        clear_memory()

        # we process seq_stride squences at a time
        # otherwise it takes too long to tokenize everything at once
        for seq_index in tqdm(range(0, len(text), seq_stride)):
            seq_index_end = min(seq_index + seq_stride, len(text))
            encodings = self.fast_tokenizer("\n\n".join(text[seq_index:seq_index_end]), return_tensors="pt")
            text_len = encodings.input_ids.size(1)

            for i in range(0, text_len, batch_size * stride):
                begin_locs, end_locs, trg_lens = [], [], []
                for j in range(batch_size):
                    j = i + j * stride
                    if j >= text_len:
                        break
                    begin_loc = max(j + stride - max_len, 0)
                    end_loc = min(j + stride, text_len)
                    trg_len = end_loc - j  # may be different from stride on last loop

                    begin_locs.append(begin_loc)
                    end_locs.append(end_loc)
                    trg_lens.append(trg_len)

                input_ids = [encodings.input_ids[:, b:e] for b, e in zip(begin_locs, end_locs)]

                target_end_locs = [sen.size(-1) for sen in input_ids]
                input_ids = [
                    F.pad(sen, (0, max_len - sen.size(-1)), "constant", 0) for sen in input_ids
                ] # we dont need attention mask as long as these padded token is not involved in loss calculation

                input_ids = torch.stack(input_ids, dim=1).squeeze(0).cuda()

                target_ids = torch.ones_like(input_ids) * -100 # -100 is the default ingore_index value in torch.nn.CrossEntropyLoss
                for i, (b, e) in enumerate(zip(trg_lens, target_end_locs)):
                    labels = input_ids[i, -b:e].clone()
                    target_ids[i, -b:e] = labels

                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)
                    log_likelihood = outputs["loss"].type(torch.float64) * sum(trg_lens)

                total_loss_len += sum(trg_lens)
                losses.append(log_likelihood)

                del input_ids, target_ids, outputs
                gc.collect()
                torch.cuda.empty_cache()

        print("END")
        print(sum(torch.stack(losses)))
        print(total_loss_len)
        print(sum(torch.stack(losses)) / total_loss_len)

        perplexity = torch.exp(sum(torch.stack(losses)) / total_loss_len).cpu().item()

        return perplexity

    def perplexity_instructions(
        self,
        model,
        data,
        batch_size=1024,
        system_prompt=""
    ):
        instructions = [sample["instruction"] for sample in data]
        outputs = [sample["output"] for sample in data]
        prompts = [instruction_to_prompt(instruction, system_prompt=system_prompt, model_output=output) for instruction, output in zip(instructions, outputs)]
        END_STR = torch.tensor(self.tokenizer.encode("[/INST]")[1:])
        perplexities = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch = self.tokenizer(batch_prompts, padding=True, return_tensors="pt")

            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].half().cuda()
            num_input_tokens = input_ids.shape[1]

            with torch.no_grad():
                logits = model(
                    input_ids,
                    attention_mask=attention_mask,
                ).logits

                perplexity = calculate_perplexity(logits, input_ids, END_STR)
                perplexity = perplexity.cpu().tolist()

            perplexities.extend(perplexity)

        return perplexities

# %% Module wrapper
class ModuleWrapper(torch.nn.Module):
    def __init__(self, module, end_of_instruction_tokens):
        super().__init__()
        self.module = module
        self.END_STR = end_of_instruction_tokens

        self.initial_input_ids = None
 
        self.do_steering = False
        self.steering_vector = None
        self.prompt_steering_positions = [-2, -1] # we only add the steering vector at the last 2 positions
        self.steer_at_generation_tokens = True

        self.normalize = True
        self.multiplier = 1.0

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)

        if not self.do_steering:
            return output

        if isinstance(output, tuple):
            activations = output[0]
        else:
            activations = output

        position_ids = kwargs.get("position_ids", None)

        assert position_ids is not None, "Position ids must be provided"
        assert self.steering_vector is not None, "Steering vector must be set"

        if self.steering_vector.dtype != activations.dtype:
            self.steering_vector = self.steering_vector.type(activations.dtype)

        if len(self.steering_vector.shape) == 1:
            self.steering_vector = self.steering_vector.reshape(1, 1, -1)

        if self.steering_vector.device != activations.device:
            self.steering_vector = self.steering_vector.to(activations.device)

        if activations.shape[1] == 1 and self.steer_at_generation_tokens:
            # steering at generation tokens
            token_positions = [0]
        elif activations.shape[1] > 1:
            # steering at prompt tokens
            token_positions = self.prompt_steering_positions

        norm_pre = torch.norm(activations, dim=-1, keepdim=True)

        activations[:, token_positions] = activations[:, token_positions] + self.steering_vector * self.multiplier

        norm_post = torch.norm(activations, dim=-1, keepdim=True)

        activations = activations / norm_post * norm_pre

        if isinstance(output, tuple):
            output = (activations,) + output[1:] 
        else:
            output = modified

        return output

class ModelWrapper:
    def __init__(self):
        self.model = None
        self.tokenizer = None

        self.jailbreak_vectors = torch.load("../../experiments/jailbreak_vector/steering_vector_at_last_pos.pt", map_location=torch.device("cpu"))

    def wrap_model(self, hf_model, tokenizer):
        self.tokenizer = tokenizer
        self.END_STR = torch.tensor(self.tokenizer.encode("[/INST]")[1:])

        self.model = hf_model

        # we only wrap layer 19 at the moment
        self.model.model.layers[19] = ModuleWrapper(
            self.model.model.layers[19], self.END_STR
        )
        self.model.model.layers[19].steering_vector = self.jailbreak_vectors[19]
        self.model.model.layers[19].do_steering = True 
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def unwrap_model(self):
        self.model.model.layers[19] = self.model.model.layers[19].module
        assert not isinstance(self.model.model.layers[19], ModuleWrapper)

# %% Modal entrypoint
@stub.local_entrypoint()
def run_benchmark():
    benchmark = Benchmark()
    model_wrapper = ModelWrapper()
 
   benchmark_output = benchmark.perplexity.remote(model_wrapper, data="openwebtext10k")

    print("Benchmark output:")
    print(benchmark_output)
