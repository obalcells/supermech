from typing import List
from torch.utils.data import Dataset
from torch import Tensor
from transformers import AutoTokenizer
from jaxtyping import Int, Float
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

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
    prompt = f"{B_INST} {dialog_content} {E_INST}{model_output.strip()}"
    return prompt

def tokenize_instructions(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    padding_length: int=None,
    system_prompt: str="",
    model_outputs: List[str]=None
) -> Int[Tensor, "batch seq_len"]:
    if model_outputs:
        assert(len(instructions) == len(model_outputs))
        prompts = [instruction_to_prompt(instruction, system_prompt, model_output) for (instruction, model_output) in zip(instructions, model_outputs)]
    else:
        prompts = [instruction_to_prompt(instruction, system_prompt) for instruction in instructions]

    instructions_toks = tokenizer(
        prompts,
        padding="max_length" if padding_length is not None else True,
        max_length=padding_length,
        truncation=False,
        return_tensors="pt"
    ).input_ids

    return instructions_toks

def generate_from_instructions(
    tl_model: HookedTransformer,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    max_new_tokens: int=64,
    temperature: float=0.0,
):
    instructions_toks = tokenize_instructions(tokenizer, instructions).cuda()

    if temperature == 0.0:
        output_ids = tl_model.generate(instructions_toks, max_new_tokens=max_new_tokens, do_sample=False)
    else:
        output_ids = tl_model.generate(instructions_toks, max_new_tokens=max_new_tokens, temperature=temperature)
    for answer_idx, answer in enumerate(tokenizer.batch_decode(output_ids)):
        print(f"\nGeneration #{answer_idx}:\n\t{repr(answer)}")

DEFAULT_INSTRUCTION_TEMPLATE = "Tell me how to make a {object}."

class InstructionDataset(Dataset):
    def __init__(
        self,
        objects: List[str],
        tokenizer: AutoTokenizer,
        padding_length: int,
        instruction_template: str = DEFAULT_INSTRUCTION_TEMPLATE,
    ):
        self.objects = objects
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.instruction_template = instruction_template

        self.prompt_toks = tokenize_instructions(
            tokenizer,
            instructions=[self.instruction_template.format(object=object) for object in objects],
            padding_length=self.padding_length
        )
        self.prompt_strs = [tokenizer.decode(self.prompt_toks[i]) for i in range(len(self.objects))]
        self.prompt_str_toks = [
            [tokenizer.decode(self.prompt_toks[i, j]) for j in range(padding_length)]
            for i in range(len(self.objects))
        ]

        self.object_tok_pos = self._get_last_object_tok_pos()

    def __len__(self):
        return len(self.objects)
    
    def __getitem__(self, idx):
        return self.prompt_toks[idx]

    def _get_last_object_tok_pos(self):
        single_tok_object = "pie"
        single_tok_object_toks = tokenize_instructions(
            self.tokenizer,
            instructions=[self.instruction_template.format(object=single_tok_object)],
            padding_length=self.padding_length
        )
        return [self.tokenizer.decode(tok) for tok in single_tok_object_toks[0]].index(single_tok_object)

class PairedInstructionDataset:
    def __init__(
        self,
        harmful_objects: List[str],
        harmless_objects: List[str],
        tokenizer: AutoTokenizer,
        prompt_template: str = DEFAULT_INSTRUCTION_TEMPLATE,
    ):
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template

        max_length = self._find_max_length(harmful_objects + harmless_objects, tokenizer, prompt_template)

        self.harmful_dataset = InstructionDataset(harmful_objects, tokenizer, max_length, prompt_template)
        self.harmless_dataset = InstructionDataset(harmless_objects, tokenizer, max_length, prompt_template)

    def _find_max_length(self, objects: List[str], tokenizer: AutoTokenizer, prompt_template: str):
        prompt_toks = tokenize_instructions(
            tokenizer,
            [prompt_template.format(object=object) for object in objects]
        )
        return prompt_toks.shape[1]