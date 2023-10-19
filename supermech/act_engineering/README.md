### [Harm Vector 1](https://github.com/obalcells/supermech/tree/main/supermech/act_engineering/harm_vector)
Generated with A/B answer pairs by conditioning the model on having answered or having refused to answer questions on personal preferences with illegal instructions.

#### Details
**Dataset**: [`harmful_data_A_B.json`](https://github.com/obalcells/supermech/blob/main/supermech/datasets/custom_harm_dataset/harmful_data_A_B.json)
**System Prompt**: `You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.`
**Prompt**:
```
[INST]
{system_prompt}
{instruction}
Choices:
(A) {refusal answer}
(B) {answer}
[/INST]
(A)
```

We randomize the position of the refusal answer such that it's in (A) half of the time. We get the activations of the position -2, when the model has to choose between (A) or (B)


### [Harm Vector 2](https://github.com/obalcells/supermech/blob/main/supermech/act_engineering/harm_vector_2)
We take 300 dangerous/harmful instructions from [advbench](https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench) and pair each instruction with a harmless instruction from the data used to fine-tune the Alpaca 7B model ([data here](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)). We then get the activations at the last token position (the instruction token) and get the first SVD eigenvector from the matrix for each of the layers.

#### Details
- **Dataset**: [`harmful_instructions.json`](https://github.com/obalcells/supermech/blob/main/supermech/datasets/custom_harm_dataset/harmful_instructions.json) and [`harmless_instructions.json`](https://github.com/obalcells/supermech/blob/main/supermech/datasets/custom_harm_dataset/harmless_instructions.json)
- **System Prompt**: `You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.`
**Prompt**:
```
[INST]
{system_prompt}
{instruction}
[/INST] 
```
There's a space after the `[/INST]`  string. We extract the activations at the last token for both harmful and harmless prompts. For each pair of `(harmful_instruction_activations, harmless_instruction_activations)` we subtract the two and get a matrix of `[n_layers n_samples d_model]`. We compute the SVD across the last two dimensions (once for each layer) and extract the first eigenvector for each layer.

### [Refusal Vector](https://github.com/obalcells/supermech/blob/main/supermech/act_engineering/refusal_vector)
See [here](https://github.com/nrimsky/LM-exp/blob/main/refusal/refusal_steering.ipynb) for more details.

We use A/B two-choice answers and condition the model on having chosen (A) or (B)
#### Details
**Dataset**: [`refusal_data_A_B.json`](https://github.com/obalcells/supermech/blob/main/supermech/datasets/nina_rimsky_refusal_data/refusal_data_A_B.json)
**System Prompt**: `You are a helpful, honest and concise assistant.`
**Prompt**:
```
[INST]
{system_prompt}
{instruction}
Choices:
(A) {refusal answer}
(B) {answer}
[/INST]
(A)
```

We randomize the position of the refusal answer such that it's in (A) half of the time. We get the activations of the position -2, when the model has to choose between (A) or (B)

See https://github.com/nrimsky/LM-exp/tree/main/refusal

### [Jailbreak vector](https://github.com/obalcells/supermech/blob/main/supermech/act_engineering/jailbreak_vector)
We pass instructions with an adversarial suffix trained to elicit a dangerous/harmful response from the model and also instructions with a random suffix (randomly chosen tokens with similar length as the adversarial suffix).