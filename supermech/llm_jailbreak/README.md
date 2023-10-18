
### `main.py`
Contains the main code to set up the attack and run it.

### `attack_manager.py`
Contains all the code that handles the attack. The whole attack is just one class `SimplifiedMultiPromptAttack` that executes a multiprompt attack with a given set of prompts and their corresponding target responses.

The code is a simplified version of the `ProgressiveMultiPromptAttack` class from the paper [https://arxiv.org/abs/2307.15043](https://arxiv.org/abs/2307.15043). The original code can be found [here](https://github.com/llm-attacks/llm-attacks/blob/main/llm_attacks/base/attack_manager.py#L819C7-L819C35).

### `llama2.py`
A wraapper for the llama2 model such that we can get and intervene on the intermediate activations of the llama2 huggingface model.

All the other files/folders in here aren't important.

### `/modal_files`
Contains some files used to set up a container in the cloud and run the attack there. The attack code hasn't been updated and it will probably not work.

### `/demos`
Contains some old demos of the attack.



