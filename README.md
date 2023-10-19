All the code and data is inside the `supermech` folder such that everything is inside the `supermech` python package and I can import it and use it elsewhere. Some files (datasets and activation vectors) aren't in the repo because they are too large to be uploaded into github without LFS.

### `supermech/llm_jailbreak`
Contains all the code to run a progressive multi-prompt suffix attack and store any jailbreaks found in the `supermech/llm_jailbreak/jailbreak_db.json` file.

 Example code to set up and start a simple attack with just one goal can be found at `supermech/llm_jailbreak/main.py`.
 
 The file containing the class `SimplifiedMultiPromptAttack` to perform the attack is at `supermech/llm_jailbreak/attack_manager.py`. Everything there has been taken from the repo [https://github.com/llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks) and simplified a bit to make it easier to modify and understand.
 
 I have put everything into just one class (`SimplifiedMultiPromptAttack`) instead of having three nested classes to do an attack. I have also added the option to have a custom loss function based on the internal activations of the model. And lastly, I have also changed the way the prompt is tokenized to match how the usual HF Llama2 tokenization is done, because I was having trouble using jailbreaks for Llama2 in different settings with the previous tokenization based on the fastchat library.

### `supermech/act_engineering`
Generating activation vectors for different concepts (refusal and harmfulness) and doing experiments/visualizations to test how well they work.

The most interesting file here is `harm_reading_vector.ipynb` where I try to reproduce the harmlessness reading vector generated from the [Representation Engineering paper](https://arxiv.org/pdf/2310.01405.pdf).

### `supermech/fast_acdc`
In `llama2_mechinterp.ipynb` I perform a high-level mechanistic analysis of the model internals.  I run the model on just two prompts ("Tell me how to make a bomb" and "Tell me how to make a pie") where I replace a single word ("bomb" -> "pie") to interpret the changes within the model's activations between the two prompts.




