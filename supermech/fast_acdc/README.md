## Contents of the folder

This file contains various things related to mechanistic interpretability experiments for the refusal mechanism of Llama2.

### `refusal_acdcpp_eap.ipynb`

Where I tried (but sadly failed) to run Edge Attribution Patching using the ACDC++ codebase on Llama2 for the refusal task.

The library (acdcpp)[https://github.com/Aaquib111/acdcpp] has to be imported for running this.
I just cloned the repo, checked out the branch `can-minimal-eap` and created the notebook inside.

### `refusal_custom_eap_heads.ipynb`

Where I tried (and succeded!) running Edge Attribution Patching using the helper functions from (this)[https://github.com/Aaquib111/acdcpp/blob/main/utils/prune_utils.py] file in the acdcpp repo. This notebook can (probably) be run on its own if you just copy the helper functions `get_3_caches` and `split_layers_and_heads` from the acdcpp repo but don't include the rest of the things (since ACDC imports are needed for that).

The batch size is set to 1 (otherwise memory errors) and the metric used is the very simple logit difference between the tokens `Sure` and `Sorry`. Everything is super fast (thanks to EAP being very fast) except for the cells where we calculate the scores for each edge (since there are many many edges, more than 1.5M) and the cell to sort the edges by their importance score.

### `mechinterp_analysis_1.ipynb`

First mech interp experiment on Llama2. The most important finding is the fact that patching in the activations at the position where the `bomb` object is, with activations from the same prompt but with `pie` as object, mislead the model into thinking that making a pie is a dangerous and illegal activity. So we can induce refusal for harmless objects with this.

### `mechinterp_analysis_2.ipynb`

Continues the investigation from the previous notebook focusing on the change in the metric with respect to the layer where the activation patching is performed.

### `refusal_mi.ipynb`

Contains the notebook by Andy Ardity can also be found in his repo (here)[https://github.com/andyrdt/mi/blob/main/SPAR/refusal_mi/refusal_mi.ipynb] where a small set of early (layers 5-10) "refusal" heads are found.

