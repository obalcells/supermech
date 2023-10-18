
def sample_random_activation(dataset, node):
    new_x = dataset.sample()
    # compute the activation for this node with a new input (new context)
    return node.activation(new_x)

# we sample a datapoint from the distribution such that the
# output of our node in our subgraph should stay the same
def sample_equivalent_activation(dataset, node, condition):
    dataset_given_condition = [x if condition(x) == True for x in dataset]
    new_x = dataset_given_condition.sample()
    return node.activation(new_x)

def main():
    # we have:
    # - A circuit condition such as "identify the indirect object in the sentence"
    # - A subgraph G' contained in the computational graph G of the transformer
    #   which contains (under the alternative hypothesis) only the only nodes in the graph
    #   which are responsible for the circuit and nothing else

    dataset_given_condition = [x if circuit_condition(x) == True for x in dataset]

    logit_diff = []

    for i in range(n_scrubs):
        x = dataset_given_condition.sample()
        actual_logits = model(x)
        altered_logits = scrub(output_node, x, subgraph, dataset)
        logit_diff.append(actual_logits - altered_logits)

    mean(logit_diff)

def scrub(node, x_ref, subgraph, dataset):
    # if this is an input node we just return the value we get
    if node.is_input_node == True:
        return x_ref

    # node.condition could be something equivalent to:
    # "this attention head at index t (in the context), attends to the last
    # previous token in the context which is equal to the token at index t"

    parent_activations = {}

    # all the nodes that I depend on
    for parent in node.parents:
        # this is not an important parent
        # just resample
        if parent not in subgraph:
            parent_activations[parent] = sample_random_activation(dataset, parent)
        else:
            new_x = sample_equivalent_activation(dataset, node, node.condition)
            parent_activations[parent] = causal_scrub(parent, new_x, subgraph, dataset)

    activation_here = node.activation(parent_activations)

    return activation_here




