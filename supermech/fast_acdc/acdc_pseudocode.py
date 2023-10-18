

class TLACDCCorrespondence:
    # store the full computational graph
    
    # what do these objects mean? -> I guess a dictionary?
    # why the str? why a mutable mapping? -> A dictionary (: ?
    # what is a TorchIndex?
    # and most importantly, what is the TLACDCInterpNode?
    graph: MutableMapping[str, MutableMapping[TorchIndex, TLCACDCInterpNode]]
    edges: MutableMapping[str, MutableMapping[TorchIndex, MutableMapping[str, MutableMapping[TorchIndex, Edge]]]]

    # how are things indexed by these two objects?
    def add_edge(
        self,
        parent_node: TLACDCInterpNode,
        child_node: TLACDCInterpNode,
        edge: Edge,
        safe=True
    ):
        # we add the parent/child if it's not in the graph yet
        if parent_node not in self.nodes():
            self.add_node(parent_node)
        if child_node not in self.nodes():
            self.add_node(child_node)
    
        parent_node._add_child(child_node)
        child_node._add_parent(parent_node)

        # what is the index? Why do we use it for indexing when we have the name already?
        self.edges[child_node.name][child_node.index][parent_node.name][parent_node.index] = edge


    @classmethod
    def setup_from_model(cls, model, use_pos_embed=False):
        # what is cls? -> I guess itself? The graph object?
        correspondence = cls()

        downstream_residual_nodes: List[TLACDCInterpNode] = []

        # the first example of a node in the graph (:
        logits_node = TLACDCInterpNode(
            name=f"blocks.{model.cfg.n_layers-1}.hook_resid_post",
            index=TorchIndex([None]), # mmm alright?
            incoming_edge_type = EdgeType.ADDITION,
        )

        # Ahh I guess the slice tells you were the node is writing to?

        # ahh this will keep increasing such that any
        # component is only tied with the residual nodes in later layers
        downstream_residual_nodes.append(logits_node)

        # we add this node to the graph
        correspondence.add_node(logits_node)

        # we start from the end of the transformer
        for layer_idx in range(model.cfg.n_layers - 1, -1, -1):
            # we don't do this if we don't have MLPs

            #########
            # 1) MLP Out
            cur_mlp_name = f"blocks.{layer_idx}.hook_mlp_out"
            cur_mlp_slice = TorchIndex([None]) # ?
            cur_mlp = TLACDCInterpNode(
                name=cur_mlp_name,
                index=TorchIndex([None])
                incoming_edge_type=EdgeType.PLACEHOLDER,
            )

            # we add an edge to all future residual streams  
            for residual_stream_node in downstream_residual_nodes:
                correspondence.add_edge(
                    parent_node=cur_mlp,
                    child_node=residual_stream_node,
                    edge=Edge(edge_type=EdgeType.ADDITION)

            ##########
            # 2) MLP In
            cur_mlp_input_name = f"blocks.{layer_idx}.hook_mlp_in"
            cur_mlp_input_slice = TorchIndex([None])
            cur_mlp_input = TLACDCInterpNode(
                name=cur_mlp_input_name,
                index=cur_mlp_input_slice,
                incoming_edge_type=EdgeType.ADDITION
            )

            # Question (before looking at the code): What edge do we have to add here?
            # Answer: From the previous residual stream? To the mlp_out?
            # Solution: Only to mlp out. The mlp in is treated as a downstream_residual_node

            correspondence.add_edge(
                parent_node=cur_mlp_input,
                child_node=cur_mlp,
                edge=Edge(edge_type=EdgeType.PLACEHOLDER)
                safe=False # it's never safe!!!
            )
        
            downstream_residual_nodes.append(cur_mlp_input)

            ###########
            # 3) Attention
            # why the other way around here? Is it necessary?
            for head_idx in range(model.cfg.n_heads - 1, -1, -1):
                # everything is the same but can you guess the
                # slice that we'll have for the head?
                # and how do we separate it between q,k,v?

                # first we have the node representing the attention head output, similarly as we did before with mlp out
                # the only difference is we have a slice 

                # alright, this is how we define our slice then
                cur_head_name = f"blocks.{layer_idx}.attn.hook_result"
                cur_head_slice = TorchIndex([None, None, head_idx])
                cur_head = TLACDCInterpNode(
                    name=cur_head_name,
                    index=cur_head_slice,
                    incoming_edge_type=EdgeType.PLACEHOLDER
                )

                correspondence.add_node(cur_head)

                for residual_stream_node in downstream_residual_nodes:
                    correspondence.add_edge(
                        parent_node=cur_head,
                        child_node=residual_stream_node,
                        edge=Edge(edge_type=EdgeType.ADDITION),
                        safe=False,
                    )

                for letter in "qkv":
                    hook_letter_name = f"blocks.{layer_idx}.attn.hook_{letter}"
                    hook_letter_slice = TorchIndex([None, None, head_idx])
                    hook_letter_node = TLACDCInterpNode(name=hook_letter_name, index=hook_letter_slice, incoming_edge_type=EdgeType.DIRECT_COMPUTATION)
                    correspondence.add_node(hook_letter_node)

                    hook_letter_input_name = f"blocks.{layer_idx}.hook_{letter}_input"
                    hook_letter_input_slice = TorchIndex([None, None, head_idx])
                    hook_letter_input_node = TLACDCInterpNode(
                        name=hook_letter_input_name, index=hook_letter_input_slice, incoming_edge_type=EdgeType.ADDITION
                    )
                    correspondence.add_node(hook_letter_input_node)

                    correspondence.add_edge(
                        parent_node = hook_letter_node,
                        child_node = cur_head,
                        edge = Edge(edge_type=EdgeType.PLACEHOLDER),
                        safe = False,
                    )

                    correspondence.add_edge(
                        parent_node=hook_letter_input_node,
                        child_node=hook_letter_node,
                        edge=Edge(edge_type=EdgeType.DIRECT_COMPUTATION),
                        safe=False,
                    )

                    new_downstream_residual_nodes.append(hook_letter_input_node)

                # for the next layer
                downstream_residual_nodes.extend(new_downstream_residual_nodes) 
                


        ########
        # 4) The embedding (also the pos embedding if there is any)
        ...
    
    """    
    # Questions and things to remember:

    # 1) What kind of operations are there?
    #    There are ADDITION and PLACEHOLDER and DIRECT_COMPUTATION edge types
    #    DIRECT_COMPUTATION is only used for the edge between
    #      - The input to q,k or v of a head and the output after applying the W_{letter} matrix
    #      - Isn't this missing the W_O? That's I guess only like a ln layer???
    #      - Why is the MLP input output edge not a computation then?

    # 2) What are the nodes that we define as residual nodes?
    #    The input residual to the mlp
    #    The input residual to q, k or v for a specific head 

    3) What's the difference between the different kinds of edges?
        The DIRECT_COMPUTATION edges are those such that the child node has only one parent
        - What do we do if we want to remove such an edge? I guess we need to read the experiment file?
        The ADDITION means that we have a linear computation so it's easy to remove the
        edge
        The PLACEHOLDER
    """

    
# Now we're going to try to understand how a step is performed
# we're inside the TLACDCExperiment now

class TLACDCExperiment:
    def __init__(self):
        
        # now we call TLACDCCorrespondence.setup_from_model (:

    def step(self):
        initial_metric = self.cur_metric

        cur_metric = initial_metric

        # what are these indices?? They are the torch indices
        # but what is that? Why is that necessary for the indexing? 
        sender_names_list = list(self.corr.edges[self.current_node.name][self.current_node.index][sender_name])

        for sender_name in sender_names_list:
            sender_indices_list = list(self.corr.edges[self.current_node.name] 

            # seems to work better with reversed order?
            sender_indices_list = list(reversed(sender_indices_list))

            # why multiple indices?
            for sender_index in sender_indices_list:
                edge = self.corr.edges[self.current_node.name][self.current_node.index][sender_name][sender_index]
                cur_parent = self.corr.graph[sender_name][sender_index]

                # we always include it if it's of PLACEHOLDER type
                if edge.edge_type == EdgeType.PLACEHOLDER:
                    is_this_node = True
                    continue

                # we delete it by default
                edge.present = False

                if edge.edge_type == EdgeType.ADDITION:
                    added_sender_hook = self.add_sender_hook(
                        cur_parent
                    ) 

                # What's the difference between DIRECT_COMPUTATION and PLACEHOLDER edges 

                old_metric = self.cur_metric
                if self.second_metric is not None:
                    old_second_metric = self.cur_second_metric

                # where everything happens
                self.update_cur_metric(recalc_edges=False)
                evaluated_metric = self.cur_metric

                result = evaluated_metric - old_metric

                edge.effect_size = abs(result)

                if result < self.threshold:
                    self.corr.remove_edge(...)
                else:
                    # we keep the edge
                    self.cur_metric = old_metric        
                    # what is this second metric?
                    if self.second_metric is not None:
                        self.cur_second_metric = old_second_metric
                    
        
"""
Remaining questions:
How is the metric updated? It's just the ioi_metric for example. There's a f for that
Why do we index our graph using the edge index too? 
How does cutting an edge influence the forward pass?
    -> I guess its
"""

    def sender_hook(self, z, hook, cache="online"):





