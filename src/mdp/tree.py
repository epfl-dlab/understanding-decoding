from collections import defaultdict


# class NodeTree(object):
#     def __init__(self, id: int = None, data: dict = None, is_leaf: bool = None):
#         self.id = id
#         self.data = data
#         self.is_leaf = is_leaf
#
#     def __hash__(self):
#         return hash(str(self.id))
#
#     def __eq__(self, other):
#         return self.id == other.id
#
#     def __repr__(self):
#         node_repr = f"{self.id}"
#         return node_repr


class Tree:
    def __init__(self, structure: dict = None):
        if structure:
            self.structure = structure
        else:
            self.structure = defaultdict(lambda: defaultdict(float))

        non_root_states = set([nbs_ids for node_id, nbs in self.structure.items() for nbs_ids in nbs.keys()])
        self.non_terminal_states = set(self.structure.keys())
        self.terminal_states = non_root_states - self.non_terminal_states
        # self.terminal_states = [self.get_node_from_id(str(i)) for i in self.terminal_states]

    def get_root(self):
        return list(self.structure.keys())[0]

    # def get_node_from_id(self, node_id):
    #     if hasattr(self, 'node_dict'):
    #         return self.node_dict[node_id]
    #     non_root_states = set([nbs_ids for node_id, nbs in self.structure.items() for nbs_ids in nbs.keys()])
    #     self.node_dict = {k.id: k for k in non_root_states}
    #     root = self.get_root()
    #     self.node_dict[root.id] = root
    #     return self.node_dict[node_id]

    def terminals_from_node(self, node):
        if node not in self.structure:
            return [node]
        else:
            terminals = []
            for child in self.structure[node]:
                terminals.extend(self.terminals_from_node(child))
            return terminals

    def get_neighbours(self, state_id):
        return self.structure.get(state_id, {})

    def nb_nodes(self):
        non_root_states = set([nbs_ids for node_id, nbs in self.structure.items() for nbs_ids in nbs.keys()])
        return len(non_root_states) + 1

    # def visualize(self):
    #     def _edge_to_str(e_id, **kwargs):
    #         desc = "\n".join([f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}" for key, value in
    #                           kwargs.items()])
    #         desc = desc.strip('\n')
    #         return (f"{e_id}\n{desc}")
    #
    #     graph = pygraphviz.AGraph(directed=True)
    #
    #     # add root
    #     root = self.get_root()
    #     graph.add_node(root.id, label=str(root), color="green")
    #
    #     for node in sorted(self.non_terminal_states):
    #         children = self.get_neighbours(node)
    #         for child, tr_p in children.items():
    #             graph.add_node(child.id,
    #                            label=str(child),
    #                            color="red")
    #
    #             # p = tr_m.get_transition_prob(node_id, child_id)
    #             # r = r_m.get_reward(node_id, child_id)
    #             graph.add_edge(node.id, child.id, label=edge_to_str(e_id=child.id))
    #
    #     return graph
