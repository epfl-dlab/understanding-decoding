from typing import Dict

#import pygraphviz
from abc import ABC


class Tree(ABC):
    NO_PARENT = -1

    def __init__(self):
        self.root_id = None
        self.nodes = None  # node specific information accessible using getters i.e. nodes[node_id][key_id]

    def get_root(self) -> str:
        return self.root_id

    def get_children(self, node_id: str) -> Dict[str, float]:
        """

        Returns
        -------
        Iterable[NodeID: str]
        """
        raise NotImplementedError()

    def get_parent(self, node_id: str) -> str:
        raise NotImplementedError()

    def is_terminal(self, node_id: str) -> bool:
        raise NotImplementedError()

    def get_transition_prob(self, src_id: str, trg_id: str) -> float:
        raise NotImplementedError()

    def nb_nodes(self):
        return NotImplementedError()

    # Is this necessary?
    # def in_tree(self):
    #   return NotImplementedError()
    #
    # def terminals_from_node(self, node_id):
    #     if self.is_terminal(node_id):
    #         return [node_id]
    #     else:
    #         children = node_id.get_children()
    #         terminals = []
    #         for child in children:
    #             terminals.extend(self.terminals_from_node(child))
    #         return terminals

    # def _get_nodes(self, current_node, result=[]):
    #     result.append(current_node)
    #     children = current_node.get_children()
    #     for child in children:
    #         self._get_nodes(child, result)
    #
    # def get_nodes(self):
    #     results = []
    #     self._get_nodes(self, self.root, results)
    #     return results

    @staticmethod
    def add_term_to_label(label, term_to_add):
        if label.strip() != "":
            label += "\n"

        label += term_to_add
        return label

    def node_str_rep(self, node_id, **kwargs):
        node_rep = node_id
        if kwargs.get("decode_node_id", False):
            node_rep = self.tree.tokenizer.decode(node_id)

        label = f'ID: "{node_rep}"'
        if "add_marginal_probs" in kwargs:
            term_to_add = f"mProb: {self.nodes[node_id]['marginal_prob']:.3f}"
            label = self.add_term_to_label(label, term_to_add)

        return label

    def _visualize_node(self, graph, node_id, **kwargs):
        if node_id == self.root_id:
            color = "green"
        else:
            if kwargs.get("mark_terminal_nodes", True) and self.is_terminal(node_id):
                color = "purple"
            else:
                color = "red"

        graph.add_node(node_id, color=color, label=self.node_str_rep(node_id, **kwargs))

    def edge_str_rep(self, src_node_id, trg_node_id, **kwargs):
        label = ""
        if kwargs.get("add_action_scores", True):
            term_to_add = f"{self.get_transition_prob(src_node_id, trg_node_id):.2f}"
            label = self.add_term_to_label(label, term_to_add)

        return label

    def _visualize_edge(self, graph, src_node_id, trg_node_id, **kwargs):
        graph.add_edge(src_node_id, trg_node_id, label=self.edge_str_rep(src_node_id, trg_node_id, **kwargs))

    def _visualize_tree(self, graph, node_id, max_depth, depth=0, visualize_node=None, visualize_edge=None, **kwargs):
        if visualize_node is None:
            self._visualize_node(graph, node_id, **kwargs)
        else:
            visualize_node(self, graph, node_id, **kwargs)

        if max_depth is not None and depth >= max_depth:
            return

        children = self.get_children(node_id)
        for child in children:
            self._visualize_tree(
                graph,
                child,
                max_depth=max_depth,
                depth=depth + 1,
                visualize_node=visualize_node,
                visualize_edge=visualize_edge,
                **kwargs,
            )

            if visualize_edge is None:
                self._visualize_edge(graph, node_id, child, **kwargs)
            else:
                visualize_edge(self, graph, node_id, child, **kwargs)

    def visualize_tree(self, max_depth=None, visualize_node=None, visualize_edge=None, **kwargs):
        graph = pygraphviz.AGraph(directed=True)
        self._visualize_tree(
            graph,
            self.root_id,
            max_depth=max_depth,
            visualize_node=visualize_node,
            visualize_edge=visualize_edge,
            **kwargs,
        )

        return graph
