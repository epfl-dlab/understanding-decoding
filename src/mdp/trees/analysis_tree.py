from typing import Dict

from .abstract import Tree


class AnalysisTree(Tree):
    def __init__(self, tree, root_id=tuple(), root_score=0, root_value=0):
        super().__init__()

        self.tree = tree

        self.root_id = root_id
        self.nodes = {root_id: dict(score=root_score, value=root_value, tr_probs={})}
        self.parents = {root_id: self.NO_PARENT}
        self.action = {root_id: None}

    def get_children(self, node_id: tuple) -> Dict[tuple, float]:
        # ToDo: All vs only in tree
        return self.nodes[node_id]["tr_probs"]

    def get_parent(self, node_id: tuple) -> tuple:
        if node_id in self.parents:
            return self.parents[node_id]

        parent = self.tree.get_parent(node_id)
        self.parents[node_id] = parent
        return parent

    def is_terminal(self, node_id: tuple) -> bool:
        return self.tree.is_terminal(node_id)

    def get_transition_prob(self, src_id: tuple, trg_id: tuple) -> float:
        pass  # ToDo

    def _add_edge_from_parent(self, parent_id, node_id):
        parent_data = self.nodes[parent_id]
        tr_probs = parent_data["tr_probs"]
        if node_id not in tr_probs:
            tr_probs[node_id] = 1  # ToDo: Pass action probabilities in path and update
            parent_data["tr_probs"] = tr_probs
            self.action[node_id] = node_id[-1]

    def _add_node(self, node_id):
        if node_id not in self.nodes:
            self.nodes[node_id] = {"tr_probs": {}}

    # def _add_node(self, node_id, scores):
    #     if self.get_parent == self.root_id:
    #         return
    #
    #     parent_id = self.get_parent(node_id)
    #     self._add_node(self, parent_id, scores[:-1])
    #     self._add_edge_from_parent(parent_id, node_id)
    #     self._add_node(node_id)
    #     self.nodes[node_id]['score'] = scores[-1]

    def add_path(self, node_id, scores, values, label, terminal=True):
        assert isinstance(node_id, tuple), node_id
        if node_id == self.root_id:
            return

        parent_id = self.get_parent(node_id)
        self.add_path(parent_id, scores[:-1], values[:-1], None, terminal=False)
        self._add_edge_from_parent(parent_id, node_id)
        self._add_node(node_id)
        self.nodes[node_id]["score"] = scores[-1] if len(scores) != 0 else None
        self.nodes[node_id]["value"] = values[-1] if len(values) != 0 else None

        if terminal:
            self.nodes[node_id]["label"] = label

    def node_str_rep(self, node_id, **kwargs):
        node_rep = node_id
        if kwargs.get("decode_node_id", False):
            node_rep = self.tree.tokenizer.decode(node_id)

        label = f'ID: "{node_rep}"'
        if 'score' in self.nodes[node_id] and self.nodes[node_id]['score'] is not None:
            term_to_add = f"Score: {self.nodes[node_id]['score']:.3f}"
            label = self.add_term_to_label(label, term_to_add)

        if 'value' in self.nodes[node_id] and self.nodes[node_id]['value'] is not None:
            term_to_add = f"Value: {self.nodes[node_id]['value']:.3f}"
            label = self.add_term_to_label(label, term_to_add)

        if 'label' in self.nodes[node_id]:
            term_to_add = f"Label: {self.nodes[node_id]['label']}"
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

    def _visualize_edge(self, graph, src_node_id, trg_node_id, **kwargs):
        label_rep = self.action[trg_node_id]
        if kwargs.get("decode_action_id", False):
            label_rep = self.tree.tokenizer.decode([label_rep])

        graph.add_edge(
            src_node_id,
            trg_node_id,
            # label=self.get_transition_prob(src_node_id, trg_node_id)
            label=f"~{label_rep}~",
        )
