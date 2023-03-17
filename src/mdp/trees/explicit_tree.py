import gzip

from typing import Tuple, Dict

from .abstract import Tree


class ExplicitTree(Tree):
    def __init__(self, path: str = None, eps: float = 1e-4):
        super().__init__()

        self.eps = eps

        tr_probs, self.parents = ExplicitTree.load_transition_probs_and_parents(path)
        self.root_id = next(iter(tr_probs))
        self.nodes = {self.root_id: {"tr_probs": {}}}

        non_root_states = set()
        for node_id, children in tr_probs.items():
            self.nodes[node_id] = {"tr_probs": children}
            non_root_states.update(children)

        for node_id in non_root_states:
            if node_id not in self.nodes:
                self.nodes[node_id] = {"tr_probs": {}}

        self.root_id = next(iter(self.nodes))
        assert self.root_id not in self.parents
        self.parents[self.root_id] = self.NO_PARENT

        self.non_terminal_states = set([node_id for node_id, node_data in self.nodes.items() if node_data["tr_probs"]])
        self.terminal_states = non_root_states - self.non_terminal_states

    def get_marginal_probability(self, state_id):
        return self.nodes[state_id]["marginal_prob"]

    def _calculate_marginal_probabilities(self, state_id=None, state_p=None):
        if state_id is None and state_p is None:
            state_id = self.root_id
            state_p = 1

        self.nodes[state_id]["marginal_prob"] = state_p

        for nb_id, tr_p in self.get_children(state_id).items():
            self._calculate_marginal_probabilities(nb_id, state_p * tr_p)

    def _verify_marginal_prob_validity(self):
        sum_of_terminal_marginal_probs = sum(
            [self.get_marginal_probability(state_id) for state_id in self.terminal_states]
        )
        assert (
            abs(sum_of_terminal_marginal_probs - 1) < self.eps
        ), f"Marginal probabilities of terminal states sum to {sum_of_terminal_marginal_probs} – they should sum to 1"

    def get_children(self, node_id: str) -> Dict[str, float]:
        return self.nodes[node_id]["tr_probs"]

    def get_parent(self, node_id: str) -> str:
        return self.parents[node_id]

    def is_terminal(self, node_id: str) -> bool:
        return node_id in self.terminal_states

    def get_transition_prob(self, src_id: str, trg_id: str) -> float:
        return self.nodes[src_id]["tr_probs"].get(trg_id, 0.0)

    def nb_nodes(self):
        return len(self.parents)

    @staticmethod
    def load_transition_probs_and_parents(path: str) -> Tuple[Dict[str, float], Dict[str, str]]:
        lines = ExplicitTree._read_lines(path)

        tr_probs = {}
        parents = {}

        for line in lines:
            parts = line.strip().split(" ")

            state_id = int(parts[0])

            # neighbours and transition probabilities
            children = {}
            for i in range(1, len(parts), 2):
                node = int(parts[i])
                transition_prob = float(parts[i + 1])
                children[node] = transition_prob
                parents[node] = state_id

            # add to dictionary of dictionaries
            tr_probs[state_id] = children
        return tr_probs, parents

    @staticmethod
    def _read_lines(path):
        if path.endswith(".txt"):
            with open(path, "r") as f:
                lines = f.readlines()
        else:
            with gzip.open(path, "rb") as f:
                lines = [line.decode() for line in f.readlines()]

        return lines

    def node_str_rep(self, node_id, **kwargs):
        node_rep = node_id
        if kwargs.get("decode_node_id", False):
            node_rep = self.tree.tokenizer.decode(node_id)

        label = f'ID: "{node_rep}"'
        if "add_marginal_probs" in kwargs:
            term_to_add = f"mProb: {self.nodes[node_id]['marginal_prob']:.3f}"
            label = self.add_term_to_label(label, term_to_add)

        return label

    def edge_str_rep(self, src_node_id, trg_node_id, **kwargs):
        label = ""
        if kwargs.get("add_action_scores", False):
            term_to_add = f"{self.get_transition_prob(src_node_id, trg_node_id):.2f}"
            label = self.add_term_to_label(label, term_to_add)

        return label
