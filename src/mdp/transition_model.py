import gzip
from collections import defaultdict

import scipy.stats as stats
import numpy as np

from src.mdp.tree import Tree


class TransitionModel(Tree):
    def __init__(self, structure: dict = None, path: str = None, eps: float = 1e-4, check_tree: bool = True):
        super().__init__()

        assert structure or path, "You should instantiate a likelihood tree with either a structure or a path"

        self.eps = eps

        if structure:
            self.structure = structure
        else:
            self.structure = TransitionModel.load_from_file(path)

        if check_tree:
            self._verify_validity()

        non_root_states = set([nbs_ids for node_id, nbs in self.structure.items() for nbs_ids in nbs.keys()])
        self.non_terminal_states = set([n for n in self.structure.keys()])
        self.terminal_states = non_root_states - self.non_terminal_states

        if check_tree:
            self.marginal_probs = {}
            self._calculate_marginal_probabilities(self.get_root(), 1)
            self._verify_marginal_prob_validity()

    def get_transition_prob(self, source_state, target_state):
        if source_state not in self.structure:
            return 0.0

        if target_state not in self.structure[source_state]:
            return 0.0

        return self.structure[source_state][target_state]

    def _verify_validity(self):
        # Verify that the transition model is valid
        for state_id, nbs in self.structure.items():
            sum_of_transition_probs = sum(nbs.values())
            assert (
                abs(sum_of_transition_probs - 1) < self.eps
            ), f"Transition probabilities sum to {sum_of_transition_probs} – they should sum to 1"

    def _verify_marginal_prob_validity(self):
        sum_of_terminal_marginal_probs = sum([self.marginal_probs[state_id] for state_id in self.terminal_states])
        assert (
            abs(sum_of_terminal_marginal_probs - 1) < self.eps
        ), f"Marginal probabilities of terminal states sum to {sum_of_terminal_marginal_probs} – they should sum to 1"

    def _calculate_marginal_probabilities(self, state_id, state_p):
        self.marginal_probs[state_id] = state_p

        for nb_id, tr_p in self.get_neighbours(state_id).items():
            self._calculate_marginal_probabilities(nb_id, state_p * tr_p)

    def _best_output_from_node(self, node):
        all_descendents = self.terminals_from_node(node)
        scored_descendent = [self.marginal_probs[s] for s in all_descendents]
        return sorted(zip(all_descendents, scored_descendent), key=lambda item: item[1], reverse=True)[0]

    def get_next_token_entropy(self):
        return np.mean([stats.entropy(list(children.values())) for _, children in self.structure.items()])

    def get_top_n_outputs(self, n):
        terminal_probs = [self.marginal_probs[s] for s in self.terminal_states]
        return sorted(zip(self.terminal_states, terminal_probs), key=lambda item: item[1], reverse=True)[:n]

    def get_partial_misalignment(self):
        pass
        # partial_misalignment = 0
        # for node, children in self.structure.items():
        #     child_prob = list(children.values())
        #     child_best_output = [self._best_output_from_node(child)[1] for child in children.keys()]
        #
        #     # marg_prob = [self.marginal_probs[child] for child in children.keys()]
        #     partial_misalignment += stats.kendalltau(child_prob, child_best_output)[0]
        # return partial_misalignment / len(self.structure)

    def save_to_file(self, path):
        with gzip.open(path, "wb") as f:
            for node, children in self.structure.items():
                line = [str(node)]
                [line.extend([str(child), str(p)]) for child, p in children.items()]
                line_str = " ".join(line) + "\n"
                f.write(line_str.encode())

    @staticmethod
    def load_from_file(path):
        structure = defaultdict(lambda: defaultdict(float))

        if path.endswith(".txt"):
            with open(path, "r") as f:
                lines = f.readlines()
        else:
            with gzip.open(path, "rb") as f:
                lines = [line.decode() for line in f.readlines()]

        for line in lines:
            parts = line.strip().split(" ")

            state_id = int(parts[0])

            # neighbours and transition probabilities
            children = defaultdict()
            for i in range(1, len(parts), 2):
                node = int(parts[i])
                transition_prob = float(parts[i + 1])
                children[node] = transition_prob

            # add to dictionary of dictionaries
            structure[state_id] = children
        return structure
