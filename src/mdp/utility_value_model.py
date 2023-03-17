import copy
import gzip
import json

from src.mdp.transition_model import TransitionModel


class UtilityValueModel:
    def __init__(self, transition_model: TransitionModel, utilities: dict = None, path: str = None):
        self.transition_model = transition_model

        if path:
            utilities = UtilityValueModel.load_from_file(path)

        self.utilities = utilities
        self.expected_utilities = {}
        self._compute_expected_utilities(self.transition_model.get_root(), self.expected_utilities)

    def _compute_expected_utilities(self, state_id, results):
        r = 0
        for child_id, transition_prob in self.transition_model.get_neighbours(state_id).items():
            branch_reward = self.get_utility(child_id) + self._compute_expected_utilities(child_id, results)
            r += transition_prob * branch_reward

        results[state_id] = r
        return r

    def get_utility(self, node_id):
        """
        Reward only when reaching a terminal node.
        Inner node do not have reward, they have expected reward.
        """
        return self.utilities.get(node_id, 0.0)

    def get_expected_utility(self, node_id):
        """
        For terminal node return the utility
        For inner node return the expected utility,
        """
        return self.expected_utilities.get(node_id, 0.0)

    def save_to_file(self, path):
        # to_save = {k: v for k, v in self.utilities.items()}
        with gzip.open(path, "wb") as f:
            f.write(json.dumps(self.utilities).encode())

    @staticmethod
    def load_from_file(path):
        if path.endswith(".txt"):
            utilities = {}
            with open(path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split(" ")
                    for i in range(0, len(parts), 2):
                        utilities[int(parts[i])] = float(parts[i + 1])
        else:
            with gzip.open(path, "rb") as f:
                utilities = json.loads(f.read().decode())
            utilities = {int(k): v for k, v in utilities.items()}

        return utilities
