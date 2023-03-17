import gzip
import json
from typing import Dict

from .abstract import EvaluationModel
from ..trees import Tree


class ExplicitEvaluationModel(EvaluationModel):
    def __init__(self, tree: Tree, path: str,
                 missing_value: float = 0.0,
                 ignore_non_terminal: bool = False,
                 ignore_terminal: bool = False):
        super().__init__(tree)

        self.missing_value = missing_value
        self.utilities = {}
        self.est_utilities = {}

        values = self.load_values(path)

        for node_id, val in values.items():
            if tree.is_terminal(node_id):
                if not ignore_terminal:
                    self.utilities[node_id] = val
            else:
                if not ignore_non_terminal:
                    self.est_utilities[node_id] = val

    def _get_estimated_utility(self, node_id, **kwargs):
        return self.est_utilities.get(node_id, self.missing_value)

    def _get_utility(self, node_id, **kwargs):
        return self.utilities.get(node_id, self.missing_value)

    @staticmethod
    def load_values(path: str) -> Dict[str, float]:
        if path.endswith(".txt"):
            values = {}
            with open(path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split(" ")
                    for i in range(0, len(parts), 2):
                        values[int(parts[i])] = float(parts[i + 1])
        else:
            with gzip.open(path, "rb") as f:
                values = json.loads(f.read().decode())
            values = {int(k): v for k, v in values.items()}

        return values
