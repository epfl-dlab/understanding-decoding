from abc import ABC

import torch

from .utils import get_np_random_state
from ..trees.abstract import Tree


class EvaluationModel(ABC):
    def __init__(self, tree: Tree):
        self.tree = tree

    def _get_estimated_utility(self, node_id, **kwargs):
        """Estimated utility -- goodness -- of an internal node"""
        raise NotImplementedError()

    def _get_utility(self, node_id, **kwargs):
        """Utility of a terminal node"""
        raise NotImplementedError()

    def evaluate(self, node_id, **kwargs):
        if self.tree.is_terminal(node_id):
            return self._get_utility(node_id, **kwargs)

        return self._get_estimated_utility(node_id, **kwargs)

    def reset_device(self, device: torch.device):
        """
        Reset the device on which the evaluation model is and move it to the given device.
        If the evaluation model does not use any device (except CPU), this method can be left empty.
        """
        pass


class BatchedEvaluationModel(ABC):
    def __init__(self, tree: Tree):
        self.tree = tree
        self.rank = 0  # The rank of the thread if running using multiple threads

        self.np_random_state = None
        self.reset_random_state()

    def evaluate(self, input_ids, **kwargs):
        raise NotImplementedError()

    def reset_device(self, device: torch.device):
        """
        Reset the device on which the evaluation model is and move it to the given device.
        If the evaluation model does not use any device (except CPU), this method can be left empty.
        """
        pass

    def reset_rank(self, rank):
        self.rank = rank

    def reset_random_state(self):
        self.np_random_state = get_np_random_state(self.rank)
