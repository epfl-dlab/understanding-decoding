from .abstract import EvaluationModel
from .. import Tree


class LikelihoodEvaluationModel(EvaluationModel):
    def __init__(self, tree: Tree):
        super().__init__(tree)

    def _get_estimated_utility(self, node_id, **kwargs):
        """Estimated utility -- goodness -- of an internal node"""
        return self._get_utility(node_id, **kwargs)

    def _get_utility(self, node_id, **kwargs):
        """Utility of a terminal node"""
        assert 'likelihood' in kwargs
        return kwargs['likelihood']
