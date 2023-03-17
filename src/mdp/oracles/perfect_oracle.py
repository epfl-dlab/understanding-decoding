from ..evaluation_models import ExplicitEvaluationModel
from ..trees import Tree


class PerfectOracle(ExplicitEvaluationModel):
    def __init__(self, tree: Tree, path: str):
        super().__init__(tree=tree, path=path, ignore_non_terminal=True, ignore_terminal=False)
        self.propagate_utilities()

    def propagate_utility_from_node(self, node_id):
        utility = self.utilities[node_id]
        curr_node_id = self.tree.get_parent(node_id)

        while curr_node_id != self.tree.NO_PARENT:
            self.est_utilities[curr_node_id] = utility
            curr_node_id = self.tree.get_parent(curr_node_id)

    def propagate_utilities(self):
        for node_id in self.utilities:
            self.propagate_utility_from_node(node_id)
