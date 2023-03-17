from . import BatchedEvaluationModel, OracleGenIE
from .utils import add_noise_binary

from .. import Tree


class NoisyOracleGenIE(BatchedEvaluationModel):
    def __init__(self, noising_function_parameters: dict, tree: Tree = None):
        """

        Parameters
        ----------
        noising_function_parameters : contains sigma – standard_deviation of the truncated Normal noise
        tree: instance of `Tree`
        """
        super().__init__(tree)
        self.noising_function_parameters = noising_function_parameters
        self.oracle = OracleGenIE(tree)

    def apply_noise(self, values):
        return add_noise_binary(values, np_random_state=self.np_random_state, **self.noising_function_parameters)

    def evaluate(self, input_ids, target_ids, next_token_ids, **kwargs):
        values = self.oracle.evaluate(input_ids, target_ids, next_token_ids)

        noisy_values = self.apply_noise(values)
        return noisy_values.to(input_ids.device)
