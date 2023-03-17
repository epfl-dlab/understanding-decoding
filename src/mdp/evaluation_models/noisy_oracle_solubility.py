import torch

from . import BatchedEvaluationModel, OracleSolubility
from .utils import add_noise_continuous
from ..trees import Tree


class NoisyOracleSolubility(BatchedEvaluationModel):
    def __init__(self, noising_function_parameters: dict, tree: Tree, oracle_parameters: dict):
        """
        See the OracleDetoxify docstring.
        Parameters
        ----------
        noising_function_parameters : contains probability_inversion – the probability of swapping the value between two actions
        """
        super().__init__(tree)
        self.noising_function_parameters = noising_function_parameters
        self.oracle = OracleSolubility(tree, **oracle_parameters)

    def apply_noise(self, values):
        return torch.cat([add_noise_continuous(v, **self.noising_function_parameters)[None, :] for v in values])

    def evaluate(self, input_ids, next_token_ids, **kwargs):
        """

        Parameters
        ----------
        input_ids : (num_samples, curr_len)
        next_token_ids : (num_samples, num_considered_tokens)
        kwargs :

        Returns
        -------
        values : (num_samples, num_considered_tokens)
        """
        values = self.oracle.evaluate(input_ids, next_token_ids)

        noisy_values = self.apply_noise(values)
        return noisy_values

    def reset_device(self, device: torch.device):
        self.oracle.reset_device(device)
