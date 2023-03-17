from . import BatchedEvaluationModel

import torch

from .. import Tree


class OracleGenIE(BatchedEvaluationModel):
    def __init__(self, tree: Tree = None):
        super().__init__(tree)

    def evaluate(self, input_ids, target_ids, next_token_ids, **kwargs):
        """

        Parameters
        ----------
        input_ids : (num_samples, curr_len)
        target_ids : (num_samples, max_target_len)
        next_token_ids : (num_samples, num_considered_tokens)
        kwargs :

        Returns
        -------
        values : (num_samples, num_considered_tokens)
        """

        # check if already generated tokens match
        curr_len = input_ids.shape[-1]
        if target_ids.shape[-1] <= curr_len:
            return torch.zeros_like(next_token_ids)
        prev_match = torch.all(torch.eq(input_ids, target_ids[:, :curr_len]), dim=-1, keepdim=True)

        # check if the next_token_id equals the next token
        values = torch.eq(target_ids[:, curr_len][:, None], next_token_ids)

        # check if prev_tokens_match and next_token_matches
        values = (prev_match & values).long()

        return values
