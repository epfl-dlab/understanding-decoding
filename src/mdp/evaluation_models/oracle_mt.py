import torch
import numpy as np

from . import BatchedEvaluationModel

from .. import Tree
from ...metrics import BLEUScore


class OracleMT(BatchedEvaluationModel):
    def __init__(self, bleu_score: BLEUScore, tree: Tree = None, operate_on: str = "ids"):
        """
        Parameters
        ----------
        tree: instance of `Tree`
        operate_on: str, either "ids" or "text"
        """
        super().__init__(None)
        self.bleu_score = bleu_score
        self.tree = tree
        self.operate_on = operate_on

    def evaluate(self, input_ids, next_token_ids, target_txt, full_target_ids, num_beams=1, **kwargs):
        """

        Parameters
        ----------
        input_ids : (num_samples, curr_len)
        next_token_ids : (num_samples, num_considered_tokens)
        target_txt : (num_samples) target text per sample
        full_target_ids : (num_samples) target ids per sample
        num_beams : the number of beams used by the decoding algorithm
        kwargs :

        Returns
        -------
        values : (num_samples, num_considered_tokens)
        """
        # ToDo: Can be speed up if necessary, by computing the BLEU per prefix and incorporating the value
        #  contributed by the next token
        num_considered_tokens = next_token_ids.shape[-1]
        next_token_ids_f = torch.flatten(next_token_ids)[:, None]

        # Get input texts
        input_ids_r = torch.repeat_interleave(input_ids, num_considered_tokens, dim=0)
        extended_input_ids = torch.cat([input_ids_r, next_token_ids_f], dim=1)

        if self.operate_on == "text":
            # Get current and target text
            text = self.tree.tokenizer.batch_decode(extended_input_ids, skip_special_tokens=True)
            target_txts = np.repeat(target_txt, num_considered_tokens * num_beams)
            values = np.array([self.bleu_score.compute([h], [[r[:len(h)]]], return_score_only=True) if h != "" else 0
                               for h, r in zip(text, target_txts)])

        elif self.operate_on == "ids":
            target_ids_r = torch.repeat_interleave(full_target_ids, num_considered_tokens * num_beams, dim=0)
            values = []
            for h, r in zip(extended_input_ids, target_ids_r):
                h = h.tolist()  # remove padding
                r = r[r != 1][:len(h)].tolist()  # Remove padding and get only the first len(h) tokens
                values.append(self.bleu_score.compute([" ".join(map(str, h))],
                                                      [[" ".join(map(str, r))]],
                                                      return_score_only=True))
        else:
            raise ValueError("Unknown type to operate on: {}".format(self.operate_on))


        values = torch.tensor(values).view_as(next_token_ids)

        return values.to(input_ids.device)
