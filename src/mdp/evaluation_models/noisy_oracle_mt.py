import torch
import numpy as np

from . import BatchedEvaluationModel

from .. import Tree
from ...metrics import BLEUScore


class NoisyOracleMT(BatchedEvaluationModel):
    def __init__(self, noising_function_parameters: dict, bleu_score: BLEUScore,
                 tree: Tree = None, operate_on: str = "ids",
                 l_strip=0):
        """
        Parameters
        ----------
        noising_function_parameters : contains lambda – weight of the linear combination of noisy and true BLUE value
        tree: instance of `Tree`
        operate_on: str, either "ids" or "text"
        """
        super().__init__(tree)
        self.noising_function_parameters = noising_function_parameters
        self.tree = tree
        self.bleu_score = bleu_score
        self.operate_on = operate_on
        self.l_strip = l_strip

    def evaluate(self, input_ids, next_token_ids,
                 target_txt, full_target_ids,
                 noisy_target_txt, full_noisy_target_ids,
                 num_beams=1, **kwargs):
        """

        Parameters
        ----------
        input_ids : (num_samples, curr_len)
        next_token_ids : (num_samples, num_considered_tokens)
        target_txt : (num_samples) target text per sample
        full_target_ids : (num_samples) target ids per sample
        noisy_target_txt : (num_samples) noisy target per sample in string format
        full_noisy_target_ids : (num_samples) noisy target ids per sample
        num_beams : the number of beams used by the decoding algorithm
        kwargs :

        Returns
        -------
        values : (num_samples, num_considered_tokens)
        """
        num_considered_tokens = next_token_ids.shape[-1]
        next_token_ids_f = torch.flatten(next_token_ids)[:, None]

        # Get input texts
        input_ids_r = torch.repeat_interleave(input_ids, num_considered_tokens, dim=0)
        extended_input_ids = torch.cat([input_ids_r, next_token_ids_f], dim=1)

        if self.operate_on == "text":
            # Get current and target text
            text = self.tree.tokenizer.batch_decode(extended_input_ids, skip_special_tokens=True)
            target_txts = np.repeat(target_txt, num_considered_tokens * num_beams)
            noisy_target_txts = np.repeat(noisy_target_txt, num_considered_tokens * num_beams)

            oracle_values = np.array(
                [self.bleu_score.compute([h], [[r[:len(h)]]], return_score_only=True) if h != "" else 0
                 for h, r in zip(text, target_txts)])
            noisy_oracle_values = np.array(
                [self.bleu_score.compute([h], [[r[:len(h)]]], return_score_only=True) if h != "" else 0
                 for h, r in zip(text, noisy_target_txts)])
        elif self.operate_on == "ids":
            target_ids_r = torch.repeat_interleave(full_target_ids, num_considered_tokens * num_beams, dim=0)
            noisy_ids_r = torch.repeat_interleave(full_noisy_target_ids, num_considered_tokens * num_beams, dim=0)

            oracle_values = []
            noisy_oracle_values = []
            for h, r, nr in zip(extended_input_ids, target_ids_r, noisy_ids_r):
                h = h[self.l_strip:].tolist()
                if len(h) == 0:
                    # Should fire in the first round if l_strip is 1
                    # however without consequences as in this round the lang_code token is generated
                    oracle_values.append(0)
                    noisy_oracle_values.append(0)
                    continue
                r = r[r != 1][self.l_strip:][:len(h)].tolist()  # Remove padding and get only the first len(h) tokens
                nr = nr[nr != 1][self.l_strip:][:len(h)].tolist()

                oracle_values.append(self.bleu_score.compute([" ".join(map(str, h))],
                                                             [[" ".join(map(str, r))]],
                                                             return_score_only=True))
                noisy_oracle_values.append(self.bleu_score.compute([" ".join(map(str, h))],
                                                                   [[" ".join(map(str, nr))]],
                                                                   return_score_only=True))
        else:
            raise ValueError("Unknown type to operate on: {}".format(self.operate_on))

        values = np.array(oracle_values) * self.noising_function_parameters["lambda"] + \
                 np.array(noisy_oracle_values) * (1 - self.noising_function_parameters["lambda"])
        values = torch.tensor(values).view_as(next_token_ids)

        return values.to(input_ids.device) / 100
