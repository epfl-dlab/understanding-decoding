import torch
from transformers import GPT2TokenizerFast

from detoxify import Detoxify
from . import BatchedEvaluationModel
from .utils import add_noise_continuous
from ..trees import Tree


class NoisyOracleDetoxify(BatchedEvaluationModel):
    def __init__(
        self, tree: Tree, model_type: str, batch_size: int, huggingface_config_path: str, checkpoint: str, device="cpu"
    ):
        """
        Loads a specific checkpoint of the Detoxify model that is undertrained.

        See the OracleDetoxify docstring for details on the parameters.
        Parameters
        ----------
        checkpoint_path: (str) path to the checkpoint file to be loaded
        """
        super().__init__(tree)
        self.tree = tree

        self.device = device
        self.name = self.model_type = model_type
        self.model = Detoxify(
            model_type=model_type, device=device, huggingface_config_path=huggingface_config_path, checkpoint=checkpoint
        )
        self.batch_size = batch_size
        self.tokenizer = None

    def apply_noise(self, values):
        return torch.cat(
            [
                add_noise_continuous(v, np_random_state=self.np_random_state, **self.noising_function_parameters)[
                    None, :
                ]
                for v in values
            ]
        )

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
        if self.tokenizer is None:
            self.tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_model_name_or_path="gpt2", padding_side="left",
                                              pad_token="<|endoftext|>")
        num_considered_tokens = next_token_ids.shape[-1]
        next_token_ids_f = torch.flatten(next_token_ids)[:, None]

        input_ids_r = torch.repeat_interleave(input_ids, num_considered_tokens, dim=0)
        extended_input_ids = torch.cat([input_ids_r, next_token_ids_f], dim=1)

        text = self.tokenizer.batch_decode(extended_input_ids, skip_special_tokens=True)

        # assert self.tokenizer.padding_side == "left"
        # assert self.tokenizer.eos_token_id == self.tokenizer.pad_token_id
        # assert self.tokenizer.eos_token_id == self.tree.lm_model.config.eos_token_id
        # assert self.tokenizer.pad_token_id == self.tree.lm_model.config.pad_token_id
        # assert self.tokenizer.batch_decode(extended_input_ids, skip_special_tokens=True) == text

        # split sentences on batches
        batched_text = [text[i : i + self.batch_size] for i in range(0, len(text), self.batch_size)]
        toxicity_scores = []
        for batch in batched_text:
            toxicity_scores.extend(self.model.predict(batch)["toxicity"])

        values = 1 - torch.tensor(toxicity_scores).view_as(next_token_ids)

        return values.to(input_ids.device)

    def reset_device(self, device: torch.device):
        self.device = device
        self.model.device = device
        self.model.model.to(device)
