import torch

from detoxify import Detoxify

from . import BatchedEvaluationModel
from ..trees import Tree


class OracleDetoxify(BatchedEvaluationModel):
    def __init__(self, tree: Tree, model_type: str, batch_size: int, device="cpu"):
        """
        Evaluation model for predicting whether a textual sequence (or a list of textual sequences) is toxic.
        Can initialize 5 different model types from model type:
        - original:
            model trained on data from the Jigsaw Toxic Comment
            Classification Challenge
        - unbiased:
            model trained on data from the Jigsaw Unintended Bias in
            Toxicity Classification Challenge
        - multilingual:
            model trained on data from the Jigsaw Multilingual
            Toxic Comment Classification Challenge
        - original-small:
            lightweight version of the original model
        - unbiased-small:
            lightweight version of the unbiased model
        Parameters
        ----------
        tree: instance of `Tree`
        model_type : the name of the model (see above)
        batch_size : number of elements to run through the value function simultaneously
        device : a string tag (e.g. 'cpu', 'cuda', 'cuda:0', 'cuda:2, etc) or a torch.device object
        """
        super().__init__(tree)
        self.tree = tree
        self.device = device
        self.name = self.model_type = model_type
        self.model = Detoxify(model_type=model_type, device=device)
        self.batch_size = batch_size

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
        num_considered_tokens = next_token_ids.shape[-1]
        next_token_ids_f = torch.flatten(next_token_ids)[:, None]

        input_ids_r = torch.repeat_interleave(input_ids, num_considered_tokens, dim=0)
        extended_input_ids = torch.cat([input_ids_r, next_token_ids_f], dim=1)

        text = self.tree.tokenizer.batch_decode(extended_input_ids, skip_special_tokens=True)

        # split sentences on batches
        batched_text = [text[i: i + self.batch_size] for i in range(0, len(text), self.batch_size)]
        toxicity_scores = []
        for batch in batched_text:
            toxicity_scores.extend(self.model.predict(batch)["toxicity"])

        values = 1 - torch.tensor(toxicity_scores).view_as(next_token_ids)

        return values.to(input_ids.device)

    def reset_device(self, device: torch.device):
        self.device = device
        self.model.device = device
        self.model.model.to(device)
