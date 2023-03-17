import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

from . import BatchedEvaluationModel
from ..trees import Tree


class OracleSolubility(BatchedEvaluationModel):
    def __init__(self, tree: Tree, batch_size: int, device="cpu"):
        """
        Class for scoring a protein sequence (or a list of textual sequences) with its solubility.
        Parameters
        ----------
        tree: instance of `Tree`
        batch_size : number of elements to run through the value function simultaneously
        device : a string tag (e.g. 'cpu', 'cuda', 'cuda:0', 'cuda:2, etc) or a torch.device object
        """
        super().__init__(tree)
        self.tree = tree
        self.device = device
        self.model = TextClassificationPipeline(
                model=AutoModelForSequenceClassification.from_pretrained("Rostlab/prot_bert_bfd_membrane"),
                tokenizer=AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd_membrane"), device=self.device)
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
        batched_text = [text[i : i + self.batch_size] for i in range(0, len(text), self.batch_size)]
        solubility_scores = []
        for batch in batched_text:
            score = self.model(batch)[0]["score"] 
            solubility_scores.extend([score])

        values = 1 - torch.tensor(solubility_scores).view_as(next_token_ids)
        return values.to(input_ids.device)

    def reset_device(self, device: torch.device):
        self.device = device
        self.model.device = device
        self.model.model.to(device)
