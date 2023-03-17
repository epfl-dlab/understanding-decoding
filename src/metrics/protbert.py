from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import re
from typing import List, Union

import numpy as np

class SolubleScore:
    def __init__(self, name, device=0):
        """
        Class for scoring a protein sequence (or a list of textual sequences) with its solubility.
        Parameters
        ----------
        device : number of cuda device integer (e.g. '0', '1', '2' etc)
        """
        self.device = device 
        self.name = name
        self.model = TextClassificationPipeline(
                model=AutoModelForSequenceClassification.from_pretrained("Rostlab/prot_bert_bfd_membrane"),
                tokenizer=AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd_membrane"), device=self.device)

        # used as cache
        self.scores_per_datapoint = None

    def compute(self, text: Union[str, List[str]]) -> List[float]:
        """
        Parameters
        ----------
        text : a string or a list of strings to score

        Returns
        -------
        scores : a list of toxicity scores for each each of the textual sequences
        """
        if isinstance(text, str):
            text = [text]
        
        scores = [] 
        for i in range(len(text)):
            #Create or load sequences and map rarely occured amino acids
            sequences = ' '.join([re.sub(r"[UZOB]", "X", sequence) for sequence in text[i]])
            score = self.model(sequences)
            soluble_score = score[0]['score']
            scores.append(soluble_score)
        return scores

    def compute_from_dataset(self, dataset, use_if_cached=False, per_datapoint=False, per_beam=False):
        assert sum([per_datapoint, per_beam]) <= 1

        if per_beam:
            hypotheses = [dataset.get_predictions(item, top_pred_only=False) for item in dataset]
            return [self.compute(hyp) for hyp in hypotheses]

        if not (use_if_cached and self.scores_per_datapoint):
            hypotheses = [dataset.get_predictions(item, top_pred_only=True) for item in dataset]
            self.scores_per_datapoint = self.compute(hypotheses)

        if per_datapoint:
            return self.scores_per_datapoint

        return np.mean(self.scores_per_datapoint)
