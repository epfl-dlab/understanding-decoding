from typing import Optional

from sacrebleu.metrics import BLEU


class BLEUScore:
    def __init__(
        self,
        lowercase: bool,
        max_ngram_order: int,
        tokenize: str,
        smooth_method: str,
        smooth_value: Optional[float] = None,
    ):
        """

        Parameters
        ----------
        lowercase : If True, lowercased BLEU is computed.
        max_ngram_order : Maximum n-gram order to use when computing BLEU score
        tokenize : The tokenizer to use. If None, defaults to language-specific tokenizers with '13a' as the fallback default. See [here](https://github.com/mjpost/sacrebleu#bleu) the list of available tokenizers.
        smooth_method : The smoothing method to use ('floor', 'add-k', 'exp' or 'none').
        smooth_value : The smoothing value for `floor` and `add-k` methods. `None` falls back to default value.
        """
        self.metric = BLEU(
            lowercase=lowercase,
            max_ngram_order=max_ngram_order,
            tokenize=tokenize,
            smooth_method=smooth_method,
            smooth_value=smooth_value,
        )

        self.name = f"BLEU-{max_ngram_order}__tok-{tokenize}_sm-_{smooth_method}_sv_{smooth_value}"

    def compute(self, hypotheses, reference_documents, return_score_only=True):
        """

        Parameters
        ----------
        hypotheses : list of N translations to score. Each translation should be string of text.
        reference_documents : list of D reference documents with a document being a sequence of N reference strings,
        one for each hypothesis.

        return_score_only : boolean flag specifying whether only the (float) score should be returned

        Returns
        -------
        : the score (float) or the metric's score object
        """
        for rd in reference_documents:
            assert len(hypotheses) == len(rd)

        bleu_score = self.metric.corpus_score(hypotheses=hypotheses, references=reference_documents)

        if return_score_only:
            return bleu_score.score

        return bleu_score

    def compute_from_dataset(self, dataset, return_score_only=True, per_datapoint=False, per_beam=False):
        assert sum([per_datapoint, per_beam]) <= 1
        
        top_pred_only = not per_beam
        hypotheses = [dataset.get_predictions(item, top_pred_only=top_pred_only) for item in dataset]
        references = [dataset.get_targets(item, wrap_in_list=True) for item in dataset]

        if per_datapoint:
            return [self.compute([h], list(zip(r)), return_score_only) for h, r in zip(hypotheses, references)]
        elif per_beam:
            return [
                [self.compute([beam], list(zip(r)), return_score_only) for beam in h]
                for h, r in zip(hypotheses, references)
            ]

        reference_documents = list(zip(*references))
        return self.compute(hypotheses, reference_documents, return_score_only)

    # def compute_from_dataset_resample_exp(self, dataset):
    #     hypotheses = [dataset.get_predictions(item, top_pred_only=False) for item in dataset]
    #     references = [dataset.get_targets(item, wrap_in_list=True) for item in dataset]
        
    #     bleu_score = [
    #         [self.compute([beam], list(zip(r)), return_score_only=True) for beam in h]
    #         for h, r in zip(hypotheses, references)
    #     ]

    #     return bleu_score

    # ToDo: Check how this is computed: I think that it might resample references and not datapoints
    # def compute_ci(self, hypotheses, references, n_bootstrap):
    #     """
    #     Returns the 95% confidence interval computed with bootstrap resampling
    #
    #     Parameters
    #     ----------
    #     hypotheses : list of translations to score. Each translation should be string of text.
    #     references : list of lists of references for each translation. Each reference should be a string of text.
    #     n_bootstrap : number of bootstrap samples to use in the calculation of the confidence interval
    #
    #     Returns
    #     -------
    #     mean : the confidence interval's mean
    #     error : the 95% confidence interval's error
    #     """
    #
    #     bleu_score = self.metric.corpus_score(
    #         hypotheses=hypotheses,
    #         references=references,
    #         n_bootstrap=n_bootstrap
    #     )
    #
    #     mean = bleu_score._mean
    #     error = bleu_score._ci
    #
    #     return mean, error
