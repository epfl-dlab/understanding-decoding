from typing import Optional


class SampleInformation:
    pred_score: float
    pred_utility: float
    tgt_score: Optional[float]

    def __init__(self, pred_score, pred_utility=None, tgt_score=None, pred_correlation=None):
        self.pred_score = pred_score
        self.pred_utility = pred_utility
        self.tgt_score = tgt_score
        self.pred_correlation = pred_correlation

    def set_correlation(self, pred_correlation):
        self.pred_correlation = pred_correlation

    def get_score(self, difference, ratio):
        """
        Returns the (normalized or unnormalized) score for the sample.

        Parameters
        ----------
        difference : "normalize" by subtracting the target_score
        ratio : "normalize" by reporting the ratio w.r.t. to the target_score

        Returns
        -------
        score : float
        is_normalized : bool
        """
        if self.tgt_score is None:
            # Normalization cannot be applied without a tgt_score
            difference = False
            ratio = False

        assert not (difference and ratio), "Difference and ratio cannot be simultaneously selected!"

        if difference:
            return self.pred_score - self.tgt_score, difference or ratio

        if ratio:
            return self.pred_score / self.tgt_score, difference or ratio

        return self.pred_score, difference or ratio

    def get_summary(self, difference=True, ratio=False):
        """
        Returns a summary for the sample that will be used in the visualization.

        Parameters
        ----------
        difference : See get_score.
        ratio : See get_score.

        Returns
        -------

        """
        summary = {"score": self.get_score(difference, ratio), "utility": self.pred_utility}
        if self.pred_correlation is not None:
            summary["correlation"] = self.pred_correlation

        return summary
