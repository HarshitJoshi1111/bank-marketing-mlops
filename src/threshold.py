"""Decision threshold: turn predicted probabilities into yes/no.

The threshold is chosen on out-of-fold predictions of the training set (every row is predicted by a model
that did not see it), never on the test set. A fixed 0.5 only works for a model trained with class
weights; for a model that predicts calibrated probabilities around the 12 % base rate it is far too high.
"""
import numpy as np
from sklearn.metrics import precision_recall_curve


def choose_threshold(y_true, scores, method: str = "max_f1", value: float = 0.5, min_recall: float = 0.5) -> float:
    """
    max_f1      threshold with the highest F1 (balance of precision and recall)
    min_recall  highest threshold that still finds at least `min_recall` of the positives
                (useful when missing a likely customer costs more than an extra call)
    fixed       `value` as given
    """
    if method == "fixed":
        return float(value)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]  # the last point has no threshold
    if method == "max_f1":
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision),
                       where=(precision + recall) > 0)
        return float(thresholds[int(np.argmax(f1))])
    if method == "min_recall":
        ok = np.flatnonzero(recall >= min_recall)
        return float(thresholds[ok[-1]]) if len(ok) else float(thresholds[0])
    raise ValueError(f"unknown threshold method {method!r}")
