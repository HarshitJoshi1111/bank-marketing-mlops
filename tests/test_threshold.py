import numpy as np
import pytest

from src.threshold import choose_threshold

Y = np.array([0, 0, 0, 0, 1, 0, 1, 1])
SCORES = np.array([0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.6, 0.8])


def test_max_f1():
    # threshold 0.35 -> predicts the last 4: precision 3/4, recall 3/3 -> F1 0.857 (the best here)
    assert choose_threshold(Y, SCORES, "max_f1") == 0.35


def test_min_recall_takes_highest_threshold_that_reaches_the_recall():
    assert choose_threshold(Y, SCORES, "min_recall", min_recall=0.6) == 0.6   # recall 2/3
    assert choose_threshold(Y, SCORES, "min_recall", min_recall=1.0) == 0.35  # recall 3/3


def test_fixed():
    assert choose_threshold(Y, SCORES, "fixed", value=0.42) == 0.42


def test_unknown_method():
    with pytest.raises(ValueError):
        choose_threshold(Y, SCORES, "median")
