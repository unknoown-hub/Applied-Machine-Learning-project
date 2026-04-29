"""
Walk-forward evaluation — exactly matching Z&T's setup.

From the paper (Section 4):
  "we repeatedly split the data set into a training sample with a length
   of 291 weeks and a single one-step ahead hold-out week.  This exercise
   is repeated 52 times, shifting the time window by one week each time."

So: fixed rolling window of 291 weeks, 52 holdout steps.
  Step 1 : train weeks  1–291, test week 292
  Step 2 : train weeks  2–292, test week 293
  ...
  Step 52: train weeks 52–342, test week 343
"""
from __future__ import annotations
import numpy as np


TRAIN_LEN  = 291   # fixed training window size (weeks)
N_HOLDOUT  = 52    # number of one-step-ahead holdout steps
FIRST_TEST = 292   # first holdout week (= TRAIN_LEN + 1)


def rolling_week_splits(n_total: int = 343) -> list:
    """
    Returns a list of (train_weeks, test_week) tuples.

    train_weeks — set of week_num values in the training window
    test_week   — single week_num value to predict
    """
    splits = []
    for step in range(N_HOLDOUT):
        test_week  = FIRST_TEST + step
        train_start = test_week - TRAIN_LEN
        train_weeks = set(range(train_start, test_week))
        splits.append((train_weeks, test_week))
    return splits
