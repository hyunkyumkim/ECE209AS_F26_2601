"""Bakeoff trial logic: prediction smoothing and pass/fail scoring.

Kept free of GUI and audio imports so it can be unit-tested.
"""

from collections import Counter, deque

NEUTRAL = "Neutral"
INTERACTION_1 = "Knock"
INTERACTION_2 = "Scratch"
DEFAULT_CLASSES = [NEUTRAL, INTERACTION_1, INTERACTION_2]

# A non-neutral label is only emitted when it fills at least MIN_VOTES of the
# last WINDOW frames. One stray frame cannot add a wrong class to the trial set.
SMOOTH_WINDOW = 6
SMOOTH_MIN_VOTES = 3


class PredictionSmoother:
    """Majority-style filter over per-frame labels, biased toward Neutral."""

    def __init__(self, window=SMOOTH_WINDOW, min_votes=SMOOTH_MIN_VOTES, neutral=NEUTRAL):
        self.window = window
        self.min_votes = min_votes
        self.neutral = neutral
        self._labels = deque(maxlen=window)

    def reset(self):
        self._labels.clear()

    def update(self, label):
        """Add one raw frame label and return the smoothed label."""
        self._labels.append(label)
        counts = Counter(l for l in self._labels if l != self.neutral)
        if counts:
            best, votes = counts.most_common(1)[0]
            if votes >= self.min_votes:
                return best
        return self.neutral


# Trial kinds: what the participant is asked to do.
TRIAL_POSITIVE_1 = "Positive: Knock"
TRIAL_POSITIVE_2 = "Positive: Scratch"
TRIAL_NEGATIVE_1 = "Negative: not Knock"
TRIAL_NEGATIVE_2 = "Negative: not Scratch"
TRIAL_KINDS = [TRIAL_POSITIVE_1, TRIAL_POSITIVE_2, TRIAL_NEGATIVE_1, TRIAL_NEGATIVE_2]


def evaluate_trial(kind, results):
    """Return True if the set of recorded results passes the given trial kind.

    Positive trial of X: results must contain X and no other interaction
    (Neutral is allowed, since a trial starts and ends in Neutral).
    Negative trial of X: results must not contain X.
    """
    results = set(results)
    other_interactions = {INTERACTION_1, INTERACTION_2}
    if kind == TRIAL_POSITIVE_1:
        return INTERACTION_1 in results and not (results & (other_interactions - {INTERACTION_1}))
    if kind == TRIAL_POSITIVE_2:
        return INTERACTION_2 in results and not (results & (other_interactions - {INTERACTION_2}))
    if kind == TRIAL_NEGATIVE_1:
        return INTERACTION_1 not in results
    if kind == TRIAL_NEGATIVE_2:
        return INTERACTION_2 not in results
    raise ValueError(f"Unknown trial kind: {kind}")


def format_results(results):
    """Stable, readable output for the experimenter."""
    ordered = sorted(results, key=lambda r: (r != NEUTRAL, r))
    return ", ".join(ordered) if ordered else "(no results)"
