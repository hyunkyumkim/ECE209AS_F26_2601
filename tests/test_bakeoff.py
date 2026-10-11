import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from bakeoff import (
    INTERACTION_1,
    INTERACTION_2,
    NEUTRAL,
    TRIAL_NEGATIVE_1,
    TRIAL_NEGATIVE_2,
    TRIAL_POSITIVE_1,
    TRIAL_POSITIVE_2,
    PredictionSmoother,
    evaluate_trial,
)
from models import predict, preprocess_data, train_model


class SmootherTest(unittest.TestCase):
    def test_single_stray_frame_is_suppressed(self):
        s = PredictionSmoother()
        out = [s.update(l) for l in [NEUTRAL, NEUTRAL, INTERACTION_2, NEUTRAL, NEUTRAL, NEUTRAL]]
        self.assertEqual(set(out), {NEUTRAL})

    def test_sustained_event_passes(self):
        s = PredictionSmoother()
        out = [s.update(l) for l in [NEUTRAL] + [INTERACTION_1] * 5 + [NEUTRAL] * 6]
        self.assertIn(INTERACTION_1, out)
        self.assertEqual(out[-1], NEUTRAL)

    def test_reset_clears_history(self):
        s = PredictionSmoother()
        for _ in range(6):
            s.update(INTERACTION_1)
        s.reset()
        self.assertEqual(s.update(NEUTRAL), NEUTRAL)


class EvaluateTrialTest(unittest.TestCase):
    def test_positive_trials(self):
        self.assertTrue(evaluate_trial(TRIAL_POSITIVE_1, {NEUTRAL, INTERACTION_1}))
        self.assertTrue(evaluate_trial(TRIAL_POSITIVE_1, {INTERACTION_1}))
        self.assertFalse(evaluate_trial(TRIAL_POSITIVE_1, {NEUTRAL}))
        self.assertFalse(evaluate_trial(TRIAL_POSITIVE_1, {NEUTRAL, INTERACTION_1, INTERACTION_2}))
        self.assertTrue(evaluate_trial(TRIAL_POSITIVE_2, {NEUTRAL, INTERACTION_2}))
        self.assertFalse(evaluate_trial(TRIAL_POSITIVE_2, {NEUTRAL, INTERACTION_1, INTERACTION_2}))

    def test_negative_trials(self):
        self.assertTrue(evaluate_trial(TRIAL_NEGATIVE_1, {NEUTRAL, INTERACTION_2}))
        self.assertFalse(evaluate_trial(TRIAL_NEGATIVE_1, {NEUTRAL, INTERACTION_1}))
        self.assertTrue(evaluate_trial(TRIAL_NEGATIVE_2, {NEUTRAL}))
        self.assertFalse(evaluate_trial(TRIAL_NEGATIVE_2, {INTERACTION_2}))

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            evaluate_trial("nonsense", set())


class PreprocessTest(unittest.TestCase):
    def test_shape_and_finite(self):
        X = np.random.rand(7, 512) * 1e5
        out = preprocess_data(X)
        self.assertEqual(out.shape, (7, 513))
        self.assertTrue(np.isfinite(out).all())

    def test_all_zero_frame_is_finite(self):
        out = preprocess_data(np.zeros((1, 512)))
        self.assertTrue(np.isfinite(out).all())

    def test_single_vector_input(self):
        self.assertEqual(preprocess_data(np.random.rand(512)).shape, (1, 513))

    def test_hit_strength_does_not_change_spectrum_shape(self):
        x = np.random.rand(1, 512) * 1e4
        a, b = preprocess_data(x), preprocess_data(x * 3)
        # Loudness feature differs, spectrum shape stays close
        self.assertLess(np.abs(a[:, :-1] - b[:, :-1]).max(), 0.05)
        self.assertGreater(b[0, -1], a[0, -1])


class SyntheticClassifierTest(unittest.TestCase):
    """Sanity check on fake spectra: low thud vs high broadband vs quiet noise."""

    def _make(self, kind, n, rng):
        x = np.linspace(0, 1, 512)
        if kind == 0:  # neutral: quiet noise
            return rng.random((n, 512)) * 200
        if kind == 1:  # knock: loud, low-frequency heavy
            return (np.exp(-x * 6) * 6e4)[None, :] * rng.uniform(0.5, 1.5, (n, 1)) + rng.random((n, 512)) * 200
        # scratch: medium, broadband high
        return (x**1.5 * 2e4)[None, :] * rng.uniform(0.5, 1.5, (n, 1)) + rng.random((n, 512)) * 200

    def test_separable_classes(self):
        rng = np.random.default_rng(0)
        X = np.vstack([self._make(k, 150, rng) for k in range(3)])
        y = np.repeat([0, 1, 2], 150)
        model, names = train_model(preprocess_data(X), y, ["Neutral", "Knock", "Scratch"])
        Xt = np.vstack([self._make(k, 50, rng) for k in range(3)])
        yt = np.repeat([0, 1, 2], 50)
        acc = (predict(model, preprocess_data(Xt)) == yt).mean()
        self.assertGreater(acc, 0.95)


if __name__ == "__main__":
    unittest.main()
