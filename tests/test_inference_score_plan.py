"""Regression tests for StudentScorer.score_plan credit handling (Stage 0).

score_plan() previously crashed with AttributeError under the live B2
DifficultyState because it read ``self._diff.index`` — but ``self._diff`` is
None whenever the scorer is loaded from a DifficultyState (only
``self._difficulty_state`` is set then). These tests pin the fix: score_plan
runs under BOTH the B2 state and the legacy lookup DataFrame, and the legacy
path still sums real per-course credits into the plan workload.

The B2 DifficultyState carries no per-course course_credits, so under B2 each
unresolved course falls back to 3.0 credits — the same default
recommendation.py uses (``fillna(3.0)``) — rather than 0, which would be an
out-of-distribution semester_reg_credits value never seen in training.
"""

import unittest

import numpy as np
import pandas as pd

from src.concurrent_group_features import (
    MODEL_CONCURRENT_FEATURES,
    compute_concurrent_group_features,
)
from src.course_difficulty import empty_difficulty_state
from src.inference import StudentScorer
from src.model_training import (
    BASELINE_41_FEATURES,
    CONCURRENT_44_FEATURES,
    CONCURRENT_MODEL_FEATURES,
    MODEL_FEATURES,
)


class _RecordingModel:
    """Stand-in LightGBM booster that records the last feature matrix it saw."""

    def __init__(self):
        self.last_X = None

    def predict(self, X):
        self.last_X = X
        # A constant in [0, 1] keeps pass_prob probability-shaped for structural
        # assertions without needing a real trained booster.
        return np.full(len(X), 0.5)


def _history():
    # score() only needs a non-empty history with a course_id column when a
    # precomputed snapshot is supplied (so feature engineering is skipped).
    return pd.DataFrame({"course_id": ["PRIOR.111"]})


def _snapshot():
    # An empty snapshot is sufficient: score() reads every field through
    # Series.get(field, default), so missing keys fall back to their defaults.
    return pd.Series(dtype=object)


class ScorePlanCreditLookupTest(unittest.TestCase):
    DEGREE = "BSCS.111"
    PLAN = ["C1.111", "C2.111", "C3.111"]
    CREDITS = [3.0, 4.0, 3.0]

    def _legacy_scorer(self) -> StudentScorer:
        # Legacy pre-B2 lookup: a DataFrame keyed by degree_course_key with a
        # course_credits column (self._diff is a real index, not None).
        lookup = pd.DataFrame(
            {
                "degree_course_key": [f"{self.DEGREE}__{c}" for c in self.PLAN],
                "course_credits": self.CREDITS,
            }
        )
        return StudentScorer(_RecordingModel(), _RecordingModel(), lookup)

    def _b2_scorer(self) -> StudentScorer:
        # B2 path: a DifficultyState, so self._diff is None (the crash trigger).
        return StudentScorer(
            _RecordingModel(), _RecordingModel(), empty_difficulty_state()
        )

    def _run_plan(self, scorer: StudentScorer) -> pd.DataFrame:
        return scorer.score_plan(
            df_history=_history(),
            plan_course_ids=self.PLAN,
            target_part_id="20261",
            degree_id=self.DEGREE,
            snapshot=_snapshot(),
        )

    def test_legacy_sums_real_plan_credits(self):
        scorer = self._legacy_scorer()
        result = self._run_plan(scorer)

        self.assertEqual(len(result), len(self.PLAN))
        # semester_reg_credits is not an output column; read it back from the
        # feature matrix the model received. All plan rows share the plan total.
        seen = scorer.grade_model.last_X
        self.assertEqual(
            float(seen["semester_reg_credits"].iloc[0]), sum(self.CREDITS)
        )

    def test_b2_state_runs_without_attributeerror(self):
        scorer = self._b2_scorer()
        # Before the fix this raised: AttributeError: 'NoneType' object has no
        # attribute 'index'. It must now run to completion.
        result = self._run_plan(scorer)

        self.assertEqual(len(result), len(self.PLAN))
        # No per-course credit source in the B2 state -> each of the 3 plan
        # courses falls back to 3.0 credits (recommendation.py's own default),
        # giving 9.0 total. Never a crash, never a fabricated 0.0 workload.
        seen = scorer.grade_model.last_X
        self.assertEqual(
            float(seen["semester_reg_credits"].iloc[0]), 3.0 * len(self.PLAN)
        )


class ScorePlanConcurrentParityTest(unittest.TestCase):
    """Train/serve parity for the concurrent peer features (item 19)."""

    DEGREE = "BSCS.111"
    PLAN = ["C1.111", "C2.111", "C3.111"]
    PASS_RATE = [0.9, 0.5, np.nan]  # C3 is a Level-6/NaN-pass_rate candidate
    DIFF_MISSING = [0, 0, 1]
    REQ_TYPE = [1, 1, 2]

    def _legacy_scorer(self) -> StudentScorer:
        lookup = pd.DataFrame(
            {
                "degree_course_key": [f"{self.DEGREE}__{c}" for c in self.PLAN],
                "course_credits": [3.0, 3.0, 3.0],
                "course_pass_rate_historical": self.PASS_RATE,
                "course_difficulty_missing": self.DIFF_MISSING,
                "requirement_type_id": self.REQ_TYPE,
            }
        )
        return StudentScorer(_RecordingModel(), _RecordingModel(), lookup)

    def _hand_frame(self) -> pd.DataFrame:
        # One semester group with the same difficulty inputs score_plan sees.
        n = len(self.PLAN)
        return pd.DataFrame(
            {
                "university_id": ["u"] * n,
                "student_id": ["s"] * n,
                "degree_id": ["d"] * n,
                "part_id": ["p"] * n,
                # Peer membership is course-grained, so the hand frame carries
                # the same course identities score_plan stamps onto its plan.
                "course_id": list(self.PLAN),
                "course_pass_rate_historical": self.PASS_RATE,
                "course_difficulty_missing": self.DIFF_MISSING,
                "requirement_type_id": self.REQ_TYPE,
            }
        )

    def test_score_plan_matches_direct_feature_computation(self):
        scorer = self._legacy_scorer()
        scorer.score_plan(
            df_history=_history(),
            plan_course_ids=self.PLAN,
            target_part_id="20261",
            degree_id=self.DEGREE,
            snapshot=_snapshot(),
        )
        # The feature matrix (captured before the output sort) is in plan order.
        seen = scorer.grade_model.last_X
        expected = compute_concurrent_group_features(self._hand_frame())

        np.testing.assert_allclose(
            seen["concurrent_peer_difficulty_mean"].to_numpy(dtype=float),
            expected["concurrent_peer_difficulty_mean"].to_numpy(dtype=float),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            seen["concurrent_peer_difficulty_max"].to_numpy(dtype=float),
            expected["concurrent_peer_difficulty_max"].to_numpy(dtype=float),
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            seen["concurrent_peer_difficulty_missing"].to_numpy(),
            expected["concurrent_peer_difficulty_missing"].to_numpy(),
        )
        # Sanity: the hand-computed values are the expected leave-one-out ones.
        self.assertEqual(
            list(expected["concurrent_peer_difficulty_mean"].round(4)), [0.5, 0.1, 0.3]
        )


class ScorerFeatureContractTest(unittest.TestCase):
    """A scorer must build the matrix its contract defines, never assume 44."""

    DEGREE = "BSCS.111"
    CANDIDATES = ["C1.111", "C2.111"]

    def _scorer(self, contract):
        return StudentScorer(
            _RecordingModel(),
            _RecordingModel(),
            empty_difficulty_state(),
            feature_contract=contract,
        )

    def _score(self, scorer):
        scorer.score(
            df_history=_history(),
            candidate_course_ids=self.CANDIDATES,
            target_part_id="20261",
            degree_id=self.DEGREE,
            snapshot=_snapshot(),
        )
        return scorer.grade_model.last_X

    def test_baseline_41_contract_builds_a_41_column_matrix(self):
        seen = self._score(self._scorer("baseline_41"))
        self.assertEqual(seen.shape[1], 41)
        self.assertEqual(list(seen.columns), BASELINE_41_FEATURES)
        for column in CONCURRENT_MODEL_FEATURES:
            self.assertNotIn(column, seen.columns)

    def test_concurrent_44_contract_builds_a_44_column_matrix(self):
        seen = self._score(self._scorer("concurrent_44"))
        self.assertEqual(seen.shape[1], 44)
        self.assertEqual(list(seen.columns), CONCURRENT_44_FEATURES)
        for column in CONCURRENT_MODEL_FEATURES:
            self.assertIn(column, seen.columns)

    def test_default_contract_is_unchanged_for_existing_callers(self):
        scorer = StudentScorer(
            _RecordingModel(), _RecordingModel(), empty_difficulty_state()
        )
        self.assertEqual(scorer.feature_contract.name, "concurrent_44")

    def test_serving_limitation_is_recorded_on_the_scorer(self):
        note = StudentScorer.SERVING_LIMITATION
        self.assertIn("baseline_41", note)
        self.assertIn("score_plan", note)

    def test_score_plan_respects_a_41_contract(self):
        scorer = self._scorer("baseline_41")
        scorer.score_plan(
            df_history=_history(),
            plan_course_ids=self.CANDIDATES,
            target_part_id="20261",
            degree_id=self.DEGREE,
            snapshot=_snapshot(),
        )
        self.assertEqual(scorer.grade_model.last_X.shape[1], 41)


class StructuralScoreTest(unittest.TestCase):
    """Refactor guard for score() independent of the parity test (item 20)."""

    DEGREE = "BSCS.111"
    CANDIDATES = ["C1.111", "C2.111", "C3.111"]

    def test_score_builds_full_placeholder_matrix(self):
        scorer = StudentScorer(
            _RecordingModel(), _RecordingModel(), empty_difficulty_state()
        )
        result = scorer.score(
            df_history=_history(),
            candidate_course_ids=self.CANDIDATES,
            target_part_id="20261",
            degree_id=self.DEGREE,
            snapshot=_snapshot(),
        )
        seen = scorer.grade_model.last_X

        # (a) X is exactly MODEL_FEATURES, width 44.
        self.assertEqual(list(seen.columns), MODEL_FEATURES)
        self.assertEqual(seen.shape[1], 44)
        # (b) pre-plan concurrent placeholder for every row.
        self.assertTrue(seen["concurrent_peer_difficulty_mean"].isna().all())
        self.assertTrue(seen["concurrent_peer_difficulty_max"].isna().all())
        self.assertTrue((seen["concurrent_peer_difficulty_missing"] == 1).all())
        for col in MODEL_CONCURRENT_FEATURES:
            self.assertIn(col, seen.columns)
        # (c) requirement_size_bucket_ord non-null for all rows.
        self.assertTrue(seen["requirement_size_bucket_ord"].notna().all())
        # (d) pass_prob in [0, 1].
        self.assertTrue(result["pass_prob"].between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()
