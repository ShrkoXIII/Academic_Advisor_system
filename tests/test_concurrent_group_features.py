"""Tests for the Concurrent Course Group feature family (Experiment 1).

Covers the binding NaN-robust leave-one-out contract documented in
``src.concurrent_group_features``: exact mean/max, singleton handling, peer
exclusion, student/degree isolation, weak-peer inclusion, order/index
invariance, audit-column semantics, and leakage independence.
"""

import unittest

import numpy as np
import pandas as pd

from src.concurrent_group_features import (
    AUDIT_CONCURRENT_FEATURES,
    add_concurrent_group_features,
    compute_concurrent_group_features,
)


def _row(
    *,
    uni="1",
    student="s",
    degree="d",
    part="20241",
    course="c",
    pass_rate=np.nan,
    diff_missing=0,
    req_type=1,
):
    return {
        "university_id": uni,
        "student_id": student,
        "degree_id": degree,
        "part_id": part,
        "course_id": course,
        "course_pass_rate_historical": pass_rate,
        "course_difficulty_missing": diff_missing,
        "requirement_type_id": req_type,
    }


def _df(rows, index=None):
    return pd.DataFrame(rows, index=index)


def _feat(rows, index=None):
    return compute_concurrent_group_features(_df(rows, index=index))


class ConcurrentGroupFeatureTest(unittest.TestCase):
    # --- 1. hand-worked 3-course group: exact mean & max of (1 - pass_rate) ---
    def test_01_hand_worked_three_course_group(self):
        # d = {A:0.1, B:0.5, C:0.3}
        f = _feat(
            [
                _row(course="A", pass_rate=0.9),
                _row(course="B", pass_rate=0.5),
                _row(course="C", pass_rate=0.7),
            ]
        )
        mean = f["concurrent_peer_difficulty_mean"].to_numpy()
        mx = f["concurrent_peer_difficulty_max"].to_numpy()
        self.assertAlmostEqual(mean[0], (0.5 + 0.3) / 2)  # A peers B,C
        self.assertAlmostEqual(mean[1], (0.1 + 0.3) / 2)  # B peers A,C
        self.assertAlmostEqual(mean[2], (0.1 + 0.5) / 2)  # C peers A,B
        self.assertAlmostEqual(mx[0], 0.5)
        self.assertAlmostEqual(mx[1], 0.3)
        self.assertAlmostEqual(mx[2], 0.5)
        self.assertTrue((f["concurrent_peer_difficulty_missing"] == 0).all())
        self.assertTrue((f["concurrent_peer_observed_count"] == 2).all())

    # --- 2. singleton group -> NaN + missing=1 ---
    def test_02_singleton_group(self):
        f = _feat([_row(course="only", pass_rate=0.8)])
        self.assertEqual(f["concurrent_peer_difficulty_missing"].iloc[0], 1)
        self.assertTrue(np.isnan(f["concurrent_peer_difficulty_mean"].iloc[0]))
        self.assertTrue(np.isnan(f["concurrent_peer_difficulty_max"].iloc[0]))
        self.assertEqual(f["concurrent_peer_observed_count"].iloc[0], 0)

    # --- 3. current row excluded from its own peer set ---
    def test_03_excludes_self(self):
        # Target A has an extreme own d that must NOT appear in its own stats.
        f = _feat(
            [
                _row(course="A", pass_rate=0.0),  # d=1.0 (extreme, own)
                _row(course="B", pass_rate=0.8),  # d=0.2
                _row(course="C", pass_rate=0.9),  # d=0.1
            ]
        )
        self.assertAlmostEqual(
            f["concurrent_peer_difficulty_mean"].iloc[0], (0.2 + 0.1) / 2
        )
        self.assertAlmostEqual(f["concurrent_peer_difficulty_max"].iloc[0], 0.2)

    # --- 4. isolation by student AND by degree ---
    def test_04_isolation_by_student_and_degree(self):
        f = _feat(
            [
                _row(student="s1", course="A", pass_rate=0.5),
                _row(student="s2", course="B", pass_rate=0.5),  # diff student
                _row(degree="d2", course="C", pass_rate=0.5),  # diff degree (student s1)
            ]
        )
        # None share a semester group -> all singletons.
        self.assertTrue((f["concurrent_peer_difficulty_missing"] == 1).all())
        self.assertTrue((f["concurrent_peer_observed_count"] == 0).all())

    # --- 5. weak-fallback peers ARE included in mean/max ---
    def test_05_weak_peer_included(self):
        f = _feat(
            [
                _row(course="A", pass_rate=0.9),  # d=0.1
                _row(course="B", pass_rate=0.4, diff_missing=1),  # d=0.6, weak but valid
            ]
        )
        self.assertAlmostEqual(f["concurrent_peer_difficulty_mean"].iloc[0], 0.6)
        self.assertAlmostEqual(f["concurrent_peer_difficulty_max"].iloc[0], 0.6)
        self.assertEqual(f["concurrent_peer_difficulty_missing"].iloc[0], 0)

    # --- 6. row count, index, and order unchanged ---
    def test_06_index_and_order_preserved(self):
        idx = [10, 3, 7]
        df = _df(
            [
                _row(course="A", pass_rate=0.9),
                _row(course="B", pass_rate=0.5),
                _row(course="C", pass_rate=0.7),
            ],
            index=idx,
        )
        out = add_concurrent_group_features(df)
        self.assertEqual(len(out), len(df))
        self.assertEqual(list(out.index), idx)
        self.assertEqual(list(out["course_id"]), ["A", "B", "C"])

    # --- 7. changing target's OWN difficulty does not change its own values ---
    def test_07_own_difficulty_change_is_inert(self):
        base = [
            _row(course="A", pass_rate=0.9),
            _row(course="B", pass_rate=0.5),
            _row(course="C", pass_rate=0.7),
        ]
        f1 = _feat(base)
        changed = [dict(base[0], course_pass_rate_historical=0.1)] + base[1:]
        f2 = _feat(changed)
        self.assertAlmostEqual(
            f1["concurrent_peer_difficulty_mean"].iloc[0],
            f2["concurrent_peer_difficulty_mean"].iloc[0],
        )
        self.assertAlmostEqual(
            f1["concurrent_peer_difficulty_max"].iloc[0],
            f2["concurrent_peer_difficulty_max"].iloc[0],
        )

    # --- 8. changing a PEER's difficulty DOES change the target's values ---
    def test_08_peer_difficulty_change_propagates(self):
        base = [
            _row(course="A", pass_rate=0.9),
            _row(course="B", pass_rate=0.5),
            _row(course="C", pass_rate=0.7),
        ]
        f1 = _feat(base)
        changed = [base[0], dict(base[1], course_pass_rate_historical=0.2)] + base[2:]
        f2 = _feat(changed)
        self.assertNotAlmostEqual(
            f1["concurrent_peer_difficulty_mean"].iloc[0],
            f2["concurrent_peer_difficulty_mean"].iloc[0],
        )

    # --- 9. values unaffected by current-semester outcome columns ---
    def test_09_outcome_columns_do_not_leak(self):
        rows = [
            _row(course="A", pass_rate=0.9),
            _row(course="B", pass_rate=0.5),
        ]
        f_clean = compute_concurrent_group_features(_df(rows))
        df_noisy = _df(rows)
        df_noisy["final_mark"] = [10, 90]
        df_noisy["gpa_points"] = [0.0, 4.0]
        df_noisy["semester_pass_credits"] = [0, 12]
        f_noisy = compute_concurrent_group_features(df_noisy)
        pd.testing.assert_frame_equal(f_clean, f_noisy)

    # --- 10. stable under row-order shuffle ---
    def test_10_order_independent(self):
        rows = [
            _row(course="A", pass_rate=0.9),
            _row(course="B", pass_rate=0.5),
            _row(course="C", pass_rate=0.7),
        ]
        f_ordered = compute_concurrent_group_features(_df(rows))
        shuffled = _df([rows[2], rows[0], rows[1]], index=[2, 0, 1])
        f_shuffled = compute_concurrent_group_features(shuffled).sort_index()
        pd.testing.assert_frame_equal(f_ordered, f_shuffled)

    # --- 11. audit columns are NOT in MODEL_FEATURES ---
    def test_11_audit_columns_absent_from_model_features(self):
        from src.model_training import MODEL_FEATURES

        for col in AUDIT_CONCURRENT_FEATURES:
            self.assertNotIn(col, MODEL_FEATURES)

    # --- 12. two-course group: each mean == other's single peer d; max == mean ---
    def test_12_two_course_group(self):
        f = _feat(
            [
                _row(course="A", pass_rate=0.9),  # d=0.1
                _row(course="B", pass_rate=0.4),  # d=0.6
            ]
        )
        self.assertAlmostEqual(f["concurrent_peer_difficulty_mean"].iloc[0], 0.6)
        self.assertAlmostEqual(f["concurrent_peer_difficulty_max"].iloc[0], 0.6)
        self.assertAlmostEqual(f["concurrent_peer_difficulty_mean"].iloc[1], 0.1)
        self.assertAlmostEqual(f["concurrent_peer_difficulty_max"].iloc[1], 0.1)

    # --- 13. LOO max: unique max -> second-highest; ties keep the max ---
    def test_13_loo_max_unique_and_ties(self):
        # d = {A:0.9 unique max, B:0.5, C:0.3}
        f_unique = _feat(
            [
                _row(course="A", pass_rate=0.1),
                _row(course="B", pass_rate=0.5),
                _row(course="C", pass_rate=0.7),
            ]
        )
        self.assertAlmostEqual(f_unique["concurrent_peer_difficulty_max"].iloc[0], 0.5)
        self.assertAlmostEqual(f_unique["concurrent_peer_difficulty_max"].iloc[1], 0.9)

        # Tie for the max: A and B both d=0.9 -> A's peer max stays 0.9.
        f_tie = _feat(
            [
                _row(course="A", pass_rate=0.1),
                _row(course="B", pass_rate=0.1),
                _row(course="C", pass_rate=0.7),
            ]
        )
        self.assertAlmostEqual(f_tie["concurrent_peer_difficulty_max"].iloc[0], 0.9)

    # --- 14. peers exist but zero valid peer values -> mean/max NaN, missing 0 ---
    def test_14_peers_but_no_valid_values(self):
        f = _feat(
            [
                _row(course="A", pass_rate=np.nan),
                _row(course="B", pass_rate=np.nan),
                _row(course="C", pass_rate=np.nan),
            ]
        )
        self.assertTrue(f["concurrent_peer_difficulty_mean"].isna().all())
        self.assertTrue(f["concurrent_peer_difficulty_max"].isna().all())
        self.assertTrue((f["concurrent_peer_difficulty_missing"] == 0).all())
        self.assertTrue((f["concurrent_peer_observed_count"] == 2).all())

    # --- 15. same_req_type_ratio: NaN target -> NaN; missing peer = non-match ---
    def test_15_same_req_type_ratio(self):
        f = _feat(
            [
                _row(course="A", pass_rate=0.9, req_type=1),  # target
                _row(course="B", pass_rate=0.9, req_type=1),  # match
                _row(course="C", pass_rate=0.9, req_type=2),  # non-match
                _row(course="D", pass_rate=0.9, req_type=np.nan),  # missing -> non-match
                _row(course="E", pass_rate=0.9, req_type=np.nan),  # NaN target
            ]
        )
        r = f["concurrent_peer_same_req_type_ratio"].to_numpy()
        # A: 4 peers, one shares req_type=1 -> 1/4
        self.assertAlmostEqual(r[0], 1.0 / 4.0)
        # E has NaN req_type -> NaN result
        self.assertTrue(np.isnan(r[4]))

    # --- 16. weak_ratio: 0 when no peers; exact fraction otherwise ---
    def test_16_weak_ratio(self):
        singleton = _feat([_row(course="only", pass_rate=0.8, diff_missing=1)])
        self.assertEqual(singleton["concurrent_peer_weak_ratio"].iloc[0], 0.0)

        f = _feat(
            [
                _row(course="A", pass_rate=0.9, diff_missing=0),  # target
                _row(course="B", pass_rate=0.9, diff_missing=1),  # weak peer
                _row(course="C", pass_rate=0.9, diff_missing=0),  # non-weak peer
            ]
        )
        self.assertAlmostEqual(f["concurrent_peer_weak_ratio"].iloc[0], 0.5)

    # --- 17. duplicate course_id within a group is defined, not a crash ---
    def test_17_duplicate_course_id(self):
        f = _feat(
            [
                _row(course="X", pass_rate=0.9),  # d=0.1
                _row(course="X", pass_rate=0.5),  # d=0.5, same course id
            ]
        )
        # Treated as two peers; no crash.
        self.assertAlmostEqual(f["concurrent_peer_difficulty_mean"].iloc[0], 0.5)
        self.assertAlmostEqual(f["concurrent_peer_difficulty_mean"].iloc[1], 0.1)

    # --- 18. target's OWN pass_rate NaN but peers valid -> computed from peers ---
    def test_18_own_nan_peers_valid(self):
        f = _feat(
            [
                _row(course="A", pass_rate=np.nan),  # target invalid
                _row(course="B", pass_rate=0.6),  # d=0.4
                _row(course="C", pass_rate=0.4),  # d=0.6
            ]
        )
        # denominator = group_valid_count (2), since own is not valid
        self.assertAlmostEqual(
            f["concurrent_peer_difficulty_mean"].iloc[0], (0.4 + 0.6) / 2
        )
        self.assertAlmostEqual(f["concurrent_peer_difficulty_max"].iloc[0], 0.6)
        self.assertEqual(f["concurrent_peer_difficulty_missing"].iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
