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
    occurrence=None,
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
        "student_course_id": (
            occurrence
            if occurrence is not None
            else f"{uni}|{student}|{degree}|{part}|{course}"
        ),
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

    # --- 17. one course contributes exactly ONE peer, not one per occurrence ---
    def test_17_duplicate_course_collapses_to_a_single_peer(self):
        # X registered twice with identical metadata + Y once. Peer membership
        # is {X, Y}, so every row sees exactly one peer -- never a phantom
        # second X. d(X)=0.1, d(Y)=0.5.
        f = _feat(
            [
                _row(course="X", pass_rate=0.9),
                _row(course="X", pass_rate=0.9),
                _row(course="Y", pass_rate=0.5),
            ]
        )
        self.assertTrue((f["concurrent_peer_observed_count"] == 1).all())
        # Both X rows resolve to the same peer entry and see only Y.
        self.assertAlmostEqual(f["concurrent_peer_difficulty_mean"].iloc[0], 0.5)
        self.assertAlmostEqual(f["concurrent_peer_difficulty_mean"].iloc[1], 0.5)
        self.assertAlmostEqual(f["concurrent_peer_difficulty_max"].iloc[0], 0.5)
        # Y sees only X.
        self.assertAlmostEqual(f["concurrent_peer_difficulty_mean"].iloc[2], 0.1)
        self.assertAlmostEqual(f["concurrent_peer_difficulty_max"].iloc[2], 0.1)

    # --- 17b. conflicting duplicate occurrences fail loudly, never a silent pick ---
    def test_17b_conflicting_duplicate_occurrences_raise(self):
        for column, rows in (
            (
                "course_pass_rate_historical",
                [_row(course="X", pass_rate=0.9), _row(course="X", pass_rate=0.5)],
            ),
            (
                "requirement_type_id",
                [
                    _row(course="X", pass_rate=0.9, req_type=1),
                    _row(course="X", pass_rate=0.9, req_type=2),
                ],
            ),
        ):
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, f"disagree on {column!r}"):
                    _feat(rows)

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

    # --- 19. two-input mode includes registered courses absent from targets ---
    def test_19_roster_only_peer_is_included(self):
        roster = _df(
            [
                _row(course="A", occurrence="occ-A", pass_rate=0.9),
                _row(course="W", occurrence="occ-W", pass_rate=0.2),
            ],
            index=[101, 102],
        )
        target = roster.iloc[[0]].copy()
        target.index = [77]

        f = compute_concurrent_group_features(target, roster)

        # W is roster-only, but is still A's registered peer: d(W) = 0.8.
        self.assertEqual(list(f.index), [77])
        self.assertAlmostEqual(f["concurrent_peer_difficulty_mean"].iloc[0], 0.8)
        self.assertAlmostEqual(f["concurrent_peer_difficulty_max"].iloc[0], 0.8)
        self.assertEqual(f["concurrent_peer_observed_count"].iloc[0], 1)
        self.assertEqual(f["concurrent_peer_set_empty"].iloc[0], 0)

    # --- 20. two-input occurrence identity must be exact and unambiguous ---
    def test_20_two_input_rejects_duplicate_unmatched_and_semester_mismatch(self):
        target = _df(
            [_row(course="A", occurrence="occ-A", pass_rate=0.9)]
        )
        good_roster = _df(
            [
                _row(course="A", occurrence="occ-A", pass_rate=0.9),
                _row(course="B", occurrence="occ-B", pass_rate=0.8),
            ]
        )

        duplicate_roster = pd.concat(
            [good_roster, good_roster.iloc[[0]]], ignore_index=True
        )
        with self.assertRaisesRegex(ValueError, "exactly one occurrence"):
            compute_concurrent_group_features(target, duplicate_roster)

        unmatched_target = target.copy()
        unmatched_target["student_course_id"] = "not-in-roster"
        with self.assertRaisesRegex(ValueError, "unmatched"):
            compute_concurrent_group_features(unmatched_target, good_roster)

        equivalent_semester_target = target.copy()
        equivalent_semester_target["part_id"] = pd.to_numeric(
            equivalent_semester_target["part_id"]
        ).astype("int64")
        equivalent = compute_concurrent_group_features(
            equivalent_semester_target, good_roster
        )
        self.assertEqual(len(equivalent), 1)

        wrong_semester_target = target.copy()
        wrong_semester_target["part_id"] = "20242"
        with self.assertRaisesRegex(ValueError, "SEMESTER_KEY mismatch"):
            compute_concurrent_group_features(wrong_semester_target, good_roster)

    # --- 21. explicit audit indicators separate empty set from missing values ---
    def test_21_empty_set_and_missing_value_indicators(self):
        singleton = _feat([_row(course="only", pass_rate=np.nan)])
        self.assertEqual(singleton["concurrent_peer_set_empty"].iloc[0], 1)
        self.assertEqual(
            singleton["concurrent_peer_difficulty_values_missing"].iloc[0], 0
        )
        self.assertEqual(
            singleton["concurrent_peer_difficulty_missing"].iloc[0],
            singleton["concurrent_peer_set_empty"].iloc[0],
        )

        unusable_peers = _feat(
            [
                _row(course="A", pass_rate=np.nan),
                _row(course="B", pass_rate=np.nan),
            ]
        )
        self.assertTrue((unusable_peers["concurrent_peer_set_empty"] == 0).all())
        self.assertTrue(
            (
                unusable_peers["concurrent_peer_difficulty_values_missing"] == 1
            ).all()
        )
        self.assertTrue(
            (
                unusable_peers["concurrent_peer_difficulty_missing"]
                == unusable_peers["concurrent_peer_set_empty"]
            ).all()
        )
        for column in (
            "concurrent_peer_difficulty_missing",
            "concurrent_peer_set_empty",
            "concurrent_peer_difficulty_values_missing",
        ):
            self.assertEqual(str(unusable_peers[column].dtype), "int64")

    # --- 22. roster selection preserves arbitrary target order and index ---
    def test_22_two_input_preserves_target_order_and_index(self):
        roster = _df(
            [
                _row(course="A", occurrence="occ-A", pass_rate=0.9),
                _row(course="B", occurrence="occ-B", pass_rate=0.5),
                _row(course="C", occurrence="occ-C", pass_rate=0.7),
            ],
            index=[5, 6, 7],
        )
        target = roster.iloc[[2, 0]].copy()
        target.index = [42, 11]

        out = add_concurrent_group_features(target, roster)

        self.assertEqual(list(out.index), [42, 11])
        self.assertEqual(list(out["student_course_id"]), ["occ-C", "occ-A"])
        self.assertAlmostEqual(out["concurrent_peer_difficulty_max"].iloc[0], 0.5)
        self.assertAlmostEqual(out["concurrent_peer_difficulty_max"].iloc[1], 0.5)

    # --- 24. REGRESSION ANCHOR: one-input == two-input when roster == targets ---
    def test_24_one_input_equals_two_input_when_roster_equals_targets(self):
        # The guard that keeps train/serve parity from silently breaking:
        # score_plan uses the one-input path, the builder uses the two-input
        # path, and they must agree exactly when the plan IS the roster.
        # Frame carries NaNs, a tie for the max, a duplicate course, and a
        # separate singleton group.
        df = _df(
            [
                _row(course="A", occurrence="o1", pass_rate=0.1),  # d=0.9
                _row(course="B", occurrence="o2", pass_rate=0.1),  # d=0.9 (tie)
                _row(course="C", occurrence="o3", pass_rate=np.nan),  # invalid
                _row(course="A", occurrence="o4", pass_rate=0.1),  # dup course
                _row(student="s2", course="Z", occurrence="o5", pass_rate=0.4),
            ],
            index=[11, 4, 9, 2, 7],
        )

        one_input = compute_concurrent_group_features(df)
        two_input = compute_concurrent_group_features(df, df)
        pd.testing.assert_frame_equal(one_input, two_input)

        # Hand-computed: membership is {A, B, C}; A collapses to one peer.
        # valid d = {A:0.9, B:0.9}; every row has 2 observed peers.
        # A: (1.8-0.9)/(2-1) = 0.9 ; tie for max keeps 0.9
        # C: own invalid -> 1.8/2 = 0.9 ; max 0.9
        mean = one_input["concurrent_peer_difficulty_mean"].to_numpy()
        mx = one_input["concurrent_peer_difficulty_max"].to_numpy()
        for position in range(4):
            self.assertAlmostEqual(mean[position], 0.9)
            self.assertAlmostEqual(mx[position], 0.9)
        self.assertTrue(
            (one_input["concurrent_peer_observed_count"].iloc[:4] == 2).all()
        )
        self.assertTrue(
            (one_input["concurrent_peer_difficulty_missing"].iloc[:4] == 0).all()
        )
        # The singleton group is untouched by any of that.
        self.assertEqual(one_input["concurrent_peer_observed_count"].iloc[4], 0)
        self.assertEqual(one_input["concurrent_peer_difficulty_missing"].iloc[4], 1)
        self.assertTrue(np.isnan(mean[4]))
        self.assertTrue(np.isnan(mx[4]))
        # Index and order survive both paths.
        self.assertEqual(list(one_input.index), [11, 4, 9, 2, 7])

    # --- 23. binding 44-feature contract remains unchanged ---
    def test_23_model_contract_stays_at_44_with_audit_columns_excluded(self):
        from src.model_training import EXPECTED_FEATURE_COUNT, MODEL_FEATURES

        legacy = "concurrent_peer_difficulty_missing"
        self.assertEqual(EXPECTED_FEATURE_COUNT, 44)
        self.assertEqual(len(MODEL_FEATURES), 44)
        self.assertIn(legacy, MODEL_FEATURES)
        self.assertEqual(MODEL_FEATURES.index(legacy), 35)
        self.assertNotIn("concurrent_peer_set_empty", MODEL_FEATURES)
        self.assertNotIn("concurrent_peer_difficulty_values_missing", MODEL_FEATURES)

    # --- 24. the re-based dataset-builder position gate stays protective ---
    def test_24_position_gate_stays_protective_once_concurrent_43_exists(self):
        from scripts.build_concurrent_group_features import (
            EXPECTED_LEGACY_MODEL_POSITION,
            _assert_contract,
        )
        from src.model_training import CONCURRENT_43_FEATURES, CONCURRENT_44_FEATURES

        legacy = "concurrent_peer_difficulty_missing"
        # The gate is pinned to concurrent_44 explicitly now, not to the
        # deprecated MODEL_FEATURES/EXPECTED_FEATURE_COUNT globals, so it must
        # keep passing exactly as before even though a second, real contract
        # (concurrent_43) now exists and drops this exact column.
        result = _assert_contract()
        self.assertTrue(all(result["gates"].values()), result["gates"])
        self.assertEqual(result["expected_feature_count"], 44)
        self.assertEqual(result["model_feature_count"], 44)
        self.assertEqual(result["legacy_indicator_zero_based_position"], 35)
        self.assertEqual(EXPECTED_LEGACY_MODEL_POSITION, 35)
        self.assertEqual(CONCURRENT_44_FEATURES.index(legacy), 35)
        # concurrent_43 legitimately drops the indicator; the dataset-builder
        # gate must not be sensitive to that (it only guards concurrent_44).
        self.assertNotIn(legacy, CONCURRENT_43_FEATURES)
        self.assertEqual(len(CONCURRENT_43_FEATURES), 43)


if __name__ == "__main__":
    unittest.main()
