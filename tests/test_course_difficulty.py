import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.course_difficulty import (
    DIFFICULTY_OUTPUT_COLUMNS,
    DifficultyConfig,
    _composite_key,
    apply_difficulty_state,
    build_temporal_train,
    fit_difficulty_state,
    load_difficulty_state,
    save_difficulty_state,
)
from src.inference import StudentScorer


def _row(
    part_id,
    degree,
    course,
    faculty,
    requirement,
    credits,
    mark,
    attempt=1,
):
    return {
        "part_id": str(part_id),
        "degree_course_key": f"{degree}__{course}",
        "degree_id": degree,
        "course_id": course,
        "faculty_id": faculty,
        "requirement_type_id": requirement,
        "course_credits": float(credits),
        "final_mark": float(mark),
        "attempt_number": int(attempt),
        "untouched": f"{part_id}-{degree}-{course}",
    }


class TemporalCourseDifficultyTests(unittest.TestCase):
    def setUp(self):
        self.config = DifficultyConfig(min_support=2, shrinkage_k=2.0)

    def test_composite_key_is_chunked_and_preserves_key_semantics(self):
        index = pd.Index([9, 3, 3, 7, 1], name="source_row")
        frame = pd.DataFrame(
            {
                "degree": pd.array(["D1", "D2", "D3", pd.NA, "D5"], dtype="string"),
                "requirement": pd.array([1, 2, pd.NA, 4, 5], dtype="Int64"),
                "credits": [3.0, 4.5, 2.0, 1.0, 6.0],
            },
            index=index,
        )
        expected = pd.Series(
            ["D1__1__3.0", "D2__2__4.5", pd.NA, pd.NA, "D5__5__6.0"],
            index=index,
            dtype="string",
        )

        # A two-row chunk forces three chunks, including boundaries with nulls
        # and a duplicate/non-default index.
        with patch("src.course_difficulty._COMPOSITE_KEY_CHUNK_ROWS", 2):
            actual = _composite_key(frame, ["degree", "requirement", "credits"])

        assert_series_equal(actual, expected)

    def test_composite_key_does_not_use_row_wise_dataframe_aggregation(self):
        frame = pd.DataFrame({"left": ["A", "B"], "right": [1, 2]})
        with patch.object(
            pd.DataFrame,
            "agg",
            side_effect=AssertionError("row-wise DataFrame.agg must not be used"),
        ):
            actual = _composite_key(frame, ["left", "right"])

        expected = pd.Series(["A__1", "B__2"], dtype="string")
        assert_series_equal(actual, expected)

    def test_composite_key_handles_empty_and_single_column_inputs(self):
        frame = pd.DataFrame({"value": pd.array(["A", pd.NA], dtype="string")})

        assert_series_equal(
            _composite_key(frame, ["value"]),
            pd.Series(["A", pd.NA], dtype="string"),
        )
        assert_series_equal(
            _composite_key(frame, []),
            pd.Series(["", ""], dtype="string"),
        )
        with self.assertRaises(KeyError):
            _composite_key(pd.DataFrame(), ["missing"])

    def test_first_semester_has_no_history_and_same_semester_does_not_leak(self):
        frame = pd.DataFrame(
            [
                _row("20201", "D1", "C1", "F1", 1, 3, 10),
                _row("20201", "D1", "C1", "F1", 1, 3, 90),
                _row("20202", "D1", "C1", "F1", 1, 3, 80),
                _row("20202", "D1", "C1", "F1", 1, 3, 20),
            ]
        )
        result = build_temporal_train(frame, self.config)

        first = result[result["part_id"] == "20201"]
        self.assertTrue((first["difficulty_fallback_level"] == 6).all())
        self.assertTrue((first["course_history_count"] == 0).all())
        self.assertTrue(first["course_pass_rate_historical"].isna().all())

        second = result[result["part_id"] == "20202"]
        self.assertEqual(second["course_pass_rate_historical"].nunique(), 1)
        self.assertEqual(second["course_avg_mark_historical"].nunique(), 1)
        self.assertEqual(second["course_history_count"].tolist(), [2, 2])

    def test_current_and_future_targets_cannot_change_past_or_current_features(self):
        frame = pd.DataFrame(
            [
                _row("20201", "D1", "C1", "F1", 1, 3, 40),
                _row("20202", "D1", "C1", "F1", 1, 3, 60),
                _row("20203", "D1", "C1", "F1", 1, 3, 80),
            ]
        )
        changed = frame.copy()
        changed.loc[changed["part_id"].isin(["20202", "20203"]), "final_mark"] = [0, 100]

        original_result = build_temporal_train(frame, self.config)
        changed_result = build_temporal_train(changed, self.config)
        through_second = frame["part_id"].isin(["20201", "20202"])
        assert_frame_equal(
            original_result.loc[through_second, DIFFICULTY_OUTPUT_COLUMNS],
            changed_result.loc[through_second, DIFFICULTY_OUTPUT_COLUMNS],
        )

    def test_all_six_fallback_levels_and_flag_definitions(self):
        history = pd.DataFrame(
            [
                _row("20201", "D1", "C1", "F1", 1, 3, 80),
                _row("20201", "D1", "C2", "F1", 1, 3, 40, 2),
                _row("20201", "D2", "C3", "F1", 1, 3, 60),
                _row("20201", "D3", "C4", "F2", 1, 3, 20, 2),
            ]
        )
        state = fit_difficulty_state(history, self.config)
        query = pd.DataFrame(
            [
                _row("20202", "D1", "C1", "F1", 1, 3, 0),  # L1
                _row("20202", "DX", "C1", "FX", 9, 9, 0),  # L2
                _row("20202", "D1", "NEW3", "F1", 1, 3, 0),  # L3
                _row("20202", "DX", "NEW4", "F1", 1, 3, 0),  # L4
                _row("20202", "DY", "NEW5", "FY", 1, 3, 0),  # L5
                _row("20202", "DZ", "NEW6", "FZ", 9, 9, 0),  # L6
            ]
        )
        result = apply_difficulty_state(query, state)

        self.assertEqual(result["difficulty_fallback_level"].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(result["course_is_new"].tolist(), [0, 0, 1, 1, 1, 1])
        self.assertEqual(result["course_history_count"].tolist(), [1, 1, 0, 0, 0, 0])
        self.assertEqual(result["course_low_support"].tolist(), [1, 1, 0, 0, 0, 0])
        self.assertEqual(result["course_difficulty_missing"].tolist(), [1, 1, 1, 1, 1, 1])

    def test_shrinkage_uses_direct_structural_parent_for_all_three_rates(self):
        history = pd.DataFrame(
            [
                _row("20201", "D1", "C1", "F1", 1, 3, 100, 2),
                _row("20201", "D2", "C1", "F2", 2, 4, 0, 1),
                _row("20201", "D1", "C2", "F1", 1, 3, 100, 2),
                _row("20201", "D3", "C3", "F1", 1, 3, 100, 2),
                _row("20201", "D4", "C4", "F2", 1, 3, 0, 1),
            ]
        )
        state = fit_difficulty_state(history, self.config)

        l2 = state.tables[2].loc["C1"]
        l1 = state.tables[1].loc["D1__C1"]
        expected_l1_pass = (1.0 + 2.0 * l2["course_pass_rate_historical"]) / 3.0
        expected_l1_mark = (100.0 + 2.0 * l2["course_avg_mark_historical"]) / 3.0
        expected_l1_retake = (1.0 + 2.0 * l2["course_retake_rate_historical"]) / 3.0
        self.assertAlmostEqual(l1["course_pass_rate_historical"], expected_l1_pass)
        self.assertAlmostEqual(l1["course_avg_mark_historical"], expected_l1_mark)
        self.assertAlmostEqual(l1["course_retake_rate_historical"], expected_l1_retake)

        l4 = state.tables[4].loc["F1__1__3"]
        l3 = state.tables[3].loc["D1__1__3"]
        self.assertEqual(l3["parent_key"], "F1__1__3")
        expected_l3_pass = (2.0 + 2.0 * l4["course_pass_rate_historical"]) / 4.0
        expected_l3_mark = (200.0 + 2.0 * l4["course_avg_mark_historical"]) / 4.0
        expected_l3_retake = (2.0 + 2.0 * l4["course_retake_rate_historical"]) / 4.0
        self.assertAlmostEqual(l3["course_pass_rate_historical"], expected_l3_pass)
        self.assertAlmostEqual(l3["course_avg_mark_historical"], expected_l3_mark)
        self.assertAlmostEqual(l3["course_retake_rate_historical"], expected_l3_retake)

    def test_non_difficulty_columns_and_index_are_preserved(self):
        frame = pd.DataFrame(
            [
                _row("20201", "D1", "C1", "F1", 1, 3, 50),
                _row("20202", "D1", "C1", "F1", 1, 3, 60),
            ],
            index=[7, 3],
        )
        result = build_temporal_train(frame, self.config)
        assert_frame_equal(frame, result[frame.columns])
        self.assertEqual(result.index.tolist(), [7, 3])

    def test_persisted_state_round_trip_is_identical(self):
        history = pd.DataFrame(
            [
                _row("20201", "D1", "C1", "F1", 1, 3, 80),
                _row("20201", "D2", "C2", "F1", 1, 3, 20, 2),
            ]
        )
        query = pd.DataFrame([_row("20202", "D1", "C1", "F1", 1, 3, 0)])
        state = fit_difficulty_state(history, self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state"
            save_difficulty_state(state, path, metadata={"test": True})
            loaded = load_difficulty_state(path)
            expected = apply_difficulty_state(query, state)
            actual = apply_difficulty_state(query, loaded)
            assert_series_equal(
                expected.loc[0, DIFFICULTY_OUTPUT_COLUMNS],
                actual.loc[0, DIFFICULTY_OUTPUT_COLUMNS],
                check_names=False,
            )

    def test_inference_lookup_matches_batch_application(self):
        history = pd.DataFrame(
            [
                _row("20201", "D1", "C1", "F1", 1, 3, 80),
                _row("20201", "D2", "C2", "F1", 1, 3, 20, 2),
            ]
        )
        state = fit_difficulty_state(history, self.config)
        scorer = StudentScorer(object(), object(), state)
        query = pd.DataFrame([_row("20202", "D9", "NEW", "F1", 1, 3, 0)])

        batch = apply_difficulty_state(query, state).iloc[0]
        online = scorer._get_difficulty_row(
            "D9",
            "NEW",
            faculty_id="F1",
            requirement_type_id=1,
            course_credits=3.0,
        )
        assert_series_equal(
            batch[DIFFICULTY_OUTPUT_COLUMNS],
            online[DIFFICULTY_OUTPUT_COLUMNS],
            check_names=False,
        )


if __name__ == "__main__":
    unittest.main()
