"""Failure-path regression tests for the guards entering the validation batch.

These pin the EXACT exception class and EXACT message string of the 14 call
sites that are about to be refactored onto shared detection helpers
(``find_missing_columns`` / ``shape_changed`` in ``src.validation``).

They are written and committed BEFORE the extraction, deliberately. Written
afterwards they would only lock in whatever the rewrite happened to produce;
written first, any drift in a message or an exception class turns them red.

Measured coverage before this file existed: of these 14 guards, exactly one
had any failure-path test at all
(``test_course_difficulty.TemporalCourseDifficultyTests
.test_composite_key_handles_empty_and_single_column_inputs``), and it asserted
the exception class only. No test anywhere asserted any of the message
strings. The parity checks cannot cover these guards: they only fire on bad
data, and the parity fixtures are good data.

NOTE on ``KeyError``: ``str(KeyError("abc"))`` is ``"'abc'"``, not ``"abc"`` --
``KeyError.__str__`` reprs its argument. Every KeyError assertion below
therefore compares ``exception.args[0]``, never ``str(exception)``.
"""

import inspect
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.concurrent_group_features import (
    _collapse_to_peer_membership,
    _compute_roster_features,
    _select_for_targets,
    _validate_two_input_contract,
    add_concurrent_group_features,
)
from src.course_difficulty import (
    _composite_key,
    _validate_query_frame,
    _validate_training_frame,
    apply_difficulty_state,
    build_level_keys,
    build_temporal_query_difficulty,
    build_temporal_train,
    empty_difficulty_state,
)
from src.registration_roster import _require_columns
from src.validation import find_missing_columns, shape_changed


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------

def _difficulty_row(part_id="20241", degree="d", course="c"):
    """A row satisfying every course-difficulty required-column set."""
    return {
        "part_id": part_id,
        "degree_course_key": f"{degree}__{course}",
        "degree_id": degree,
        "course_id": course,
        "faculty_id": "f",
        "requirement_type_id": "1",
        "course_credits": 3.0,
        "final_mark": 70.0,
        "attempt_number": 1,
    }


def _two_difficulty_rows():
    return pd.DataFrame(
        [_difficulty_row(course="c1"), _difficulty_row(course="c2")]
    )


def _concurrent_row(course="c", occurrence="occ"):
    return {
        "university_id": "1",
        "student_id": "s",
        "degree_id": "d",
        "part_id": "20241",
        "course_id": course,
        "student_course_id": occurrence,
        "course_pass_rate_historical": 0.5,
        "course_difficulty_missing": 0,
        "requirement_type_id": 1,
    }


def _concurrent_df(drop=()):
    frame = pd.DataFrame([_concurrent_row("c1", "o1"), _concurrent_row("c2", "o2")])
    return frame.drop(columns=list(drop))


class _TruncateOuterCall:
    """``_drop_stale_columns`` stand-in that truncates only the OUTER call.

    Identity, not call order, decides: the outer call receives the exact
    parameter object the test passed in, while every nested call receives a
    fresh ``.iloc[...]`` slice. Anything that is not the outer frame passes
    through untouched, so a nested guard cannot fire first (see the shadowing
    note below).

    ``activations`` exists so the test can prove the truncating branch was
    actually taken. Without that assertion, a refactor that copies the frame
    before the outer call would break the identity match and silently turn
    these tests into no-ops that pass for the wrong reason.
    """

    def __init__(self, outer_frame):
        self._outer_frame = outer_frame
        self.activations = 0

    def __call__(self, df):
        if df is self._outer_frame:
            self.activations += 1
            return df.iloc[:-1]
        return df


# ==========================================================================
# Part A -- the detection helpers themselves
# ==========================================================================

class FindMissingColumns(unittest.TestCase):
    def test_a1_returns_empty_when_every_column_is_present(self):
        frame = pd.DataFrame(columns=["a", "b", "c"])
        self.assertEqual(find_missing_columns(frame, ["a", "c"]), [])

    def test_a2_preserves_argument_order_not_frame_order(self):
        frame = pd.DataFrame(columns=["a"])
        # Argument order is z, b, y -- deliberately neither sorted nor the
        # frame's order, because three call sites depend on argument order
        # being what lands in the message.
        self.assertEqual(
            find_missing_columns(frame, ["z", "b", "a", "y"]), ["z", "b", "y"]
        )

    def test_a3_sorted_input_reproduces_the_sorted_set_difference(self):
        # The invariant sites 7, 8 and 9 rely on: passing sorted(required)
        # yields exactly what `sorted(required - set(df.columns))` produced.
        required = {"part_id", "final_mark", "degree_id", "attempt_number"}
        frame = pd.DataFrame(columns=["degree_id", "unrelated"])
        self.assertEqual(
            find_missing_columns(frame, sorted(required)),
            sorted(required - set(frame.columns)),
        )

    def test_a4_duplicate_frame_column_names_still_count_as_present(self):
        frame = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
        self.assertEqual(find_missing_columns(frame, ["a", "b", "c"]), ["c"])


class ShapeChanged(unittest.TestCase):
    def test_a5_false_for_an_identical_frame(self):
        frame = pd.DataFrame({"x": [1, 2, 3]})
        self.assertIs(shape_changed(frame, frame, check_index=True), False)

    def test_a6_true_when_the_row_count_changes(self):
        before = pd.DataFrame({"x": [1, 2, 3]})
        after = before.iloc[:-1]
        self.assertIs(shape_changed(before, after, check_index=True), True)
        # A length change is caught even when the index is not compared.
        self.assertIs(shape_changed(before, after, check_index=False), True)

    def test_a7_true_on_a_reordered_index_when_check_index_is_true(self):
        before = pd.DataFrame({"x": [1, 2, 3]}, index=[0, 1, 2])
        after = pd.DataFrame({"x": [3, 2, 1]}, index=[2, 1, 0])
        self.assertIs(shape_changed(before, after, check_index=True), True)

    def test_a8_false_on_a_reordered_index_when_check_index_is_false(self):
        before = pd.DataFrame({"x": [1, 2, 3]}, index=[0, 1, 2])
        after = pd.DataFrame({"x": [3, 2, 1]}, index=[2, 1, 0])
        self.assertIs(shape_changed(before, after, check_index=False), False)

    def test_a9_check_index_is_keyword_only_and_has_no_default(self):
        """A default would let a call site silently change what it compares."""
        parameter = inspect.signature(shape_changed).parameters["check_index"]
        self.assertIs(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(parameter.default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            shape_changed(pd.DataFrame(), pd.DataFrame())


# ==========================================================================
# Part B -- require_columns sites: KeyError (6) and ValueError (3)
# ==========================================================================

class RequireColumnsSiteMessages(unittest.TestCase):
    """Sites 6-14: the missing-required-columns family."""

    # --- site 6: course_difficulty._composite_key -----------------------
    def test_b01_site06_composite_key_raises_keyerror_with_exact_message(self):
        with self.assertRaises(KeyError) as cm:
            _composite_key(pd.DataFrame(), ["missing"])
        self.assertEqual(
            cm.exception.args[0],
            "Composite-key columns not found: ['missing']",
        )

    # --- site 7: course_difficulty.build_level_keys ---------------------
    def test_b02_site07_build_level_keys_raises_valueerror_with_exact_message(self):
        frame = pd.DataFrame(columns=["degree_id", "course_credits", "unrelated_col"])
        with self.assertRaises(ValueError) as cm:
            build_level_keys(frame)
        self.assertEqual(
            str(cm.exception),
            "Missing columns required for difficulty keys: "
            "['degree_course_key', 'faculty_id', 'requirement_type_id']",
        )

    # --- site 8: course_difficulty._validate_training_frame -------------
    def test_b03_site08_validate_training_frame_raises_valueerror_with_exact_message(self):
        frame = pd.DataFrame(columns=["degree_id", "course_credits", "unrelated_col"])
        with self.assertRaises(ValueError) as cm:
            _validate_training_frame(frame)
        self.assertEqual(
            str(cm.exception),
            "Missing columns required for course difficulty: "
            "['attempt_number', 'degree_course_key', 'faculty_id', "
            "'final_mark', 'part_id', 'requirement_type_id']",
        )

    # --- site 9: course_difficulty._validate_query_frame ----------------
    def test_b04_site09_validate_query_frame_raises_valueerror_with_exact_message(self):
        frame = pd.DataFrame(columns=["degree_id", "course_credits", "unrelated_col"])
        with self.assertRaises(ValueError) as cm:
            _validate_query_frame(frame)
        self.assertEqual(
            str(cm.exception),
            "Missing columns required for course-difficulty queries: "
            "['degree_course_key', 'faculty_id', 'part_id', 'requirement_type_id']",
        )

    # --- site 10: _collapse_to_peer_membership, both label values -------
    # The message interpolates ``label``, so both callers are pinned.
    def test_b05_site10_collapse_roster_label_raises_keyerror_with_exact_message(self):
        frame = _concurrent_df(drop=("course_id", "requirement_type_id"))
        with self.assertRaises(KeyError) as cm:
            _collapse_to_peer_membership(frame, "roster")
        self.assertEqual(
            cm.exception.args[0],
            "compute_concurrent_group_features roster missing required "
            "columns: ['course_id', 'requirement_type_id']",
        )

    def test_b06_site10_collapse_target_label_raises_keyerror_with_exact_message(self):
        frame = _concurrent_df(drop=("course_id", "requirement_type_id"))
        with self.assertRaises(KeyError) as cm:
            _collapse_to_peer_membership(frame, "target rows")
        self.assertEqual(
            cm.exception.args[0],
            "compute_concurrent_group_features target rows missing required "
            "columns: ['course_id', 'requirement_type_id']",
        )

    # --- site 11: _compute_roster_features ------------------------------
    def test_b07_site11_compute_roster_features_raises_keyerror_with_exact_message(self):
        frame = _concurrent_df(drop=("course_id", "requirement_type_id"))
        with self.assertRaises(KeyError) as cm:
            _compute_roster_features(frame)
        self.assertEqual(
            cm.exception.args[0],
            "compute_concurrent_group_features missing required columns: "
            "['course_id', 'requirement_type_id']",
        )

    # --- site 12: _validate_two_input_contract, target side -------------
    def test_b08_site12_two_input_target_raises_keyerror_with_exact_message(self):
        target = _concurrent_df(drop=("student_course_id", "course_id"))
        roster = _concurrent_df()
        with self.assertRaises(KeyError) as cm:
            _validate_two_input_contract(target, roster)
        self.assertEqual(
            cm.exception.args[0],
            "compute_concurrent_group_features target rows missing required "
            "columns: ['student_course_id', 'course_id']",
        )

    # --- site 13: _validate_two_input_contract, roster side -------------
    def test_b09_site13_two_input_roster_raises_keyerror_with_exact_message(self):
        target = _concurrent_df()
        roster = _concurrent_df(drop=("course_pass_rate_historical",))
        with self.assertRaises(KeyError) as cm:
            _validate_two_input_contract(target, roster)
        self.assertEqual(
            cm.exception.args[0],
            "compute_concurrent_group_features roster missing required "
            "columns: ['course_pass_rate_historical']",
        )

    # --- site 14: registration_roster._require_columns ------------------
    def test_b10_site14_require_columns_raises_keyerror_with_exact_message(self):
        frame = pd.DataFrame(columns=["present"])
        with self.assertRaises(KeyError) as cm:
            _require_columns(frame, ["a", "b"], "clean_acd")
        self.assertEqual(
            cm.exception.args[0],
            "clean_acd is missing required columns: ['a', 'b']",
        )


# ==========================================================================
# shape_changed sites -- AssertionError (4 reachable) + 1 unreachable
# ==========================================================================

# B11-B14 use mock.patch rather than crafted input, and that is deliberate.
#
# On pandas 3.0.2 (the version pinned in .venv) no legitimate input can make
# these four conditions true. Measured: pd.concat([a, b], axis=1) where both
# frames carry the same duplicate index returns the original row count (3 in,
# 3 out) rather than fanning out; and every `features` frame in these code
# paths is constructed with index=<clean frame>.index, while _drop_stale_columns
# only ever drops COLUMNS. So length and index always agree by construction and
# the guard can only be reached by breaking an internal.
#
# If a future pandas makes duplicate-index concat(axis=1) fan out, these
# conditions become naturally reachable and the patches here should be replaced
# with real duplicate-index inputs.
#
# SHADOWING -- why the patch is identity-scoped to the outer call.
# build_temporal_train (course_difficulty.py:651) and
# build_temporal_query_difficulty (course_difficulty.py:714) each call
# _drop_stale_columns on their own input AND then call apply_difficulty_state,
# which calls it again at course_difficulty.py:569. A patch that truncates
# every call therefore shortens the nested frame too, and site 1's guard fires
# first with "Difficulty enrichment changed row count, order, or index" --
# control never reaches sites 2 and 3. Measured: that is exactly what happened
# on the first run of this file, before the identity scoping was added.
#
# So: under a row-dropping fault in _drop_stale_columns, site 1 shadows sites 2
# and 3. Those two guards are NOT independently reachable through that fault.
# _TruncateOuterCall exists to reach them anyway, by leaving nested calls alone.

class ShapePreservedSiteMessages(unittest.TestCase):
    """Sites 1-5: the row-count/index-preservation family."""

    # --- site 1: apply_difficulty_state ---------------------------------
    def test_b11_site01_apply_difficulty_state_raises_assertion_with_exact_message(self):
        frame = _two_difficulty_rows()
        truncate = _TruncateOuterCall(frame)
        with patch(
            "src.course_difficulty._drop_stale_columns", side_effect=truncate
        ):
            with self.assertRaises(AssertionError) as cm:
                apply_difficulty_state(frame, empty_difficulty_state())
        self.assertGreaterEqual(
            truncate.activations, 1, "outer frame was never truncated"
        )
        self.assertEqual(
            str(cm.exception),
            "Difficulty enrichment changed row count, order, or index",
        )

    # --- site 2: build_temporal_train -----------------------------------
    def test_b12_site02_build_temporal_train_raises_assertion_with_exact_message(self):
        frame = _two_difficulty_rows()
        truncate = _TruncateOuterCall(frame)
        with patch(
            "src.course_difficulty._drop_stale_columns", side_effect=truncate
        ):
            with self.assertRaises(AssertionError) as cm:
                build_temporal_train(frame)
        self.assertGreaterEqual(
            truncate.activations, 1, "outer frame was never truncated"
        )
        self.assertEqual(
            str(cm.exception),
            "Temporal train enrichment changed row count, order, or index",
        )

    # --- site 3: build_temporal_query_difficulty ------------------------
    def test_b13_site03_build_temporal_query_raises_assertion_with_exact_message(self):
        history = _two_difficulty_rows()
        query = _two_difficulty_rows()
        truncate = _TruncateOuterCall(query)
        with patch(
            "src.course_difficulty._drop_stale_columns", side_effect=truncate
        ):
            with self.assertRaises(AssertionError) as cm:
                build_temporal_query_difficulty(history, query)
        self.assertGreaterEqual(
            truncate.activations, 1, "outer frame was never truncated"
        )
        self.assertEqual(
            str(cm.exception),
            "Temporal query enrichment changed row count, order, or index",
        )

    # --- site 5: add_concurrent_group_features --------------------------
    def test_b14_site05_add_concurrent_features_raises_assertion_with_exact_message(self):
        target = _concurrent_df()
        disjoint = pd.DataFrame(
            {"concurrent_peer_difficulty_mean": [0.1, 0.2]},
            index=[10, 11],
        )
        with patch(
            "src.concurrent_group_features.compute_concurrent_group_features",
            return_value=disjoint,
        ):
            with self.assertRaises(AssertionError) as cm:
                add_concurrent_group_features(target)
        self.assertEqual(
            str(cm.exception),
            "Concurrent enrichment changed row count, order, or index",
        )

    # --- site 4: _select_for_targets -- DOCUMENTS UNREACHABILITY --------
    def test_b15_site04_select_for_targets_guard_is_unreachable(self):
        """The AssertionError at _select_for_targets cannot fire.

        ``out.index = target_df.index`` executes on the line immediately
        before the guard. If the lengths differ, that assignment raises
        pandas' own ValueError first; if it succeeds, ``out.index`` IS
        ``target_df.index`` and both disjuncts of the guard are false.

        This test pins pandas' error, so that reordering those two lines --
        which would make the guard live -- shows up here rather than silently.
        """
        features = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        target = pd.DataFrame({"x": [1, 2]})
        with self.assertRaisesRegex(ValueError, "Length mismatch"):
            _select_for_targets(features, np.array([0, 1, 2]), target)


if __name__ == "__main__":
    unittest.main()
