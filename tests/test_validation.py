"""Tests for src.validation and for the guards routed through it.

Part A covers the two helpers directly: exception class, exact message text,
and the signature contract that ``label``, ``error`` and ``check_index`` are
all keyword-only with no defaults.

Part B covers the 16 call sites, asserting the exact message each one now
produces. Unlike the previous detection-only design, the message text is owned
by ``src.validation``, so these tests are the definition of what a guard says
rather than a record of what it used to say.

NOTE on ``KeyError``: ``str(KeyError("abc"))`` is ``"'abc'"``, not ``"abc"`` --
``KeyError.__str__`` reprs its argument. Every KeyError assertion below
compares ``exception.args[0]``, never ``str(exception)``.
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
<<<<<<< HEAD
from src.registration_roster import (
    _prepare_acd,
    _prepare_target,
    build_registration_roster,
)
from src.validation import assert_shape_preserved, require_columns
=======
from src.registration_roster import _require_columns
from src.validation import find_missing_columns, shape_changed
>>>>>>> 770c7a147a9b3b0664121b3c8c165bf6cbfc57a9


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


def _valid_target_frame():
    """A target frame that clears every check inside _prepare_target."""
    return pd.DataFrame(
        [
            {
                "university_id": "111",
                "student_id": "s1",
                "degree_id": "d1",
                "part_id": "20241",
                "student_course_id": "o1",
                "course_id": "c1",
            }
        ]
    )


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
<<<<<<< HEAD
# Part A -- the helpers themselves
=======
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
>>>>>>> 770c7a147a9b3b0664121b3c8c165bf6cbfc57a9
# ==========================================================================

class RequireColumnsHelper(unittest.TestCase):
    def test_a1_silent_when_every_column_is_present(self):
        frame = pd.DataFrame(columns=["a", "b", "c"])
        self.assertIsNone(
            require_columns(frame, ["a", "c"], label="x", error=KeyError)
        )

    def test_a2_raises_the_given_class_keyerror_with_exact_message(self):
        frame = pd.DataFrame(columns=["a"])
        with self.assertRaises(KeyError) as cm:
            require_columns(frame, ["a", "b"], label="roster", error=KeyError)
        self.assertEqual(
            cm.exception.args[0], "roster: missing required columns: ['b']"
        )

    def test_a3_raises_the_given_class_valueerror_with_exact_message(self):
        frame = pd.DataFrame(columns=["a"])
        with self.assertRaises(ValueError) as cm:
            require_columns(frame, ["a", "b"], label="roster", error=ValueError)
        self.assertEqual(
            str(cm.exception), "roster: missing required columns: ['b']"
        )

    def test_a4_missing_list_is_sorted_not_argument_order(self):
        frame = pd.DataFrame(columns=["a"])
        with self.assertRaises(ValueError) as cm:
            require_columns(frame, ["z", "b", "a", "y"], label="f", error=ValueError)
        self.assertEqual(
            str(cm.exception), "f: missing required columns: ['b', 'y', 'z']"
        )

    def test_a5_a_set_argument_produces_a_deterministic_message(self):
        # Python randomises string hashing per process, so an unsorted set
        # would give a different message on every run.
        frame = pd.DataFrame(columns=["degree_id"])
        required = {"part_id", "final_mark", "degree_id", "attempt_number"}
        with self.assertRaises(ValueError) as cm:
            require_columns(frame, required, label="course difficulty", error=ValueError)
        self.assertEqual(
            str(cm.exception),
            "course difficulty: missing required columns: "
            "['attempt_number', 'final_mark', 'part_id']",
        )

    def test_a6_duplicate_frame_column_names_still_count_as_present(self):
        frame = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
        with self.assertRaises(KeyError) as cm:
            require_columns(frame, ["a", "b", "c"], label="f", error=KeyError)
        self.assertEqual(cm.exception.args[0], "f: missing required columns: ['c']")


class AssertShapePreservedHelper(unittest.TestCase):
    def test_a7_silent_for_an_identical_frame(self):
        frame = pd.DataFrame({"x": [1, 2, 3]})
        self.assertIsNone(
            assert_shape_preserved(
                frame, frame, label="x", check_index=True, error=AssertionError
            )
        )

    def test_a8_row_count_change_message_names_both_counts(self):
        before = pd.DataFrame({"x": [1, 2, 3]})
        after = before.iloc[:-1]
        with self.assertRaises(AssertionError) as cm:
            assert_shape_preserved(
                before, after, label="enrichment", check_index=True,
                error=AssertionError,
            )
        self.assertEqual(
            str(cm.exception), "enrichment: row count changed: 3 -> 2"
        )

    def test_a9_index_change_message_is_distinct_from_row_count(self):
        before = pd.DataFrame({"x": [1, 2, 3]}, index=[0, 1, 2])
        after = pd.DataFrame({"x": [3, 2, 1]}, index=[2, 1, 0])
        with self.assertRaises(ValueError) as cm:
            assert_shape_preserved(
                before, after, label="enrichment", check_index=True,
                error=ValueError,
            )
        self.assertEqual(
            str(cm.exception),
            "enrichment: row order or index changed (row count 3 unchanged)",
        )

    def test_a10_check_index_false_ignores_a_reordered_index(self):
        before = pd.DataFrame({"x": [1, 2, 3]}, index=[0, 1, 2])
        after = pd.DataFrame({"x": [3, 2, 1]}, index=[2, 1, 0])
        self.assertIsNone(
            assert_shape_preserved(
                before, after, label="x", check_index=False, error=AssertionError
            )
        )

    def test_a11_row_count_change_is_caught_even_with_check_index_false(self):
        before = pd.DataFrame({"x": [1, 2, 3]})
        with self.assertRaises(AssertionError) as cm:
            assert_shape_preserved(
                before, before.iloc[:-1], label="e", check_index=False,
                error=AssertionError,
            )
        self.assertEqual(str(cm.exception), "e: row count changed: 3 -> 2")


class HelperSignatureContract(unittest.TestCase):
    """No keyword may acquire a default; every one is keyword-only."""

    def _assert_required_keyword_only(self, func, names):
        parameters = inspect.signature(func).parameters
        for name in names:
            with self.subTest(function=func.__name__, parameter=name):
                self.assertIs(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIs(parameters[name].default, inspect.Parameter.empty)

    def test_a12_all_keywords_are_keyword_only_and_have_no_defaults(self):
        self._assert_required_keyword_only(require_columns, ["label", "error"])
        self._assert_required_keyword_only(
            assert_shape_preserved, ["label", "check_index", "error"]
        )
        # Omitting them is a TypeError, not a silent default.
        with self.assertRaises(TypeError):
            require_columns(pd.DataFrame(), [])
        with self.assertRaises(TypeError):
            assert_shape_preserved(pd.DataFrame(), pd.DataFrame())


# ==========================================================================
# Part B -- the 16 call sites
# ==========================================================================

class CourseDifficultySites(unittest.TestCase):
    def test_b01_site06_composite_key(self):
        with self.assertRaises(KeyError) as cm:
            _composite_key(pd.DataFrame(), ["missing"])
        self.assertEqual(
            cm.exception.args[0],
            "composite key: missing required columns: ['missing']",
        )

    def test_b02_site07_build_level_keys(self):
        frame = pd.DataFrame(columns=["degree_id", "course_credits", "unrelated_col"])
        with self.assertRaises(ValueError) as cm:
            build_level_keys(frame)
        self.assertEqual(
            str(cm.exception),
            "difficulty keys: missing required columns: "
            "['degree_course_key', 'faculty_id', 'requirement_type_id']",
        )

    def test_b03_site08_validate_training_frame(self):
        frame = pd.DataFrame(columns=["degree_id", "course_credits", "unrelated_col"])
        with self.assertRaises(ValueError) as cm:
            _validate_training_frame(frame)
        self.assertEqual(
            str(cm.exception),
            "course difficulty: missing required columns: "
            "['attempt_number', 'degree_course_key', 'faculty_id', "
            "'final_mark', 'part_id', 'requirement_type_id']",
        )

    def test_b04_site09_validate_query_frame(self):
        frame = pd.DataFrame(columns=["degree_id", "course_credits", "unrelated_col"])
        with self.assertRaises(ValueError) as cm:
            _validate_query_frame(frame)
        self.assertEqual(
            str(cm.exception),
            "course-difficulty queries: missing required columns: "
            "['degree_course_key', 'faculty_id', 'part_id', 'requirement_type_id']",
        )


class ConcurrentGroupFeatureSites(unittest.TestCase):
    def test_b05_site10_collapse_with_roster_source(self):
        frame = _concurrent_df(drop=("course_id", "requirement_type_id"))
        with self.assertRaises(KeyError) as cm:
            _collapse_to_peer_membership(frame, "roster")
        self.assertEqual(
            cm.exception.args[0],
            "concurrent group features (roster): missing required columns: "
            "['course_id', 'requirement_type_id']",
        )

    def test_b06_site10_collapse_with_target_rows_source(self):
        frame = _concurrent_df(drop=("course_id", "requirement_type_id"))
        with self.assertRaises(KeyError) as cm:
            _collapse_to_peer_membership(frame, "target rows")
        self.assertEqual(
            cm.exception.args[0],
            "concurrent group features (target rows): missing required columns: "
            "['course_id', 'requirement_type_id']",
        )

    def test_b07_site11_compute_roster_features(self):
        frame = _concurrent_df(drop=("course_id", "requirement_type_id"))
        with self.assertRaises(KeyError) as cm:
            _compute_roster_features(frame)
        self.assertEqual(
            cm.exception.args[0],
            "concurrent group features (roster aggregation): missing required "
            "columns: ['course_id', 'requirement_type_id']",
        )

    def test_b08_site12_two_input_target_side(self):
        target = _concurrent_df(drop=("student_course_id", "course_id"))
        roster = _concurrent_df()
        with self.assertRaises(KeyError) as cm:
            _validate_two_input_contract(target, roster)
        self.assertEqual(
            cm.exception.args[0],
            "concurrent group features (target rows): missing required columns: "
            "['course_id', 'student_course_id']",
        )

    def test_b09_site13_two_input_roster_side(self):
        target = _concurrent_df()
        roster = _concurrent_df(drop=("course_pass_rate_historical",))
        with self.assertRaises(KeyError) as cm:
            _validate_two_input_contract(target, roster)
        self.assertEqual(
            cm.exception.args[0],
            "concurrent group features (roster): missing required columns: "
            "['course_pass_rate_historical']",
        )


class RegistrationRosterSites(unittest.TestCase):
    """The three former _require_columns callers, now calling require_columns."""

    def test_b10_caller_prepare_target(self):
        # _normalize_source_frame always derives university_id, so it is never
        # reported missing.
        frame = pd.DataFrame([{"student_id": "s1"}])
        with self.assertRaises(KeyError) as cm:
            _prepare_target(frame)
        self.assertEqual(
            cm.exception.args[0],
            "target_frame: missing required columns: "
            "['course_id', 'degree_id', 'part_id', 'student_course_id']",
        )

    def test_b11_caller_prepare_acd(self):
        frame = pd.DataFrame([{"degree_id": "d1"}])
        with self.assertRaises(KeyError) as cm:
            _prepare_acd(frame)
        self.assertEqual(
            cm.exception.args[0],
            "clean_acd: missing required columns: "
            "['course_id', 'requirement_type_id']",
        )

    def test_b12_caller_build_registration_roster_raw_crg(self):
        raw = pd.DataFrame(
            [
                {
                    "student_course_id": "o1",
                    "student_id": "s1",
                    "course_id": "c1",
                    "part_id": "20241",
                    "degree_id": "d1",
                    "faculty_id": "f1",
                    "course_credits": 3.0,
                }
            ]
        )
        with self.assertRaises(KeyError) as cm:
            build_registration_roster(raw, _valid_target_frame(), pd.DataFrame())
        self.assertEqual(
            cm.exception.args[0],
            "raw_crg: missing required columns: ['active', 'register_status']",
        )


# B13-B16 use mock.patch rather than crafted input, and that is deliberate.
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
# build_temporal_train and build_temporal_query_difficulty each call
# _drop_stale_columns on their own input AND then call apply_difficulty_state,
# which calls it again. A patch that truncates every call therefore shortens
# the nested frame too, and the apply_difficulty_state guard fires first --
# control never reaches the temporal guards. Measured: that is exactly what
# happened before the identity scoping was added.
#
# So: under a row-dropping fault in _drop_stale_columns, the
# apply_difficulty_state guard shadows the two temporal guards. Those are NOT
# independently reachable through that fault. _TruncateOuterCall exists to
# reach them anyway, by leaving nested calls alone.

class ShapePreservedSites(unittest.TestCase):
    def test_b13_site01_apply_difficulty_state(self):
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
            "difficulty enrichment: row count changed: 2 -> 1",
        )

    def test_b14_site02_build_temporal_train(self):
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
            "temporal train enrichment: row count changed: 2 -> 1",
        )

    def test_b15_site03_build_temporal_query(self):
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
            "temporal query enrichment: row count changed: 2 -> 1",
        )

    def test_b16_site05_add_concurrent_group_features(self):
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
            "concurrent enrichment: row count changed: 2 -> 4",
        )

    # --- site 4: _select_for_targets -- DOCUMENTS UNREACHABILITY --------
    # Kept verbatim from the previous design: it pins pandas' own error, not
    # one of ours, so the message reversal does not affect it.
    def test_b15_site04_select_for_targets_guard_is_unreachable(self):
        """The guard at _select_for_targets cannot fire.

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
