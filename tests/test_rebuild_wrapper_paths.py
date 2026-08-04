"""Path-resolution tests for the three feature wrappers.

Pure path and gate logic: nothing here builds a feature, reads a real split, or
writes into the project data root. Synthetic roots live under the OS temporary
directory.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import MODEL_DATA_DIR, MODEL_RUNS_DIR  # noqa: E402
from src.rebuild_paths import (  # noqa: E402
    CONCURRENT_SUBDIR,
    FEATURES_SUBDIR,
    SPLIT_SUBDIR,
)


def _load(module_name: str, filename: str):
    """Import a builder script by path without executing its ``main``."""
    path = PROJECT_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


B2 = _load("_wrapper_b2", "build_b2_temporal_course_stats.py")
TREND = _load("_wrapper_trend", "build_gpa_trend_dataset.py")
CONCURRENT = _load("_wrapper_concurrent", "build_concurrent_group_features.py")

SPLITS = ("train", "valid", "test")


class _RebuildRoot(unittest.TestCase):
    """A synthetic version root carrying the Phase 1 base artifacts."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="wrapper_paths_test_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "2026-08_temporal_rebuild_v1"
        (self.root / SPLIT_SUBDIR).mkdir(parents=True)
        (self.root / FEATURES_SUBDIR).mkdir()
        for basename in (
            "train_base_candidate.parquet",
            "valid_base_candidate.parquet",
            "test_provisional_base_candidate.parquet",
        ):
            (self.root / SPLIT_SUBDIR / basename).write_bytes(b"")
        for split_prefix in ("train", "valid", "test_provisional"):
            for generation in ("difficulty", "final"):
                (
                    self.root
                    / FEATURES_SUBDIR
                    / f"{split_prefix}_{generation}_candidate.parquet"
                ).write_bytes(b"")


class B2PathPlanTests(_RebuildRoot):
    def test_legacy_plan_is_unchanged_when_no_root_is_named(self) -> None:
        plan = B2._plan_io(
            B2.default_namespace(feature_contract="baseline_41"),
            Path("out"),
        )
        self.assertEqual(plan.mode, "legacy_generation_names")
        self.assertTrue(plan.has_final)
        self.assertEqual(
            plan.inputs["base"]["train"], MODEL_DATA_DIR / "df_train_base.parquet"
        )
        self.assertEqual(
            plan.inputs["final"]["valid"], MODEL_DATA_DIR / "df_valid_final.parquet"
        )
        self.assertEqual(
            plan.outputs["difficulty"]["test"], Path("out") / "df_test_difficulty.parquet"
        )
        self.assertEqual(
            plan.outputs["final"]["train"], Path("out") / "df_train_final.parquet"
        )

    def test_rebuild_root_resolves_base_only(self) -> None:
        plan = B2._plan_io(
            B2.default_namespace(
                feature_contract="baseline_41", rebuild_root=self.root
            ),
            self.root / FEATURES_SUBDIR,
        )
        self.assertEqual(plan.mode, "rebuild_root")
        # Phase 1 produced base only, so no final template is invented.
        self.assertFalse(plan.has_final)
        self.assertNotIn("final", plan.inputs)
        self.assertEqual(
            plan.inputs["base"]["test"],
            self.root / SPLIT_SUBDIR / "test_provisional_base_candidate.parquet",
        )
        self.assertEqual(
            plan.outputs["difficulty"]["test"],
            self.root / FEATURES_SUBDIR / "test_provisional_difficulty_candidate.parquet",
        )

    def test_source_for_maps_outputs_back_to_their_template(self) -> None:
        plan = B2._plan_io(
            B2.default_namespace(
                feature_contract="baseline_41", rebuild_root=self.root
            ),
            self.root / FEATURES_SUBDIR,
        )
        self.assertEqual(
            plan.source_for("train", "difficulty"), plan.inputs["base"]["train"]
        )
        self.assertIsNone(plan.source_for("train", "final"))

    def test_missing_rebuild_input_raises(self) -> None:
        (self.root / SPLIT_SUBDIR / "valid_base_candidate.parquet").unlink()
        with self.assertRaises(FileNotFoundError):
            B2._plan_io(
                B2.default_namespace(
                    feature_contract="baseline_41", rebuild_root=self.root
                ),
                self.root / FEATURES_SUBDIR,
            )

    def test_final_templates_are_all_or_nothing(self) -> None:
        with self.assertRaises(ValueError) as caught:
            B2._plan_io(
                B2.default_namespace(
                    feature_contract="baseline_41",
                    rebuild_root=self.root,
                    train_final=self.root / FEATURES_SUBDIR / "train_final_candidate.parquet",
                ),
                self.root / FEATURES_SUBDIR,
            )
        self.assertIn("all-or-nothing", str(caught.exception))

    def test_outputs_must_share_the_build_directory(self) -> None:
        with self.assertRaises(ValueError) as caught:
            B2._plan_io(
                B2.default_namespace(
                    feature_contract="baseline_41",
                    rebuild_root=self.root,
                    train_difficulty_out=self.root / "elsewhere.parquet",
                ),
                self.root / FEATURES_SUBDIR,
            )
        self.assertIn("must sit directly in the build directory", str(caught.exception))

    def test_unknown_argument_is_refused(self) -> None:
        with self.assertRaises(TypeError):
            B2.default_namespace(feature_contract="baseline_41", typo_root=".")


class B2ContractGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        contract_path = (
            MODEL_RUNS_DIR / B2.REFERENCE_RUN / "feature_contract.json"
        )
        cls.reference_features = json.loads(
            contract_path.read_text(encoding="utf-8")
        )["features"]

    def test_named_contracts_pass_and_report_the_gpa_trend_delta(self) -> None:
        for name in B2.B2_ALLOWED_CONTRACTS:
            passed, detail = B2._feature_contract_gate(name, self.reference_features)
            self.assertTrue(passed, detail)
            self.assertEqual(detail["contract"], name)
            self.assertEqual(detail["removed_versus_reference"], [])
            self.assertIn("gpa_trend_delta", detail["added_over_reference"])
            self.assertIn("gpa_trend_missing", detail["added_over_reference"])
            self.assertEqual(
                detail["feature_count"], detail["declared_count_from_name"]
            )

    def test_archived_concurrent_44_is_refused(self) -> None:
        passed, detail = B2._feature_contract_gate(
            "concurrent_44", self.reference_features
        )
        self.assertFalse(passed)
        self.assertFalse(detail["checks"]["contract_is_accepted_for_b2"])

    def test_unknown_contract_raises(self) -> None:
        with self.assertRaises(ValueError):
            B2._feature_contract_gate("baseline_39", self.reference_features)

    def test_difficulty_audit_columns_are_kept_out_of_every_contract(self) -> None:
        for name in B2.B2_ALLOWED_CONTRACTS:
            _, detail = B2._feature_contract_gate(name, self.reference_features)
            self.assertTrue(
                detail["checks"]["difficulty_audit_columns_stay_out_of_the_contract"]
            )


class GpaTrendTemplateTests(_RebuildRoot):
    def test_legacy_templates_are_the_live_generation_names(self) -> None:
        args = TREND.parse_args(["--feature-contract", "baseline_41"])
        templates = TREND._resolve_templates(args)
        self.assertEqual(sorted(templates), ["base", "final"])
        self.assertEqual(
            templates["base"]["train"], MODEL_DATA_DIR / "df_train_base.parquet"
        )
        self.assertEqual(
            templates["final"]["test"], MODEL_DATA_DIR / "df_test_final.parquet"
        )

    def test_rebuild_root_yields_base_only(self) -> None:
        args = TREND.parse_args(
            ["--feature-contract", "baseline_41", "--rebuild-root", str(self.root)]
        )
        templates = TREND._resolve_templates(args)
        self.assertEqual(list(templates), ["base"])
        self.assertEqual(
            templates["base"]["valid"],
            self.root / SPLIT_SUBDIR / "valid_base_candidate.parquet",
        )

    def test_explicit_paths_may_mix_directories(self) -> None:
        # This is what makes the pre-trend regression run expressible at all:
        # the base and final templates no longer live in one directory.
        other = self.root / "elsewhere"
        other.mkdir()
        finals = {}
        for split in SPLITS:
            path = other / f"df_{split}_final.parquet"
            path.write_bytes(b"")
            finals[split] = path
        args = TREND.parse_args(
            [
                "--feature-contract",
                "baseline_41",
                "--rebuild-root",
                str(self.root),
                *[
                    item
                    for split in SPLITS
                    for item in (f"--{split}-final", str(finals[split]))
                ],
            ]
        )
        templates = TREND._resolve_templates(args)
        self.assertEqual(templates["final"], finals)
        self.assertEqual(
            templates["base"]["train"],
            self.root / SPLIT_SUBDIR / "train_base_candidate.parquet",
        )

    def test_feature_contract_is_required(self) -> None:
        with self.assertRaises(SystemExit):
            TREND.parse_args([])


class ConcurrentPathTests(_RebuildRoot):
    def test_comparison_dir_has_no_default(self) -> None:
        self.assertIsNone(CONCURRENT.parse_args([]).comparison_dir)

    def test_legacy_templates_and_outputs_keep_the_live_names(self) -> None:
        args = CONCURRENT.parse_args([])
        templates = CONCURRENT._resolve_templates(args)
        self.assertEqual(
            templates["train_difficulty"],
            MODEL_DATA_DIR / "df_train_difficulty.parquet",
        )
        outputs = CONCURRENT._resolve_outputs(args, Path("stage"))
        self.assertEqual(
            outputs["valid_concurrent"], Path("stage") / "df_valid_concurrent.parquet"
        )
        self.assertEqual(
            outputs["test_final"], Path("stage") / "df_test_final.parquet"
        )

    def test_rebuild_reads_the_feature_stage_and_writes_the_concurrent_stage(self) -> None:
        args = CONCURRENT.parse_args(["--rebuild-root", str(self.root)])
        templates = CONCURRENT._resolve_templates(args)
        self.assertEqual(
            templates["train_final"],
            self.root / FEATURES_SUBDIR / "train_final_candidate.parquet",
        )
        outputs = CONCURRENT._resolve_outputs(args, Path("stage"))
        self.assertEqual(
            outputs["train_final"], Path("stage") / "train_final_candidate.parquet"
        )
        # The staged final output must never be the final template it reads.
        self.assertNotEqual(templates["train_final"], outputs["train_final"])

    def test_output_stage_names_do_not_collide(self) -> None:
        args = CONCURRENT.parse_args(["--rebuild-root", str(self.root)])
        outputs = CONCURRENT._resolve_outputs(args, Path("stage"))
        names = [path.name for path in outputs.values()]
        self.assertEqual(len(set(names)), len(names))

    def test_missing_rebuild_template_raises(self) -> None:
        (self.root / FEATURES_SUBDIR / "train_difficulty_candidate.parquet").unlink()
        args = CONCURRENT.parse_args(["--rebuild-root", str(self.root)])
        with self.assertRaises(FileNotFoundError):
            CONCURRENT._resolve_templates(args)

    def test_registration_roster_source_is_the_registration_time_raw_file(self) -> None:
        # Peer membership must come from registration-time CRG rows, before any
        # withdrawal or outcome filtering; --rebuild-root does not change it.
        self.assertEqual(
            CONCURRENT.parse_args([]).raw_crg.name,
            "v_crg_student_course_raw.parquet",
        )
        self.assertEqual(
            CONCURRENT.parse_args(["--rebuild-root", str(self.root)]).raw_crg,
            CONCURRENT.parse_args([]).raw_crg,
        )


if __name__ == "__main__":
    unittest.main()
