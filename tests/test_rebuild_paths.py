"""Tests for the ``2026-08_temporal_rebuild_v1`` path resolver.

Synthetic roots only: every test builds its own directory tree under the OS
temporary directory, so the suite never reads or writes project data.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import MODEL_DATA_GENERATIONS, MODEL_SPLITS  # noqa: E402
from src.rebuild_paths import (  # noqa: E402
    CONCURRENT_SUBDIR,
    FEATURES_SUBDIR,
    REBUILD_GENERATIONS,
    REBUILD_SPLITS,
    REBUILD_STAGES,
    SPLIT_SUBDIR,
    rebuild_basename,
    rebuild_dataset_path,
    rebuild_diploma_bucket_map_path,
    rebuild_generation_paths,
    rebuild_split_path,
    rebuild_version_root,
)
from src.rebuild_paths import DATASET_SUBDIR  # noqa: E402

LIVE_BASENAMES = frozenset(
    f"df_{split}_{generation}.parquet"
    for split in sorted(MODEL_SPLITS)
    for generation in sorted(MODEL_DATA_GENERATIONS)
)

PHASE1_BASE_NAMES = {
    "train": "train_base_candidate.parquet",
    "valid": "valid_base_candidate.parquet",
    "test": "test_provisional_base_candidate.parquet",
}


class _TempRoot(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="rebuild_paths_test_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "2026-08_temporal_rebuild_v1"
        (self.root / SPLIT_SUBDIR).mkdir(parents=True)


class ExplicitResolutionTests(_TempRoot):
    def test_base_generation_uses_the_phase1_filenames_under_01_split(self) -> None:
        for split, basename in PHASE1_BASE_NAMES.items():
            self.assertEqual(
                rebuild_split_path(self.root, split, "base"),
                self.root / SPLIT_SUBDIR / basename,
            )

    def test_feature_generations_live_under_03_features(self) -> None:
        expected = {
            ("train", "difficulty"): "train_difficulty_candidate.parquet",
            ("valid", "difficulty"): "valid_difficulty_candidate.parquet",
            ("test", "final"): "test_provisional_final_candidate.parquet",
        }
        for (split, generation), basename in expected.items():
            self.assertEqual(
                rebuild_split_path(self.root, split, generation),
                self.root / FEATURES_SUBDIR / basename,
            )

    def test_concurrent_stage_is_a_separate_directory(self) -> None:
        self.assertEqual(
            rebuild_split_path(self.root, "valid", "concurrent"),
            self.root / CONCURRENT_SUBDIR / "valid_concurrent_candidate.parquet",
        )
        # final exists in both stages; a builder must never read and write one path.
        pre = rebuild_split_path(self.root, "train", "final")
        post = rebuild_split_path(self.root, "train", "final", stage="concurrent")
        self.assertEqual(pre.name, post.name)
        self.assertNotEqual(pre, post)
        self.assertEqual(pre.parent, self.root / FEATURES_SUBDIR)
        self.assertEqual(post.parent, self.root / CONCURRENT_SUBDIR)

    def test_unknown_stage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rebuild_split_path(self.root, "train", "final", stage="05_later")

    def test_generation_helper_covers_all_three_splits(self) -> None:
        paths = rebuild_generation_paths(self.root, "base")
        self.assertEqual(sorted(paths), sorted(REBUILD_SPLITS))
        self.assertEqual(
            paths["test"],
            self.root / SPLIT_SUBDIR / PHASE1_BASE_NAMES["test"],
        )

    def test_dataset_stage_resolves_under_05_dataset(self) -> None:
        self.assertEqual(
            rebuild_dataset_path(self.root, "train"),
            self.root / DATASET_SUBDIR / "train_dataset_candidate.parquet",
        )
        self.assertEqual(
            rebuild_dataset_path(self.root, "test"),
            self.root / DATASET_SUBDIR / "test_provisional_dataset_candidate.parquet",
        )
        with self.assertRaises(FileNotFoundError):
            rebuild_dataset_path(self.root, "train", must_exist=True)
        with self.assertRaises(ValueError):
            rebuild_dataset_path(self.root, "holdout")

    def test_diploma_map_resolves_version_locally(self) -> None:
        self.assertEqual(
            rebuild_diploma_bucket_map_path(self.root),
            self.root / "diploma_type_bucket_map.json",
        )

    def test_split_and_generation_inputs_are_normalized_and_validated(self) -> None:
        self.assertEqual(
            rebuild_split_path(self.root, "TRAIN", "Base"),
            rebuild_split_path(self.root, "train", "base"),
        )
        with self.assertRaises(ValueError):
            rebuild_split_path(self.root, "holdout", "base")
        with self.assertRaises(ValueError):
            rebuild_split_path(self.root, "train", "raw")


class LoudRefusalTests(_TempRoot):
    def test_missing_root_raises(self) -> None:
        missing = self.root.parent / "no_such_version"
        with self.assertRaises(FileNotFoundError):
            rebuild_version_root(missing)
        with self.assertRaises(FileNotFoundError):
            rebuild_split_path(missing, "train", "base")
        with self.assertRaises(FileNotFoundError):
            rebuild_diploma_bucket_map_path(missing)

    def test_root_is_required_and_never_defaults(self) -> None:
        with self.assertRaises(TypeError):
            rebuild_split_path(split="train", generation="base")  # type: ignore[call-arg]
        with self.assertRaises(ValueError):
            rebuild_version_root(None)  # type: ignore[arg-type]

    def test_a_file_as_root_raises(self) -> None:
        as_file = self.root.parent / "not_a_dir"
        as_file.write_text("", encoding="utf-8")
        with self.assertRaises(NotADirectoryError):
            rebuild_version_root(as_file)

    def test_missing_required_artifact_raises_only_when_demanded(self) -> None:
        # Resolution alone never touches the filesystem contents.
        path = rebuild_split_path(self.root, "train", "base")
        self.assertFalse(path.exists())
        with self.assertRaises(FileNotFoundError):
            rebuild_split_path(self.root, "train", "base", must_exist=True)
        path.write_bytes(b"")
        self.assertEqual(
            rebuild_split_path(self.root, "train", "base", must_exist=True),
            path,
        )

    def test_missing_diploma_map_raises_when_demanded(self) -> None:
        with self.assertRaises(FileNotFoundError):
            rebuild_diploma_bucket_map_path(self.root, must_exist=True)


class ContainmentAndCollisionTests(_TempRoot):
    def test_no_resolved_path_escapes_the_version_root(self) -> None:
        resolved_root = self.root.resolve()
        candidates = [
            rebuild_split_path(self.root, split, generation, stage=stage)
            for split in REBUILD_SPLITS
            for generation in REBUILD_GENERATIONS
            for stage in REBUILD_STAGES
        ]
        candidates.append(rebuild_diploma_bucket_map_path(self.root))
        for path in candidates:
            # relative_to raises if the path is not inside the root.
            path.resolve().relative_to(resolved_root)
            self.assertNotIn("..", path.parts)

    def test_no_returned_basename_matches_a_live_artifact_name(self) -> None:
        for split in REBUILD_SPLITS:
            for generation in REBUILD_GENERATIONS:
                name = rebuild_basename(split, generation)
                self.assertNotIn(name, LIVE_BASENAMES)
                self.assertFalse(name.startswith("df_"))
                self.assertNotEqual(name, f"df_{split}_final.parquet")

    def test_resolver_vocabularies_match_the_live_ones(self) -> None:
        self.assertEqual(set(REBUILD_SPLITS), set(MODEL_SPLITS))
        self.assertEqual(set(REBUILD_GENERATIONS), set(MODEL_DATA_GENERATIONS))


if __name__ == "__main__":
    unittest.main()
