"""Parquet append/verify contract for the registration-roster build script.

The live difficulty/final templates are written by pandas WITH a real index, so
they carry a ``__index_level_0__`` column plus pandas schema metadata. A
pandas round-trip during the append re-derives the schema and moves that
column; these tests pin the Arrow-native behaviour instead:

  output columns == source columns (original order) + CONCURRENT_FEATURE_COLUMNS

and pin that the verifier rejects extra columns and dtype changes rather than
casting them away.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.concurrent_group_features import CONCURRENT_FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_build_module():
    """Import the build script by path: scripts/ is not a package."""
    path = PROJECT_ROOT / "scripts" / "build_concurrent_group_features.py"
    spec = importlib.util.spec_from_file_location(
        "build_concurrent_group_features", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILD = _load_build_module()


def _features(n: int) -> pd.DataFrame:
    rng = np.arange(n, dtype="float64")
    return pd.DataFrame(
        {
            "concurrent_peer_difficulty_mean": rng / 10.0,
            "concurrent_peer_difficulty_max": rng / 5.0,
            "concurrent_peer_difficulty_missing": (rng % 2).astype("int64"),
            "concurrent_peer_set_empty": (rng % 2).astype("int64"),
            "concurrent_peer_difficulty_values_missing": np.zeros(n, dtype="int64"),
            "concurrent_peer_observed_count": np.arange(n, dtype="int64"),
            "concurrent_peer_weak_ratio": rng / 100.0,
            "concurrent_peer_same_req_type_ratio": rng / 50.0,
        }
    )


class StreamAppendIndexMetadataTest(unittest.TestCase):
    ROWS = 7

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        # A source frame written WITH a non-default index, exactly like the
        # live templates: this produces __index_level_0__ + pandas metadata.
        source = pd.DataFrame(
            {
                "student_id": [f"{i}.111" for i in range(self.ROWS)],
                "part_id": np.arange(self.ROWS, dtype="int64"),
                "final_mark": np.linspace(50.0, 95.0, self.ROWS),
            },
            index=pd.Index(np.arange(100, 100 + self.ROWS), name=None),
        )
        self.source_path = self.tmp / "source.parquet"
        source.to_parquet(self.source_path, index=True)
        self.features = _features(self.ROWS)

    def _source_names(self):
        return list(pq.ParquetFile(self.source_path).schema_arrow.names)

    def test_source_fixture_really_has_an_index_column(self):
        # If this ever stops holding, the rest of the file stops testing the
        # thing it claims to test.
        self.assertIn("__index_level_0__", self._source_names())

    def test_append_puts_concurrent_columns_after_every_source_column(self):
        out = self.tmp / "out.parquet"
        BUILD._stream_append_columns(
            self.source_path, out, self.features, batch_size=3
        )

        expected = self._source_names() + list(CONCURRENT_FEATURE_COLUMNS)
        actual = list(pq.ParquetFile(out).schema_arrow.names)
        self.assertEqual(actual, expected)
        # __index_level_0__ keeps its original position, it does not drift to
        # the end behind the appended columns.
        self.assertEqual(
            actual.index("__index_level_0__"),
            self._source_names().index("__index_level_0__"),
        )

    def test_append_preserves_source_dtypes_and_values_and_the_index(self):
        out = self.tmp / "out.parquet"
        BUILD._stream_append_columns(
            self.source_path, out, self.features, batch_size=3
        )

        source_schema = pq.ParquetFile(self.source_path).schema_arrow
        output_schema = pq.ParquetFile(out).schema_arrow
        for column in source_schema.names:
            self.assertEqual(
                source_schema.field(column).type,
                output_schema.field(column).type,
                msg=f"dtype drifted for {column}",
            )

        before = pd.read_parquet(self.source_path)
        after = pd.read_parquet(out)
        # The pandas index round-trips identically, and no source column moved.
        pd.testing.assert_index_equal(before.index, after.index)
        pd.testing.assert_frame_equal(before, after[before.columns])
        for column in CONCURRENT_FEATURE_COLUMNS:
            self.assertIn(column, after.columns)
        np.testing.assert_allclose(
            after["concurrent_peer_difficulty_mean"].to_numpy(dtype=float),
            self.features["concurrent_peer_difficulty_mean"].to_numpy(),
        )

    def test_verifier_accepts_the_arrow_native_output(self):
        out = self.tmp / "out.parquet"
        BUILD._stream_append_columns(
            self.source_path, out, self.features, batch_size=3
        )
        result = BUILD._verify_streamed_output(
            self.source_path, out, self.features, "unit", "concurrent"
        )
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["schema_is_exactly_source_plus_concurrent"])
        self.assertTrue(result["pandas_index_columns_preserved"])

    def test_verifier_rejects_an_unexpected_extra_column(self):
        out = self.tmp / "extra.parquet"
        BUILD._stream_append_columns(
            self.source_path, out, self.features, batch_size=3
        )
        table = pq.read_table(out).append_column(
            "sneaked_in", pa.array(np.zeros(self.ROWS, dtype="int64"))
        )
        tampered = self.tmp / "extra_written.parquet"
        pq.write_table(table, tampered)

        with self.assertRaisesRegex(AssertionError, "unexpected"):
            BUILD._verify_streamed_output(
                self.source_path, tampered, self.features, "unit", "concurrent"
            )

    def test_verifier_rejects_a_source_dtype_change_instead_of_casting(self):
        # Reproduce the old pandas-round-trip hazard: same values, wider type.
        table = pq.read_table(self.source_path)
        casted = table.set_column(
            table.schema.get_field_index("part_id"),
            "part_id",
            table.column("part_id").cast(pa.float64()),
        )
        for column in CONCURRENT_FEATURE_COLUMNS:
            casted = casted.append_column(
                column, pa.array(self.features[column].to_numpy())
            )
        tampered = self.tmp / "dtype.parquet"
        pq.write_table(casted, tampered)

        with self.assertRaisesRegex(AssertionError, "dtype change"):
            BUILD._verify_streamed_output(
                self.source_path, tampered, self.features, "unit", "concurrent"
            )


if __name__ == "__main__":
    unittest.main()
