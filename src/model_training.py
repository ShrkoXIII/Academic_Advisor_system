"""Train LightGBM pass/fail (M1) and final-mark (M2) models — V1 baseline.

Model naming — LOCKED (do not reverse):
    M1 = pass/fail CLASSIFIER   target = (final_mark >= 50).astype(int)
    M2 = final_mark REGRESSOR    target = final_mark   (raw 0-100, no transform)

Artifact mapping:
    persistent --run-name runs -> MODELS_DIR/runs/<timestamp>__<case>/
    no --run-name quick runs  -> MODELS_DIR/quick/latest/

V1 scope: LightGBM only. No sample weights. No scale_pos_weight. No XGBoost.

Usage
-----
python -m src.model_training

All arguments default to the canonical final-generation splits returned by
``src.paths.model_split_path`` (written by 03_diploma_type_bucketing) and
MODELS_DIR. Pass --train/--valid/--test/--out only to override. From a
notebook, call main([]) so Jupyter's own argv is not parsed.

This module never builds the parquet splits. Every invocation trains both
models from scratch; loading saved weights for prediction is handled by the
inference/analysis code instead.
"""

from __future__ import annotations

from src.feature_contracts import *   # noqa: F401,F403
from src.data_prep import *           # noqa: F401,F403
from src.train_evaluate import *      # noqa: F401,F403
from src.train_evaluate import main   # explicit: main drives the CLI

# ``import *`` never re-exports underscore-prefixed names, but existing importers
# read these off this module: tests/test_feature_contracts.py (_SHARED_PARAMS,
# _read_existing_split), scripts/r2_parity.py and the report generators
# (_effective_seed_settings), 02_results_analysis.ipynb (_THRESHOLDS).
# Reading them here works; REBINDING one on this module (mt._THRESHOLDS = ...)
# no longer reaches src.train_evaluate's own global — patch it there instead.
from src.data_prep import _read_existing_split
from src.train_evaluate import _SHARED_PARAMS, _THRESHOLDS, _effective_seed_settings

if __name__ == "__main__":
    main()
