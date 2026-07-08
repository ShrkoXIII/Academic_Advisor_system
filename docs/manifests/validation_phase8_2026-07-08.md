# Phase 8 — Final Cross-Pipeline Validation (2026-07-08, read-only)

Tool: `scripts/parity_check.py` (created this phase — `python -m scripts.parity_check`, requires `ACADEMIC_ADVISOR_DATA_DIR` set). Read-only: no notebook, code, or data write. **Result: 51/51 checks PASS, zero unexplained drift.**

## 1. End-to-end lineage & join integrity
`merged_add_acd_crg` (761,347) → `merged_with_diploma` (761,347, row-invariant left join, no duplicate expansion) → `selected_model_population` (761,346, diploma columns survive) → `feature_engineered_primary` (727,852, diploma columns pass through) → base splits (sum 716,570 = 727,852 − 11,282 unassigned 2025-s2, exactly as designed) → difficulty (+7 cols) → final (+1 col). Row counts identical across all three split generations per split. `diploma_type_bucket_map.json` (contract 14) present.

## 2. Split boundaries
Train 2005–2021, valid 2022–2023, test 2024–2025 with **zero** 2025-semester-2 rows in test (the intentionally-unassigned incomplete semester) — matches the locked rule in `obsidian_vault/Decisions_Log.md`.

## 3. Feature contract (`models/feature_contract.json`)
All 39 locked features present in `df_train_final`; `requirement_size_bucket` (derivation source for the ordinal) present; `requirement_type_id` categorical levels **currently derived from train == contract's locked levels** (`[1,2,3,4,5,6]`); no dropped/leakage feature reappeared; M1/M2 target definitions unchanged. The rebuilt final splits are contract-conformant without any contract edit.

## 4. Target distributions
| split | n | pass_rate (final_mark≥50) | mean | median |
|---|---|---|---|---|
| train | 450,465 | 0.8413 | 65.58 | 67.0 |
| valid | 156,097 | 0.8966 | 69.52 | 71.0 |
| test  | 110,008 | 0.9104 | 71.83 | 74.0 |

Zero nulls, zero out-of-range values, all three splits. The pass-rate drift upward from train→test is a pre-existing, expected temporal pattern (not introduced by this rebuild) — informational only, not a gate.

## 5. ID suffix integrity
`student_id` is pandas `string` dtype with a dotted university suffix on 100% of rows (0 malformed) in `merged_with_diploma`, `feature_engineered_primary`, and `df_train_final` — spot-checked across the whole chain, not just the final artifact.

## 6. Drift screen vs the Phase 7c baseline manifest
Four untouched sentinel artifacts (`merge_crg_add.parquet`, `course_difficulty_lookup.parquet`, `knn_index.pkl`, `without_outliers.parquet`) are byte-identical (sha256 match) to their pre-7c hashes — confirms 7c touched only what it declared. All seven retired bare/typo-named artifacts are confirmed absent from their old live paths (present only in `archive/pre_7c/`).

## 7. Repo-wide re-grep
Zero live hardcoded data-root paths and zero bare `df_{train,valid,test}.parquet` references remain in any notebook code cell.

## Findings NOT gated by Phase 8 (informational, carried forward)

- **Stale docstring** — `src/model_training.py` lines 16–18 (module docstring CLI example) still shows `MODEL_DATA_DIR / 'df_train.parquet'` etc. The bare files no longer exist; a literal copy-paste of the example would now fail. Not a pipeline defect (no executable code path affected — `argparse` takes `--train/--valid/--test` as free-form paths), but it is misleading and was flagged back in `docs/paths_audit.md`. Cosmetic doc fix, candidate for Phase 9 or a trivial follow-up.
- **Duplicate model artifacts** — `models/` contains both `grade_model.lgbm`/`pass_model.lgbm` (mtime 2026-06-28, no `feature_contract.json` counterpart of that vintage) and the current `m1_pass_model.lgbm`/`m2_grade_model.lgbm` (mtime 2026-06-30, matching `feature_contract.json`). Looks like a leftover pre-rename pair. Not a data-governance conflict (`MODELS_DIR` has one owner, `model_training.py`), but "clean tree" is a Phase 9 checklist item — flagged, not touched (out of scope for a read-only validation phase).
- **Guard wiring gap (carried from 7a/7c notes)** — `assert_data_root` is not yet wired into `01_train_lightgbm.ipynb` or `src/inference.py`. Still open; does not block Phase 8 since training was not rerun this cycle.
- Models were **not retrained** in 7c or 8 — the current `.lgbm` artifacts were trained before the rebuild, on data now proven (by this phase's checks) to be value-identical to the rebuilt final splits. If you want the artifacts to literally originate from the new files, that's a separate, explicitly-approved retraining step, not part of governance Phase 8.

## Verdict

**PASS.** No unexplained drift. Ready for Phase 9 (final freeze) whenever you approve it — not started per your instruction.
