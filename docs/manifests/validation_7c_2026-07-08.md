# Phase 7c Rebuild — Validation Record (2026-07-08)

Baseline manifest: `baseline_7bc_2026-07-08.json`. Isolated backups: `D:/AI/data_clean_academic_advisor/backup_7bc_2026-07-08/` (13 files, sha256-verified). Executed headlessly (nbclient, kernel cwd = project root, `ACADEMIC_ADVISOR_DATA_DIR` set explicitly). Comparisons are against the pre-rebuild backups per plan §4/§4.1.

## Stage results (all PASS)

| Stage | Notebook | Validation type | Result |
|---|---|---|---|
| S1 diploma-source cleaning | `clean_v_add_academic_info.ipynb` | EXPECTED DELTA (pre-declared: `student_id` double→string only) | 32,524 rows both; column list identical; new dtype `string`; **1:1 value mapping (old normalized == new), zero ID collisions**; `diploma_gpa`/`diploma_type_id` values unchanged. |
| S2 upstream diploma merge (D1, first materialization) | `02_merge_diploma.ipynb` | LOGICAL PARITY, population-aware | Left-join row invariance: 761,347. **Match-status parity on the common population: zero lost, zero recovered; 6 unmatched students old = 6 unmatched students new** (Phase 3 evidence reproduced exactly). All in-notebook guards passed at runtime. |
| S3 select | `01_select_model_population.ipynb` | LOGICAL PARITY + survival assert | 761,346 rows; exactly +2 columns (`diploma_gpa`, `diploma_type_id`); 26/26 old columns survive with **positional value parity**; in-notebook diploma-survival assert passed. |
| S4 feature engineering | `02_feature_engineering.ipynb` | LOGICAL PARITY | 727,852 rows; column set identical (63); **positional value parity on all 63 columns including the diploma pass-through**. One DECLARED DELTA — see below. |
| S5 base splits | `01_split_diagnostics.ipynb` | LOGICAL PARITY | 450,465 / 156,097 / 110,008 rows; year boundaries 2005–2021 / 2022–2023 / 2024–2025s1; positional parity on all 63 base columns vs the old splits; split indexes internally consistent with the new FE index. |
| S6 difficulty generation | `02_course_difficulty.ipynb` | LOGICAL PARITY | +7 columns exactly (70); all 7 difficulty columns positionally identical to the old enriched splits, all three splits. |
| S7 final generation + fitted state | `03_diploma_type_bucketing.ipynb` | LOGICAL PARITY + contract 14 | +1 column exactly (71); column set identical to old final splits; `diploma_type_bucket` positionally identical; `artifacts/diploma_type_bucket_map.json` persisted — top codes [15, 16, 13, 19, 26], rare=6, unseen=−1, categories [13, 15, 16, 19, 26, 6, −1]. |

## Declared delta (explained, non-semantic)

**Index provenance.** The old `after_fet_eng.parquet` carried a contiguous RangeIndex (0..727851) — an accidental side effect of `pd.merge` inside the retired `merge_diploma.py`, which resets the index. The rebuilt `feature_engineered_primary.parquet` preserves the true selection-frame positions (non-contiguous, 0..761345). Verified both directions: old index is exactly the merge-reset RangeIndex; positional row order (student_id sequence) is element-for-element identical. All value comparisons are therefore positional, per §4B. The same provenance difference propagates to the split-generation indexes. No values, row order, row counts, or schemas are affected.

## Findings surfaced during execution

- The pre-rebuild on-disk cleaned diploma artifact had `student_id` as **float64** — the normalization code existed but the artifact predated it; `02_merge_diploma`'s dtype guard would have stopped any run against it (and evidently did: `merged_with_diploma.parquet` had never been materialized before this rebuild). Stage 1's expected delta closed this.
- Models were **NOT retrained** (out of 7c scope). The final-generation splits are value-identical (positionally) to the data the current models were trained on; Phase 8 performs the cross-pipeline confirmation.
- `course_difficulty_lookup.parquet` producer: adoption stands (user decision), implementation **deferred** — the artifact is a per-key table whose derivation (key-level fallback assignment, non-LOO stats, metadata selection) is not present in any committed code; writing it inside 7c would have required inventing data logic, violating the no-logic-change guardrail. Needs its own gated task with an expected-delta declaration against `inference.py`'s contract. The existing artifact is untouched (hash in baseline manifest).

## Post-validation promotion/retirement

Superseded originals moved to `archive/pre_7c/` after validation (hash-verified): old after_fet_eng, old select output, three bare-named splits, old features-location merge output, old typo-named cleaned + raw diploma files, old ACD cleaned artifact, plus the earlier-archived debug residues and pre-drop ADD twin. Working tree now contains only canonical-named artifacts; the bare split names are fully retired.
