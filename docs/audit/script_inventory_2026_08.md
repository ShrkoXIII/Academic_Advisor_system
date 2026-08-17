# Script inventory — 2026-08 temporal rebuild

Read-only investigation. No file outside this report was created, modified,
renamed, or deleted. No build, training, or writing script was executed. No git
state-changing command was run. Nothing under the TEST split (`20251`) was read:
`test_provisional_base_candidate.parquet`, `test_provisional_*_candidate.parquet`,
`df_test_base.parquet`, and `df_test_final.parquet` were never opened. Where a
script reads TEST, that is reported as a property of the script, established from
its source, not by running it.

---

## 0. Environment facts these findings depend on

Established by reading `src/paths.py` and listing directories.

| Fact | Evidence |
|---|---|
| `ACADEMIC_ADVISOR_DATA_DIR` is **not set** in this session | `echo $ACADEMIC_ADVISOR_DATA_DIR` → empty |
| Therefore `DATA_DIR` = `<repo>/data` | `src/paths.py:14` — `DATA_DIR = Path(os.environ.get("ACADEMIC_ADVISOR_DATA_DIR", PROJECT_ROOT / "data"))` |
| `MODEL_DATA_VERSIONS_DIR` = `<repo>/data/model_data/versions` | `src/paths.py:37` — `MODEL_DATA_VERSIONS_DIR = MODEL_DATA_DIR / "versions"` |
| `<repo>/data/model_data/versions/` is **empty** | `ls data/model_data/versions` → 0 entries |
| `<repo>/data/model_data/` holds only `df_train_base.parquet`, `df_valid_base.parquet`, `df_test_base.parquet`, `versions/` | directory listing |
| `<repo>/data/model_data/CURRENT_VERSION.txt` **does not exist** | `ls` → `No such file or directory` |
| `<repo>/data/artifacts/` is **empty** (no `diploma_type_bucket_map.json`) | directory listing |
| `<repo>/data/raw/v_crg_student_course_raw.parquet` **exists** | directory listing |
| `<repo>/data/preprocessed/V_ACD_DEGREE_COURSE/clean_v_acd_degree_course.parquet` **exists** | directory listing |
| `<repo>/data/features/selected_model_population.parquet` **exists** | directory listing |
| The completed `2026-08_temporal_rebuild_v1` tree exists **only** under `<repo>/data_old/model_data/versions/` | directory listing |
| `data_old/` is gitignored | `.gitignore:33` — `data_old/` |

### The rebuild version root as it exists on disk (under `data_old/`)

```
2026-08_temporal_rebuild_v1/
  00_preflight/            (5 report files)
  01_5_lineage_gate/       (3 report files)
  01_split/                train_base_candidate.parquet, valid_base_candidate.parquet,
                           test_provisional_base_candidate.parquet, split_summary.json,
                           split_row_counts.csv, + 8 more reports
  03_features/             {train,valid,test_provisional}_{difficulty,final}_candidate.parquet,
                           difficulty_state/, b2_data_report.json,
                           gpa_trend_build_report.json, gpa_trend_audit/,
                           diploma_bucket_apply_report.json, diploma_type_bucket_map.json,
                           REPORT.md
  04_concurrent/           {train,valid,test_provisional}_{concurrent,final}_candidate.parquet,
                           registration_roster_*.parquet (5 kinds x 3 splits),
                           phase7_registration_roster_report.json, SHA256SUMS.txt
  05_dataset/              {train,valid,test_provisional}_dataset_candidate.parquet,
                           feature_manifest.csv, phase3_dataset_report.json
  diploma_type_bucket_map.json
```

Every stage of the five-script chain has already produced its outputs in that
tree.

### Row counts

Phase-1 split (`01_split/split_summary.json`, `01_split/split_row_counts.csv`):

| split | row_count | min_part_id | max_part_id |
|---|---:|---:|---:|
| train | 606,562 | 20051 | 20233 |
| valid | 75,380 | 20241 | 20243 |
| test | 34,628 | 20251 | 20251 |

New base layer (`data/model_data/df_*_base.parquet`, parquet metadata only,
TEST not opened):

| file | rows | columns |
|---|---:|---:|
| `df_train_base.parquet` | 606,563 | 66 |
| `df_valid_base.parquet` | 75,383 | 66 |
| `df_test_base.parquet` | not read | not read |

These two row counts are **not equal**: train differs by +1 (606,563 vs 606,562),
valid by +3 (75,383 vs 75,380). Stated as a measured fact; the cause was not
investigated.

### The 9+1 columns in the new base layer

Parquet schema check of `df_train_base.parquet` and `df_valid_base.parquet`
(TEST not opened). Both files give identical results:

- **ABSENT (all 10):** `course_pass_rate_historical`, `course_avg_mark_historical`,
  `course_retake_rate_historical`, `course_history_count`,
  `course_difficulty_missing`, `concurrent_peer_difficulty_mean`,
  `concurrent_peer_difficulty_max`, `concurrent_peer_difficulty_missing`,
  `diploma_type_bucket`, `difficulty_fallback_level`
- **PRESENT:** `gpa_trend_delta`, `gpa_trend_missing`, `degree_course_key`,
  `diploma_gpa`, `part_year`, `requirement_size_bucket`

`SEGMENT_ONLY_COLUMNS = ["difficulty_fallback_level"]` — `src/feature_contracts.py:187`.

---

# PART A — BUILD SCRIPTS

## A.1 `scripts/build_b2_temporal_course_stats.py`

**1. How it is invoked.**
CLI via argparse (`parse_args`, lines 841–902); `if __name__ == "__main__": main()`
at lines 940–941. Also importable in-process: `build()`, `main()`, and
`default_namespace()` (lines 911–937), whose docstring records the in-process
caller: *"In-process callers (the GPA-trend wrapper) construct B2 arguments
directly."*

Arguments: `--input-dir` (default `str(MODEL_DATA_DIR)`), `--output-root`
(default `str(MODEL_DATA_VERSIONS_DIR)`), `--build-id`, `--min-support` (20),
`--shrinkage-k` (20.0), `--reference-run` (default `REFERENCE_RUN`),
**`--feature-contract` (REQUIRED**, `choices=list(B2_ALLOWED_CONTRACTS)` =
`baseline_41`, `concurrent_43`), `--rebuild-root`, `--output-dir`, and per split
`--{split}-base`, `--{split}-final`, `--{split}-difficulty-out`,
`--{split}-final-out`.

No env var is read by the script itself. `ACADEMIC_ADVISOR_DATA_DIR` reaches it
indirectly through `src.paths` (see section 0).

**2. Every input path it reads.**

| Literal expression | Line | Resolves to |
|---|---|---|
| `rebuild_split_path(rebuild_root, split, "base", must_exist=True)` | 276 | NEW split data — `<root>/01_split/{train,valid,test_provisional}_base_candidate.parquet` |
| `model_split_path(split, "base", input_dir)` | 278 | legacy naming — `<input_dir>/df_{split}_base.parquet`; with defaults this is `data/model_data/df_{split}_base.parquet` (the new base layer) |
| `model_split_path(split, "final", input_dir)` | 308 | legacy naming — `<input_dir>/df_{split}_final.parquet`. Under `--rebuild-root` the `final` inputs are `{}` (line 305), never auto-resolved |
| `MODEL_RUNS_DIR / args.reference_run / "feature_contract.json"` | 743 | `models/runs/2026-07-16_1025__new-difficulty-logic/feature_contract.json` — in-repo, exists |

Columns read from each base file: `DIFFICULTY_INPUT_COLUMNS` (lines 86–96:
`part_id`, `part_year`, `final_mark`, `attempt_number`, `degree_course_key`,
`degree_id`, `faculty_id`, `requirement_type_id`, `course_credits`) plus
`diploma_gpa` for the isolation check (lines 663–668).

It reads all three splits including TEST — `SPLITS = ("train", "valid", "test")`
(line 70), loop at line 691 `for split in ("valid", "test")`.

**3. Every output path it writes, and which of the 9+1 columns.**

| Output | Expression | Line |
|---|---|---|
| difficulty generation, 3 splits | `rebuild_split_path(rebuild_root, split, "difficulty").name` under `output_dir`, or `model_split_path(split, "difficulty", input_dir).name` | 285 / 287 |
| final generation, 3 splits | only when `final` templates were supplied | 314–323 |
| `difficulty_state/` | `save_difficulty_state(full_train_state, staging_dir / "difficulty_state", ...)` | 676 |
| `b2_data_report.json` | `staging_dir / "b2_data_report.json"` | 832 |
| `REPORT.md` | `staging_dir / "REPORT.md"` | 836 |

Columns written: `DIFFICULTY_OUTPUT_COLUMNS` (line 195), which is
`STAT_OUTPUT_COLUMNS + AUDIT_OUTPUT_COLUMNS` (`src/course_difficulty.py:46`) =
9 columns:
`course_pass_rate_historical`, `course_avg_mark_historical`,
`course_retake_rate_historical`, `course_history_count`,
`difficulty_group_support_count`, `difficulty_fallback_level`, `course_is_new`,
`course_low_support`, `course_difficulty_missing`.

Of the missing set this produces **5 of the 9** (the whole difficulty group) plus
**`difficulty_fallback_level`** (the +1).

**4. Hard dependencies among the five.**
**None.** It reads the `base` generation, which none of the other four produces.
Its `base` input under `--rebuild-root` comes from `01_split/`, written by
`scripts/rebuild_2026_08_phase1_split.py`, which is outside this five-script set
(named in `rebuild_2026_08_phase3_assemble.py:95` as the builder module for
`split_assignment`, `exclusion_reason`, `pipeline_version`,
`test_provisional_20251_only`).

Recorded fact about how it actually ran: `03_features/b2_data_report.json` gives
`"io_plan": {"mode": "rebuild_root", "final_generation_built": false}` and base
inputs of
`data\model_data\versions\gpa_trend_build_nqgkdg1x\{train,valid,test_provisional}_base_candidate.parquet`
— a temporary directory, not `01_split/`. That is
`scripts/build_gpa_trend_dataset.py` calling `build_b2(...)` in-process
(`docs/map/03-rebuild-2026-08.md:110` — *"`build_gpa_trend_dataset.py:234` calls
`build_b2(...)`"*).

**5. Hardcoded assumptions now stale on the new split.**

- `REFERENCE_RUN = "2026-07-16_1025__new-difficulty-logic"` (line 71) — a run
  directory that exists in `models/runs/`. Used at line 743.
- **No semester list, no split boundary, no expected row count** is hardcoded.
  The time contract is derived: `part_numeric = pd.to_numeric(train_base["part_id"], errors="raise")`
  (line 628), `"n_train_semesters": int(part_numeric.nunique())` (line 786).
- **No hardcoded feature-count constant.** The gate reads the count off the named
  contract. Its docstring (lines 365–373) records the predecessor: *"The
  predecessor of this gate asserted ``EXPECTED_FEATURE_COUNT == len(MODEL_FEATURES)
  == 41``. Those deprecated globals now alias ``concurrent_44``, so the clause is
  unsatisfiable and B2 could not run at all."*
- `B2_ALLOWED_CONTRACTS = ("baseline_41", "concurrent_43")` (line 75) —
  `concurrent_44` deliberately excluded.
- `DIFFICULTY_AUDIT_ONLY_COLUMNS` (lines 78–82) and
  `GPA_TREND_FEATURES = ("gpa_trend_delta", "gpa_trend_missing")` (line 84).

**6. Guards or assertions that will fail on the new split.**

Against the completed rebuild root, the output directory already exists:

```python
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing B2 output: {output_dir}")
```
(lines 587–588; under `--rebuild-root` `output_dir` is
`rebuild_split_path(args.rebuild_root, "train", "difficulty").parent` = `<root>/03_features`, lines 579–582)

```python
    if staging_dir.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete B2 output: {staging_dir}")
```
(lines 590–591)

```python
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite output parquet: {output_path}")
```
(lines 172–173)

Data gates that raise on content (lines 766–769):
```python
    if failed:
        raise AssertionError(
            f"B2 data gates failed; incomplete output retained at {staging_dir}: {failed}"
        )
```
The eight gates are listed at lines 752–764, including
`"first train semester is no-history Level 6"`,
`"train exercises more than one fallback level"`,
`"valid/test statistics contain no nulls"`, and
`"diploma_gpa values are byte-for-value unchanged"`.

Alignment guard (lines 156–160):
```python
    if not batch_frame.index.equals(pd.RangeIndex(len(batch_frame))):
        raise AssertionError(
            f"Unindexed source {source_path.name} returned a non-positional batch "
            f"index at rows {offset}:{end}; refusing to align by position"
        )
```

---

## A.2 `scripts/rebuild_2026_08_fit_diploma_bucket_map.py`

**1. How it is invoked.**
CLI via argparse (lines 109–123); `if __name__ == "__main__": main()` at lines
138–139. Two arguments only:
- `--rebuild-root`, `default=MODEL_DATA_VERSIONS_DIR / REBUILD_VERSION` (line 114)
- `--expected-train-rows`, `default=EXPECTED_TRAIN_ROWS` (line 120), *"0 disables the check"*

No env var read directly.

**2. Every input path it reads.**

| Literal expression | Line | Resolves to |
|---|---|---|
| `rebuild_generation_paths(root, "base", must_exist=True)` | 57 | NEW split data — `<root>/01_split/{train,valid,test_provisional}_base_candidate.parquet` |

With the **default** root this is
`data/model_data/versions/2026-08_temporal_rebuild_v1/01_split/...`, a directory
that **does not exist** (section 0). The populated copy is under `data_old/`.

`rebuild_generation_paths` iterates all three splits
(`src/rebuild_paths.py:189–194`, `for split in REBUILD_SPLITS`), so this script
**opens the TEST file**. It reads one column:
`pd.read_parquet(path, columns=[DIPLOMA_TYPE_COLUMN])` (line 61). The docstring
states the reason (lines 16–18): *"TEST is read for one purpose alone: the
reserved-label collision check the rule requires across all three splits, which
reads ``diploma_type_id`` and nothing else."*

**3. Every output path it writes, and which of the 9+1 columns.**

| Output | Expression | Line |
|---|---|---|
| version-local bucket map | `rebuild_diploma_bucket_map_path(root)` → `<root>/diploma_type_bucket_map.json` | 58, written at 98 |

**Zero of the 9+1 columns.** This script writes no dataset column. Its own
docstring, line 15: *"This script only fits and persists. It builds no feature,
writes no split, trains nothing, and never opens the live map for writing."*
Confirmed at line 105: `print(f"Live map  : {DIPLOMA_TYPE_BUCKET_MAP_PATH} (not opened for writing)")`.

**4. Hard dependencies among the five.**
**None.** It reads only the `base` generation from `01_split/`, which none of the
other four writes.

**5. Hardcoded assumptions now stale on the new split.**

```python
EXPECTED_TRAIN_ROWS = 606_562  # Amendment 3, verified against 01_split/split_summary.json
```
(line 51)

This value **matches** the new split's `01_split` train
(`split_summary.json` → `splits.train.row_count = 606562`). It is **not stale for
that input**. It does **not** match `data/model_data/df_train_base.parquet`
(606,563 rows) — a file this script never reads.

Also hardcoded: `FIT_NOTE` (line 52), and `REBUILD_VERSION` /
`REBUILD_SPLITS` imported from `src.rebuild_paths`. No semester list, no split
boundary, no expected column count, no feature-count constant.

**6. Guards or assertions that will fail on the new split.**

```python
    if args.expected_train_rows and len(train) != args.expected_train_rows:
        raise AssertionError(
            f"TRAIN has {len(train):,} rows; Amendment 3 records "
            f"{args.expected_train_rows:,}. Refusing to fit on an unexpected split."
        )
```
(lines 69–73) — fires against any train frame that is not 606,562 rows.

```python
    assert_no_reserved_label_collision(frames)
```
(line 66) — raises inside `src/diploma_bucketing.py` on a real code colliding with
a reserved label, evaluated across all three splits.

With the default root, the run stops earlier, inside `src/rebuild_paths.py:136–139`:
```python
    if not resolved.exists():
        raise FileNotFoundError(
            f"Rebuild version root does not exist: {resolved}"
        )
```

---

## A.3 `scripts/rebuild_2026_08_phase3_diploma_bucket_apply.py`

**1. How it is invoked.**
CLI via argparse (lines 240–253); `if __name__ == "__main__": main()` at lines
261–262. Arguments:
- `--rebuild-root`, `default=MODEL_DATA_VERSIONS_DIR / REBUILD_VERSION` (line 245)
- `--diploma-map-path`, help *"Override the version-local map. Never defaults to the live map."* (line 251)

No env var read directly.

**2. Every input path it reads.**

| Literal expression | Line | Resolves to |
|---|---|---|
| `rebuild_diploma_bucket_map_path(root, must_exist=True)` | 164 | `<root>/diploma_type_bucket_map.json` — the version-local map written by A.2 |
| `rebuild_split_path(root, split, "difficulty", must_exist=True)` | 169 | NEW split data — `<root>/03_features/{train,valid,test_provisional}_difficulty_candidate.parquet` |

All three splits (`for split in REBUILD_SPLITS`, lines 170–171), so it **opens the
TEST difficulty file**. Column read for mapping:
`pd.read_parquet(path, columns=[DIPLOMA_TYPE_COLUMN])` (line 180); the full table
is then streamed through Arrow-native.

The live map is never read — docstring lines 12–13: *"the live map at
``data/artifacts/diploma_type_bucket_map.json`` is never read."*

**3. Every output path it writes, and which of the 9+1 columns.**

| Output | Expression | Line |
|---|---|---|
| final generation, 3 splits | `rebuild_split_path(root, split, "final")` → `<root>/03_features/*_final_candidate.parquet` (stage defaults to `features`, `src/rebuild_paths.py:87`) | 173 |
| apply report | `outputs["train"].parent / "diploma_bucket_apply_report.json"` | 232 |

Columns added: exactly one, `DIPLOMA_BUCKET_COLUMN` (line 108,
`"columns_added": [DIPLOMA_BUCKET_COLUMN]`) = **`diploma_type_bucket`** — **1 of
the 9** missing columns. `diploma_type_id` is preserved untouched and audit-only
(line 226).

**4. Hard dependencies among the five.**

| Must run before | Evidence line |
|---|---|
| `build_b2_temporal_course_stats.py` | line 169 — `rebuild_split_path(root, split, "difficulty", must_exist=True)` |
| `rebuild_2026_08_fit_diploma_bucket_map.py` | line 164 — `rebuild_diploma_bucket_map_path(root, must_exist=True)` |

Both use `must_exist=True`, which raises `FileNotFoundError` in
`src/rebuild_paths.py:174–177` when the artifact is absent.

**5. Hardcoded assumptions now stale on the new split.**
`TEST_PROVISIONAL_COLUMN = "test_provisional_20251_only"` (line 47) — a column
name, checked for presence at line 196 and reported, never asserted here. No
semester list, no split boundary, no expected row count, no expected column
count, no feature-count constant.

**6. Guards or assertions that will fail on the new split.**

```python
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite final generation: {existing}")
```
(lines 175–177) — fires against the completed rebuild root, where all three
`*_final_candidate.parquet` already exist in `03_features/`.

```python
    if DIPLOMA_BUCKET_COLUMN in source_schema.names:
        raise AssertionError(
            f"Template already contains {DIPLOMA_BUCKET_COLUMN}: {source_path}"
        )
```
(lines 67–70)

```python
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_path}")
```
(lines 63–64)

Readback assertions (lines 121–147): row count changed, unexpected column change,
template column order changed, bucket column contains nulls, bucket values changed
on readback, bucket values outside the fitted categories, audit-only
`diploma_type_id` changed.

---

## A.4 `scripts/build_concurrent_group_features.py`

**1. How it is invoked.**
CLI via argparse (lines 2021–2114); `if __name__ == "__main__": main()` at lines
2126–2127. **No argument is required.** Arguments: `--template-dir`
(`default=MODEL_DATA_DIR`), `--output-root` (`default=MODEL_DATA_VERSIONS_DIR`),
`--build-id`, `--raw-crg` (`default=RAW_CRG_PATH`), `--acd-metadata`
(`default=ACD_METADATA_PATH`), `--current-version-file`
(`default=CURRENT_VERSION_FILE`), `--difficulty-state-dir`, `--comparison-dir`,
`--rebuild-root`, `--output-dir`, and per split `--{split}-difficulty`,
`--{split}-final`, `--{split}-concurrent-out`, `--{split}-final-out`.

No env var read directly.

**2. Every input path it reads.**

| Literal expression | Line | Resolves to |
|---|---|---|
| `rebuild_split_path(rebuild_root, split, generation, stage="features", must_exist=True)` — for `generation in ("difficulty", "final")` | 1480–1486 | NEW split data — `<root>/03_features/*_{difficulty,final}_candidate.parquet` |
| `model_split_path(split, generation, template_dir)` | 1488 | legacy naming — `<template_dir>/df_{split}_{difficulty,final}.parquet`; with defaults, `data/model_data/` (which holds no `difficulty`/`final` file today) |
| `RAW_CRG_PATH = RAW_DIR / "v_crg_student_course_raw.parquet"` | 91 | `data/raw/v_crg_student_course_raw.parquet` — **exists**. Not redirected by `--rebuild-root` |
| `ACD_METADATA_PATH = PREPROCESSED_DIR / "V_ACD_DEGREE_COURSE" / "clean_v_acd_degree_course.parquet"` | 92–96 | `data/preprocessed/V_ACD_DEGREE_COURSE/clean_v_acd_degree_course.parquet` — **exists** |
| `CURRENT_VERSION_FILE = MODEL_DATA_DIR / "CURRENT_VERSION.txt"` | 90 | `data/model_data/CURRENT_VERSION.txt` — **does not exist** |
| `output_root / promoted_build_id / "difficulty_state"` | 1533 | a legacy versioned directory named by `CURRENT_VERSION.txt`, unless `--difficulty-state-dir` overrides |
| `model_split_path(split, "concurrent", comparison_dir)` | 1539 | optional legacy versioned build, only when `--comparison-dir` given |

The docstring is explicit that the raw source is not redirected (lines 14–17):
*"Peer membership always comes from the registration-time CRG roster in
``data/raw/v_crg_student_course_raw.parquet``, before any withdrawal or outcome
filtering; ``--rebuild-root`` does not change that source."*

`SPLITS = ("train", "valid", "test")` (line 86), main loop `for split in SPLITS`
(line 1602) — reads TEST.

**3. Every output path it writes, and which of the 9+1 columns.**

| Output | Expression | Line |
|---|---|---|
| concurrent generation, 3 splits | `rebuild_split_path(rebuild_root, split, generation, stage="concurrent").name` under staging, else `model_split_path(split, generation, staging).name` | 1506–1510 |
| final generation, 3 splits | same resolver, `generation="final"`, `stage="concurrent"` | 1501–1510 |
| `registration_roster_{split}.parquet` | `staging / f"registration_roster_{split}.parquet"` | 1725 |
| `registration_roster_row_audit_{split}.parquet` | line 1727 | |
| `registration_roster_count_comparison_{split}.parquet` | line 1730 | |
| `registration_roster_mismatch_examples_{split}.parquet` | line 1733 | |
| `registration_roster_excluded_status_{split}.parquet` | line 1736 | |
| `phase7_registration_roster_report.json` | `staging / "phase7_registration_roster_report.json"` | 1982 |
| `PHASE7_REGISTRATION_ROSTER_REPORT.md` | line 1983 | |
| `SHA256SUMS.txt` | `staging / "SHA256SUMS.txt"` | 2003 |

Columns appended: `CONCURRENT_FEATURE_COLUMNS` (line 375/394) =
`MODEL_CONCURRENT_FEATURES + AUDIT_CONCURRENT_FEATURES` (8 columns).
`MODEL_CONCURRENT_FEATURES` = `concurrent_peer_difficulty_mean`,
`concurrent_peer_difficulty_max`, `concurrent_peer_difficulty_missing`.

Of the missing set this produces **3 of the 9** (the whole concurrent group), plus
5 audit-only columns (`concurrent_peer_set_empty`,
`concurrent_peer_difficulty_values_missing`, `concurrent_peer_observed_count`,
`concurrent_peer_weak_ratio`, `concurrent_peer_same_req_type_ratio`).

**4. Hard dependencies among the five.**

| Must run before | Evidence line |
|---|---|
| `rebuild_2026_08_phase3_diploma_bucket_apply.py` | lines 1473–1486 — `for generation in ("difficulty", "final")` … `rebuild_split_path(..., stage="features", must_exist=True)`. The `final` at stage `features` is written only by the diploma-apply step (A.3 field 3). |
| `build_b2_temporal_course_stats.py` | same loop, `generation="difficulty"`; and `load_difficulty_state(difficulty_state_dir)` at line 1591 consumes `difficulty_state/`, written by B2 (A.1 field 3) |

**5. Hardcoded assumptions now stale on the new split.**

```python
EXPECTED_LEGACY_MODEL_POSITION = 35  # zero-based; pinned by the concurrent_44 contract.
```
(line 106) — a hardcoded column-position constant, pinned to the **archived**
`concurrent_44` contract.

```python
PRIOR_BUILD_ID = "2026-07-23_160509__concurrent_group_feature"
PRIOR_BUILD_DIR = MODEL_DATA_VERSIONS_DIR / PRIOR_BUILD_ID
```
(lines 97–98) — a legacy versioned directory name.

```python
APPROXIMATE_PRIOR_VALUES = {
    "peer_difficulty_mean": {
        "train": 0.186,
        "valid": 0.134,
        "test": 0.135,
    },
    "maximum_peer_count": {
        "train": 14,
        "valid": 7,
        "test": 7,
    },
}
```
(lines 154–165) — per-split reference values measured on the previous split.
Used only in the Markdown report (lines 1343–1344) and only when
`--comparison-dir` is supplied.

Hardcoded feature-count constants inside `_assert_contract` (lines 292–297):
```python
        "expected_feature_count_is_44": len(CONCURRENT_44_FEATURES) == 44,
        "model_feature_count_is_44": len(CONCURRENT_44_FEATURES) == 44,
        "legacy_indicator_in_model_features": legacy in CONCURRENT_44_FEATURES,
        "legacy_indicator_position_preserved": (
            legacy in CONCURRENT_44_FEATURES
            and CONCURRENT_44_FEATURES.index(legacy) == EXPECTED_LEGACY_MODEL_POSITION
        ),
```

Also `MATERIAL_MEAN_SHIFT_THRESHOLD = 0.02` and
`MATERIALLY_UNCHANGED_GAP_RATIO = 0.80` (lines 107–108), both report-only.

No semester list, no split boundary, no expected row count.

**6. Guards or assertions that will fail on the new split.**

```python
    if not path.is_file():
        raise FileNotFoundError(f"Current-version marker not found: {path}")
```
(lines 218–219, `_read_current_version`, called at line 1529) — fires today:
`data/model_data/CURRENT_VERSION.txt` does not exist.

```python
    assert_data_root(
        *template_paths.values(),
        *prior_paths.values(),
        raw_crg_path,
        acd_metadata_path,
        current_version_file,
        difficulty_state_dir / "manifest.json",
    )
```
(lines 1545–1552), which raises in `src/paths.py:100–101`:
```python
        if not required.exists() or required.stat().st_size == 0:
            raise RuntimeError(f"Required data artifact missing or empty: {required}")
```

```python
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise AssertionError(f"44-feature contract gates failed: {failed}")
```
(lines 311–313)

```python
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite versioned dataset: {output_dir}"
        )
```
(lines 1566–1569; under `--rebuild-root`, `output_dir` is
`rebuild_split_path(rebuild_root, "train", "concurrent", stage="concurrent").parent`
= `<root>/04_concurrent`, lines 1559–1562)

```python
    if staging.exists():
        raise FileExistsError(
            f"Refusing to overwrite incomplete build: {staging}"
        )
```
(lines 1570–1573)

Content assertions that raise: template already contains a concurrent column
(lines 375–380); output schema is not source + concurrent columns (lines 447–456);
dtype change for a template column (lines 461–465); outcome columns entered the
roster (lines 1614–1619); legacy indicator is not the empty peer-set flag
(lines 1683–1690); persisted legacy indicator dtype not `int64` (lines 1771–1775);
live model-data files changed (lines 1862–1868); same-semester target difficulty
conflicts within the full registration context (lines 712–717).

---

## A.5 `scripts/rebuild_2026_08_phase3_assemble.py`

**1. How it is invoked.**
CLI via argparse (lines 448–456); `if __name__ == "__main__": main()` at lines
467–468. One argument:
- `--rebuild-root`, `default=MODEL_DATA_VERSIONS_DIR / REBUILD_VERSION` (line 453),
  help *"Version root holding 04_concurrent; 05_dataset is written under it."*

No env var read directly.

**2. Every input path it reads.**

| Literal expression | Line | Resolves to |
|---|---|---|
| `rebuild_split_path(root, split, "final", stage="concurrent", must_exist=True)` | 348–350 | NEW split data — `<root>/04_concurrent/{train,valid,test_provisional}_final_candidate.parquet` |

All three splits (`for split in REBUILD_SPLITS`, line 351), so it **opens the TEST
concurrent-stage final file**.

With the **default** root this is
`data/model_data/versions/2026-08_temporal_rebuild_v1/04_concurrent/...`, which
does not exist (section 0).

**3. Every output path it writes, and which of the 9+1 columns.**

| Output | Expression | Line |
|---|---|---|
| dataset, 3 splits | `rebuild_dataset_path(root, split)` → `<root>/05_dataset/*_dataset_candidate.parquet` | 353 |
| `feature_manifest.csv` | `outputs["train"].parent / "feature_manifest.csv"` | 375 |
| `phase3_dataset_report.json` | `outputs["train"].parent / "phase3_dataset_report.json"` | 439 |

It creates no new feature column; it selects and drops. It **drops exactly one**:
```python
DEAD_FEATURE = "concurrent_peer_difficulty_missing"
```
(line 74; `keep = [name for name in source_names if name != DEAD_FEATURE]`, line 207)

So its outputs carry **8 of the 9** missing columns (all except
`concurrent_peer_difficulty_missing`) and **`difficulty_fallback_level`** (the +1),
which the manifest labels `segment_only` via `SEGMENT_ONLY_COLUMNS` (lines 164–165).

Recorded fact from `05_dataset/phase3_dataset_report.json`: train 606,562 rows
86→85 columns; valid 75,380 rows 86→85; test 34,628 rows 87→86.

**4. Hard dependencies among the five.**

| Must run before | Evidence line |
|---|---|
| `build_concurrent_group_features.py` | lines 348–350 — `rebuild_split_path(root, split, "final", stage="concurrent", must_exist=True)`; the `concurrent` stage is written only by the concurrent builder (A.4 field 3) |

Transitively it also requires A.1, A.2, A.3, because `_assemble_split` asserts the
difficulty contract and the union contract are complete (field 6 below).

**5. Hardcoded assumptions now stale on the new split.**

- `M1_CONTRACT = "baseline_41"`, `M2_CONTRACT = "concurrent_43"` (lines 71–72)
- `DEAD_FEATURE = "concurrent_peer_difficulty_missing"` (line 74)
- `TEST_PROVISIONAL_COLUMN = "test_provisional_20251_only"` (line 75)
- `LINEAGE_APPLIED = False`, `LINEAGE_REASON = "NOT_AUTHORISED_BY_OWNER_DESPITE_PROCEED"` (lines 77–78)
- `IDENTITY_COLUMNS = ["degree_id", "course_id"]` (line 79)
- `BUILDER_MODULES` (lines 89–101), which names
  `scripts/rebuild_2026_08_phase1_split.py` and `scripts/audit_gpa_trend.py`

**No feature-count constant is hardcoded.** The union is computed:
`union = list(m2.features)` (line 120) after asserting `m1 ⊆ m2`. The docstring's
*"43 features, of which 42 are read from the parquet"* (lines 9–10) is prose;
lines 10–11 state *"Both are checked here rather than assumed."*

No semester list, no split boundary, no expected row count.

**6. Guards or assertions that will fail on the new split.**

```python
    existing = [str(p) for p in outputs.values() if p.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite dataset files: {existing}")
```
(lines 354–356) — fires against the completed rebuild root, where all three
`*_dataset_candidate.parquet` already exist in `05_dataset/`.

```python
    if DEAD_FEATURE not in source_names:
        raise AssertionError(
            f"{source_path.name}: {DEAD_FEATURE} absent from the concurrent "
            "output; the builder contract changed and this drop is now wrong"
        )
```
(lines 202–206)

```python
    if missing:
        raise AssertionError(f"{split}: union contract columns not built: {missing}")
```
(lines 214–215)

```python
    missing_difficulty = [c for c in DIFFICULTY_CONTRACT if c not in keep]
    if missing_difficulty:
        raise AssertionError(f"{split}: difficulty contract incomplete: {missing_difficulty}")
```
(lines 224–226)

```python
    if is_test and TEST_PROVISIONAL_COLUMN not in keep:
        raise AssertionError(
            f"{split}: {TEST_PROVISIONAL_COLUMN} column missing from a provisional split"
        )
```
(lines 229–232)

```python
    if not set(m1.features).issubset(m2.features):
        raise AssertionError(
            f"{M1_CONTRACT} is not a subset of {M2_CONTRACT}; the union is not "
            "the larger contract and this assembly's assumption is void"
        )
```
(lines 115–119)

Also: derivation source missing (lines 218–221); target missing (line 223);
columns absent from the manifest (line 396); manifest contract membership
disagrees with the contracts (line 401); readback assertions at lines 288–326.

---

## A.6 Dependency-ordered run sequence

Derived from the `must_exist=True` evidence lines in field 4 of each script.

```
1.  scripts/build_b2_temporal_course_stats.py
        reads   01_split/*_base_candidate.parquet
        writes  03_features/*_difficulty_candidate.parquet
                03_features/difficulty_state/
        gives   5 of 9 (difficulty group) + difficulty_fallback_level

2.  scripts/rebuild_2026_08_fit_diploma_bucket_map.py
        reads   01_split/*_base_candidate.parquet   (all 3 splits, diploma_type_id only)
        writes  <root>/diploma_type_bucket_map.json
        gives   0 dataset columns
        NOTE: depends on no other script in this set. It needs only 01_split,
              so it may equally run at position 1. It is placed here because
              step 3 is the first step that consumes its output.

3.  scripts/rebuild_2026_08_phase3_diploma_bucket_apply.py
        requires 1 (03_features difficulty) and 2 (the map)
        reads   03_features/*_difficulty_candidate.parquet
                <root>/diploma_type_bucket_map.json
        writes  03_features/*_final_candidate.parquet
        gives   diploma_type_bucket  (1 of 9)

4.  scripts/build_concurrent_group_features.py
        requires 3 (needs BOTH difficulty and final at stage="features")
        reads   03_features/*_{difficulty,final}_candidate.parquet
                data/raw/v_crg_student_course_raw.parquet
                data/preprocessed/V_ACD_DEGREE_COURSE/clean_v_acd_degree_course.parquet
                data/model_data/CURRENT_VERSION.txt
                <difficulty_state>/manifest.json
        writes  04_concurrent/*_{concurrent,final}_candidate.parquet
                04_concurrent/registration_roster_*.parquet
        gives   3 of 9 (concurrent group) + 5 audit columns

5.  scripts/rebuild_2026_08_phase3_assemble.py
        requires 4 (stage="concurrent" final)
        reads   04_concurrent/*_final_candidate.parquet
        writes  05_dataset/*_dataset_candidate.parquet
                05_dataset/feature_manifest.csv
        gives   8 of 9 persisted (drops concurrent_peer_difficulty_missing)
                + difficulty_fallback_level as segment_only
```

Not one of the five, but interposed in the recorded run: `03_features/b2_data_report.json`
shows B2's base inputs came from `versions/gpa_trend_build_nqgkdg1x/`, a temporary
directory, meaning B2 was driven in-process by
`scripts/build_gpa_trend_dataset.py` rather than run standalone
(`docs/map/03-rebuild-2026-08.md:110`).

## A.7 BLOCKERS

Anything that stops the sequence running start-to-finish. There are blockers;
they are listed below.

**B-1. The default rebuild root does not exist.**
Three of the five scripts default `--rebuild-root` to
`MODEL_DATA_VERSIONS_DIR / REBUILD_VERSION`
(`rebuild_2026_08_fit_diploma_bucket_map.py:114`,
`rebuild_2026_08_phase3_diploma_bucket_apply.py:245`,
`rebuild_2026_08_phase3_assemble.py:453`). With `ACADEMIC_ADVISOR_DATA_DIR`
unset that is `<repo>/data/model_data/versions/2026-08_temporal_rebuild_v1`, and
`<repo>/data/model_data/versions/` is empty. `src/rebuild_paths.py:136–139` raises
`FileNotFoundError: Rebuild version root does not exist: …`. The populated tree is
at `<repo>/data_old/model_data/versions/2026-08_temporal_rebuild_v1`, which is
gitignored (`.gitignore:33`).

**B-2. Every output of the chain already exists in the only populated root.**
`01_split/`, `03_features/`, `04_concurrent/`, and `05_dataset/` are all fully
populated under `data_old/`. Each of the four writing scripts refuses to overwrite:
`build_b2_temporal_course_stats.py:587–588`,
`rebuild_2026_08_phase3_diploma_bucket_apply.py:175–177`,
`build_concurrent_group_features.py:1566–1569`,
`rebuild_2026_08_phase3_assemble.py:354–356`.
Pointed at that root the sequence stops at step 1 with `FileExistsError`.

**B-3. `CURRENT_VERSION.txt` is absent from the new data root.**
`build_concurrent_group_features.py:90` resolves
`CURRENT_VERSION_FILE = MODEL_DATA_DIR / "CURRENT_VERSION.txt"` =
`data/model_data/CURRENT_VERSION.txt`, which does not exist.
`_read_current_version` raises `FileNotFoundError: Current-version marker not
found` (lines 218–219) before any other work. `--rebuild-root` does not redirect
it; only `--current-version-file` does. The file exists at
`data_old/model_data/CURRENT_VERSION.txt` and contains
`2026-07-21_gpa_trend_feature`.

**B-4. The promoted `difficulty_state` cannot be resolved from the new data root.**
`build_concurrent_group_features.py:1530–1534` computes
`output_root / promoted_build_id / "difficulty_state"`. With
`--output-root` at its default and `promoted_build_id` read from B-3's missing
file, the path cannot be formed. Even given the marker's value, the target
`data/model_data/versions/2026-07-21_gpa_trend_feature/difficulty_state` does not
exist in the new root. `assert_data_root` (lines 1545–1552) then raises
`RuntimeError: Required data artifact missing or empty` for
`difficulty_state/manifest.json` (`src/paths.py:100–101`). Only
`--difficulty-state-dir` overrides.

**B-5. The new base layer's row counts do not match the constant guarding the
diploma-map fit.**
`rebuild_2026_08_fit_diploma_bucket_map.py:51` pins `EXPECTED_TRAIN_ROWS = 606_562`
and lines 69–73 raise `AssertionError` on any other count. `01_split/train_base_candidate.parquet`
satisfies it (606,562 per `split_summary.json`). `data/model_data/df_train_base.parquet`
does **not** (606,563 rows). Whether this blocks depends on which file the fit is
pointed at; the script reads only the `01_split` path (line 57), so it is not
blocked on its own resolver, and the constant is checkable only against a root
supplied by B-1.

**B-6. `data/artifacts/` is empty.**
`src/paths.py:38` defines `DIPLOMA_TYPE_BUCKET_MAP_PATH = ARTIFACTS_DIR / "diploma_type_bucket_map.json"`.
The directory `data/artifacts/` contains no files. The two diploma scripts in
Part A never read this path (A.2 field 3, A.3 field 2), so they are not blocked by
it. `build_gpa_trend_dataset.py:53` imports it and its `--diploma-map-path` help
says it *"Defaults to the live map, or to the version-local map under
--rebuild-root"* — that script is Part B, not one of the five.

**Not blockers, verified present in the new data root:**
`data/raw/v_crg_student_course_raw.parquet`;
`data/preprocessed/V_ACD_DEGREE_COURSE/clean_v_acd_degree_course.parquet`;
`data/features/selected_model_population.parquet`;
`models/runs/2026-07-16_1025__new-difficulty-logic/feature_contract.json`
(B2's `REFERENCE_RUN`).

---

# PART B — UNKNOWN-STATUS SCRIPTS

## B.1 `scripts/r2_coverage_rescore.py`

1. **What it does.** Re-scores existing five-seed `baseline_41` M1 model binaries
   on difficulty-coverage segments of VALID and writes a covered-vs-uncovered
   comparison report. Docstring line 1: *"Read-only five-seed M1 R2 rescore by
   difficulty-coverage segment."*
2. **Writes dataset columns?** No — read-only diagnostic. Docstring lines 2–5:
   *"Existing ``baseline_41`` M1 binaries are reused; no model is trained or
   tuned. Only the immutable TRAIN course IDs and VALID model rows are read. TEST
   remains closed."* Its writes are `OUT_MD` and `OUT_JSON` (lines 52–53), both
   under `models/runs/`.
3. **Data version.** A **legacy versioned directory**:
   ```python
   DATASET_VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"
   VERSION_DIR = MODEL_DATA_VERSIONS_DIR / DATASET_VERSION
   TRAIN_PATH = VERSION_DIR / "df_train_final.parquet"
   VALID_PATH = VERSION_DIR / "df_valid_final.parquet"
   ```
   (lines 46–49). Present only under `data_old/model_data/versions/`; absent from
   the new root. No TEST path is constructed.
4. **Referenced by.**
   - `tests/test_r2_coverage_rescore.py:10` — `from scripts.r2_coverage_rescore import direction`
   - `tests/test_r2_coverage_rescore.py:61` — `ROOT / "scripts" / "r2_coverage_rescore.py"`
   - `Decisions_Log.md:762`
   - `docs/manifests/codebase_map_2026-08.md:81`
   - `docs/manifests/project_pipeline_routes_2026-08.md:199, 244`
   - `docs/map/05-off-route.md:78`
   - `models/runs/PHASE0_EVIDENCE_RECOVERY.md:568`
   - It imports `scripts/r2_parity.py` at line 36.
5. **Classification: DIAGNOSTIC_ONLY.** Read-only; pinned to a superseded dataset
   version; no importer in `src/` or `scripts/` other than tests.

## B.2 `scripts/r2_parity.py`

1. **What it does.** Asserts that an R2 run differs from its same-seed control by
   `num_leaves` alone. Docstring line 1: *"Parity check: an R2 run must differ from
   its control by num_leaves ALONE."*
2. **Writes dataset columns?** No — read-only. It writes no artifact of its own;
   `docs/map/05-off-route.md:79` records *"no artifact of its own — returns checks
   to the two callers above."* Exit code 0/1 when run standalone (docstring line 8).
3. **Data version.** **None.** It reads no dataset. Its inputs are run artifacts:
   `run_dir / "feature_contract.json"`, `run_dir / "metrics.json"` (lines 49–52),
   and `run_dir / "m1_pass_model.lgbm"` / `m2_grade_model.lgbm` (lines 107–117).
   `ROOT = Path(__file__).resolve().parents[1]` (line 19).
4. **Referenced by.**
   - `scripts/r2_coverage_rescore.py:36` — `from scripts.r2_parity import check as parity_check`
   - `scripts/generate_r2_confirmation_report.py:41` — same import
   - `src/model_training.py:36` (comment)
   - `Decisions_Log.md:438`
   - `docs/PROBLEM_SOLUTION_MAP.md:584`, `docs/map/new.md:584`
   - `docs/manifests/codebase_map_2026-08.md:82`
   - `docs/manifests/project_pipeline_routes_2026-08.md:197, 199`
   - `docs/map/05-off-route.md:79`
   - `models/runs/R2_COVERED_UNCOVERED_FIVE_SEED_REPORT.md:28`
5. **Classification: DIAGNOSTIC_ONLY.** A shared check library for the two R2
   report producers.

## B.3 `scripts/generate_r2_confirmation_report.py`

1. **What it does.** Generates the five-seed R2 (`num_leaves=31`) confirmation
   report by re-scoring each run's saved models against TRAIN/VALID.
2. **Writes dataset columns?** No — read-only diagnostic. Docstring line 15:
   *"Reads TRAIN and VALID only. Never constructs or reads a TEST path. Modifies no
   model or dataset artifact."* Writes `OUT_JSON` / `OUT_MD` under `models/runs/`
   (lines 54–55).
3. **Data version.** A **legacy versioned directory**:
   ```python
   VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"
   TRAIN = ROOT / f"data/model_data/versions/{VERSION}/df_train_final.parquet"
   VALID = ROOT / f"data/model_data/versions/{VERSION}/df_valid_final.parquet"
   ```
   (lines 51–53). Note this one hardcodes the `data/model_data/versions/` segment
   as a literal string relative to `ROOT`, bypassing `src.paths`, so
   `ACADEMIC_ADVISOR_DATA_DIR` cannot redirect it.
4. **Referenced by.**
   - `docs/PROBLEM_SOLUTION_MAP.md:584`, `docs/map/new.md:584`
   - `docs/manifests/codebase_map_2026-08.md:68`
   - `docs/manifests/project_pipeline_routes_2026-08.md:197, 242`
   - `docs/map/05-off-route.md:77`
   - `models/runs/PHASE0_EVIDENCE_RECOVERY.md:667`
   - It imports `scripts/r2_parity.py` at line 41.
   No importer in `src/` or `scripts/`.
5. **Classification: DIAGNOSTIC_ONLY.** Read-only report generator pinned to a
   superseded dataset version.

## B.4 `scripts/diagnose_failure_thresholds.py`

1. **What it does.** Sweeps M1 probability thresholds
   (`THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)`, line 25) and
   reports precision/recall/F1 per threshold and per segment.
2. **Writes dataset columns?** No — read-only diagnostic. Docstring line 3:
   *"This script never trains, saves, or alters a model."* Writes under
   `--output-root` default `root / "models" / "diagnostics"` (line 42).
3. **Data version.** The **live/new** `data/model_data/` directory, via literal
   `Path` composition rather than `src.paths`:
   ```python
   parser.add_argument("--valid", type=Path, default=root / "data" / "model_data" / "df_valid_final.parquet")
   parser.add_argument("--test", type=Path, default=root / "data" / "model_data" / "df_test_final.parquet")
   ```
   (lines 40–41). **Its `--test` default points at TEST.** That is already on
   record: `models/runs/PHASE0_EVIDENCE_RECOVERY.md:663` — *"`--test` **default**
   is `data/model_data/df_test_final.parquet` | would load TEST if run | **would be
   yes** | **latent risk**"*, repeated at line 734. Neither
   `df_valid_final.parquet` nor `df_test_final.parquet` exists in the new
   `data/model_data/` today (which holds only `df_*_base.parquet`).
4. **Referenced by.**
   - `docs/manifests/codebase_map_2026-08.md:63`
   - `docs/manifests/project_pipeline_routes_2026-08.md:204, 238`
   - `docs/map/05-off-route.md:70`
   - `docs/plans/2026-07-21_gpa_trend_feature_plan.md:554, 555, 607`
   - `models/runs/PHASE0_EVIDENCE_RECOVERY.md:663, 734`
   `docs/manifests/codebase_map_2026-08.md:63` states *"no pipeline caller found."*
   No importer in `src/`, `scripts/`, or `note_books/`.
5. **Classification: DIAGNOSTIC_ONLY.**

## B.5 `scripts/diagnose_missed_failures.py`

1. **What it does.** A2/A3 diagnostics for failures the reference M1 model misses,
   segmented by year, degree, attempt number, difficulty fallback level and mark
   band.
2. **Writes dataset columns?** No — read-only diagnostic. Docstring lines 3–6:
   *"It never trains or saves a model and never changes source parquet files or
   reference-run artifacts. Its only writes are four new files inside a new,
   uniquely named diagnostics directory."*
3. **Data version.** The **live/new** `data/model_data/` directory, by literal
   `Path` composition:
   ```python
   parser.add_argument("--valid", type=Path, default=root / "data" / "model_data" / "df_valid_final.parquet")
   parser.add_argument("--train", type=Path, default=root / "data" / "model_data" / "df_train_final.parquet")
   ```
   (lines 61–70). No TEST path is constructed. It also reads
   `root / "models" / "runs" / REFERENCE_RUN_NAME` where
   `REFERENCE_RUN_NAME = "2026-07-12_1513__remove-dead-const"` (lines 25, 58).
   Neither `df_train_final.parquet` nor `df_valid_final.parquet` exists in the new
   `data/model_data/` today.
4. **Referenced by.**
   - `docs/manifests/codebase_map_2026-08.md:64`
   - `docs/manifests/project_pipeline_routes_2026-08.md:205, 238`
   - `docs/map/05-off-route.md:71`
   No importer in `src/`, `scripts/`, or `note_books/`.
5. **Classification: DIAGNOSTIC_ONLY.**

## B.6 `scripts/audit_gpa_trend.py`

1. **What it does.** Rebuilds and audits the leak-safe GPA-trend feature on the
   pre-split timeline and emits a JSON/Markdown report pair. Docstring lines 3–5:
   *"The audit reruns feature engineering from the selected modeling population so
   over-policy semesters remain available to past-only GPA history. It writes a
   compact JSON/Markdown report and never mutates pipeline datasets."*
2. **Writes dataset columns?** No — read-only diagnostic **as a script**. Its
   outputs are `gpa_trend_audit.json` and `GPA_TREND_AUDIT.md` under
   `GPA_TREND_REPORTS_DIR / run_id` (line 737). However, functions it exports are
   used by a script that does write dataset columns (see B.7), and
   `rebuild_2026_08_phase3_assemble.py:93` attributes the two trend columns to it:
   `"scripts/audit_gpa_trend.py": TREND_FEATURES`.
3. **Data version.** Caller-supplied, defaulting to the **live/new** root:
   ```python
   parser.add_argument("--input", type=Path, default=SELECTED_MODEL_POPULATION_PATH)
   parser.add_argument("--split-dir", type=Path, default=MODEL_DATA_DIR)
   ```
   (lines 758, and the parser at the top of `parse_args`), with
   ```python
       split: model_split_path(split, "final", Path(split_dir)) for split in SPLITS
   ```
   (line 241). `SPLITS = ("train", "valid", "test")` (line 47), so the default
   directory lookup constructs a TEST path. Lines 233–237 show callers may pass
   `split_paths` instead; the docstring at lines 228–231 records this is *"how
   callers point the audit at artifacts that are not named ``df_{split}_final.parquet``
   - the rebuild version's files, for instance."*
   `SELECTED_MODEL_POPULATION_PATH` = `data/features/selected_model_population.parquet`
   (`src/paths.py:35`), which **exists** in the new root.
4. **Referenced by.**
   - `scripts/build_gpa_trend_dataset.py:38` — `from scripts.audit_gpa_trend import (...)`
   - `scripts/rebuild_2026_08_phase3_assemble.py:93`
   - `note_books/feature_eng/03_gpa_trend_audit.ipynb:34` — `from scripts.audit_gpa_trend import run_audit`
   - `Decisions_Log.md:1673`
   - `docs/PROBLEM_SOLUTION_MAP.md:242`, `docs/map/new.md:242`
   - `docs/manifests/codebase_map_2026-08.md:54, 105`
   - `docs/manifests/project_pipeline_routes_2026-08.md:91, 172`
   - `docs/map/03-rebuild-2026-08.md:86, 101, 110`
5. **Classification: NEEDED_FOR_CURRENT_REBUILD.** Evidence:
   `docs/map/03-rebuild-2026-08.md:110` — *"lines 269–275 call
   `create_semester_audit_report(...)` from `scripts/audit_gpa_trend.py`"*; and the
   rebuild tree contains `03_features/gpa_trend_audit/` produced by that call.

## B.7 `scripts/build_gpa_trend_dataset.py`

1. **What it does.** Builds a versioned model dataset carrying the isolated
   GPA-trend feature, streams it into copies of the locked split generations, and
   delegates course-difficulty enrichment to the B2 builder. Docstring lines 1–4.
2. **Writes dataset columns?** **Yes.** `TREND_COLUMNS = ["gpa_trend_delta", "gpa_trend_missing"]`
   (line 67). Docstring lines 12–15: *"under ``--rebuild-root`` the trend columns
   are streamed into ``base`` alone and reach the ``difficulty`` generation through
   B2, which copies every non-difficulty column forward."* Both columns are
   **present** in the new base layer (section 0), so this step's output is already
   reflected there.
3. **Data version.** Selectable; defaults to the **live/new** root, with an
   explicit rebuild mode:
   ```python
   parser.add_argument("--input", type=Path, default=SELECTED_MODEL_POPULATION_PATH)
   parser.add_argument("--output-root", type=Path, default=MODEL_DATA_VERSIONS_DIR)
   parser.add_argument("--template-dir", type=Path, default=MODEL_DATA_DIR)
   ```
   and per-split resolution
   ```python
               rebuild_split_path(rebuild_root, split, "base", must_exist=True)
               if rebuild_root
               else model_split_path(split, "base", template_dir)
   ```
   (lines 84–86). `--feature-contract` is **required**,
   `choices=list(B2_ALLOWED_CONTRACTS)`. It also imports
   `DIPLOMA_TYPE_BUCKET_MAP_PATH` (line 53) and `rebuild_diploma_bucket_map_path`
   (line 61).
   Recorded fact: `03_features/gpa_trend_build_report.json` gives
   `"build_id": "phase3_features"`, `"created_at": "2026-08-04T15:26:06+03:00"`,
   `"feature_contract": "concurrent_43"`.
4. **Referenced by.**
   - `tests/test_rebuild_wrapper_paths.py:44` — `TREND = _load("_wrapper_trend", "build_gpa_trend_dataset.py")`
   - `Decisions_Log.md:1606`
   - `docs/PROBLEM_SOLUTION_MAP.md:242, 421`, `docs/map/new.md:242, 421`
   - `docs/manifests/codebase_map_2026-08.md:54`
   - `docs/manifests/project_pipeline_routes_2026-08.md:89, 171, 224, 268`
   - `docs/map/03-rebuild-2026-08.md:44, 73, 86, 99, 110`
   - `models/runs/PHASE0_EVIDENCE_RECOVERY.md:302`
   It imports `scripts/audit_gpa_trend.py` (line 38) and
   `scripts/build_b2_temporal_course_stats.py` (line 44, including `build as build_b2`).
5. **Classification: NEEDED_FOR_CURRENT_REBUILD.** Evidence:
   `docs/map/03-rebuild-2026-08.md:73` — *"`build_gpa_trend_dataset.py` is the
   orchestrator for the feature phase: it calls the B2 builder in-process rather
   than the two being run separately"*; and `03_features/b2_data_report.json`
   records B2's inputs as the temporary `versions/gpa_trend_build_nqgkdg1x/`
   directory this script creates.

## B.8 `scripts/audit_id_columns.py`

1. **What it does.** Scans raw parquet tables, profiles conservative ID-like column
   candidates, and writes CSV/JSON/Markdown audit reports for manual canonical-ID
   planning. Docstring lines 1–11.
2. **Writes dataset columns?** No — read-only diagnostic. Docstring lines 9–11:
   *"The output is for manual canonical-ID planning only. It does not cast,
   normalize, rename, or overwrite any source column."*
3. **Data version.** `RAW_DIR` only — neither a split nor a version:
   docstring line 4 *"scans raw parquet files under ``src.paths.RAW_DIR`` by
   default"*, line 7 output `DATA_DIR / "audit" / "id_dtype_audit"`, and line 610
   `"source_scope": "RAW_DIR only"`. `--data-root` overrides `DATA_DIR`
   (parse_args), and `ACADEMIC_ADVISOR_DATA_DIR` is consulted only as an
   `src.paths`-import fallback (line 142). Resolves to `data/raw/`, which
   **exists**. It never constructs a split path, so it never touches TEST.
4. **Referenced by.**
   - `docs/manifests/codebase_map_2026-08.md:55` — which states *"no pipeline importer found."*
   - `docs/manifests/project_pipeline_routes_2026-08.md:203, 236`
   - `docs/map/05-off-route.md:69`
   - `docs/map/01-extract-and-clean.md:74`
   No importer in `src/`, `scripts/`, or `note_books/`.
5. **Classification: DIAGNOSTIC_ONLY.**

---

# PART C — NAME COLLISION CHECK

Family (a) only: `phase0_*`, `phase1_*`, `phase2_*`, `phase3_predecessor_prior_*`.
Nine files. No guard was added, removed, or altered.

Method: `grep -n "__main__"` per file; `grep -cn "^raise "` per file to detect an
unindented, import-time `raise`.

| File | Execution guard | Quoted guard line |
|---|---|---|
| `scripts/phase0_evidence_recovery.py` | `__main__` gate, line 694 | `if __name__ == "__main__":` → `raise SystemExit(main())` |
| `scripts/phase1_name_key_layer.py` | `__main__` gate, line 888 | `if __name__ == "__main__":` → `raise SystemExit(main())` |
| `scripts/phase2_link_corrections.py` | `__main__` gate, line 1168 | `if __name__ == "__main__":` → `raise SystemExit(main())` |
| `scripts/phase2_mapping_tables.py` | `__main__` gate, line 1048 | `if __name__ == "__main__":` → `raise SystemExit(main())` |
| `scripts/phase2_mapping_tables_scope_fix.py` | `__main__` gate, line 2123 | `if __name__ == "__main__":` → `raise SystemExit(main())` |
| `scripts/phase2_mapping_tables_train_membership.py` | `__main__` gate, line 2035 | `if __name__ == "__main__":` → `raise SystemExit(main())` |
| `scripts/phase3_predecessor_prior_pilot_build.py` | `__main__` gate, line 960 | `if __name__ == "__main__":` → `raise SystemExit(main())` |
| `scripts/phase3_predecessor_prior_pilot_evaluate.py` | `__main__` gate, line 475 | `if __name__ == "__main__":` → `raise SystemExit(main())` |
| `scripts/phase3_predecessor_prior_pilot_report.py` | `__main__` gate, line 1249 | `if __name__ == "__main__":` → `raise SystemExit(main())` |

**Import-time raise: none.** All nine files return `0` matches for an unindented
`^raise ` statement. Every `raise SystemExit(...)` in these files is inside a
function body or inside the `__main__` block.

For completeness, the `SystemExit` calls inside function bodies are runtime STOP
guards, not import guards. Examples, quoted:

- `scripts/phase0_evidence_recovery.py:648` —
  `raise SystemExit("STOP: DifficultyConfig.min_support is no longer 20.")`
- `scripts/phase0_evidence_recovery.py:655` —
  `raise SystemExit(f"STOP: expected 67 best-match pairs, found {len(pairs)}.")`
- `scripts/phase0_evidence_recovery.py:666`, `phase1_name_key_layer.py:182`,
  `phase2_mapping_tables.py:234` —
  `raise SystemExit("STOP: a VALID outcome column was loaded.")`
- `scripts/phase2_link_corrections.py:102` —
  `raise SystemExit(f"STOP: a TEST path entered the input allowlist: {path}")`
- `scripts/phase2_mapping_tables_scope_fix.py:138` —
  `raise SystemExit(f"STOP: a TEST path entered the input allowlist: {path}")`
- `scripts/phase2_mapping_tables_train_membership.py:196` —
  `raise SystemExit(f"STOP: VALID outcome columns were loaded: {forbidden_loaded}")`
- `scripts/phase3_predecessor_prior_pilot_build.py:117` and
  `phase3_predecessor_prior_pilot_evaluate.py:101` —
  `raise SystemExit(f"STOP: {message}")`

Family (b) — `rebuild_2026_08_phase1_*`, `rebuild_2026_08_phase3_*` — was not
examined for guards; the task scopes Part C to family (a) only.

---

# UNKNOWN

Facts that could not be established from the files:

- Whether `data/model_data/df_{train,valid,test}_base.parquet` are the same
  artifacts as `01_split/*_base_candidate.parquet` after a copy/rename, or a
  separately produced base layer. Their row counts differ (606,563 vs 606,562;
  75,383 vs 75,380) and their provenance is not recorded in any file read here.
- The cause of that row-count difference.
- Whether `data_old/` is intended as the live data root for the next run, or is a
  retired copy. No file read states this.
- Which script produced `data/model_data/df_*_base.parquet`. No report in the
  repo names them as an output.
- Whether `2026-08_temporal_rebuild_v1` under `data_old/` is considered complete
  and closed, or is to be rebuilt. `05_dataset/phase3_dataset_report.json` records
  `"model_trained": false`, `"model_evaluated": false`, `"version_promoted": false`,
  but no file states the intended next state.
- The contents and schema of any TEST-split artifact — not read, by constraint.

# BUGS FOUND, NOT FIXED

Reported per the task rule. No change was made to any of these.

1. **`scripts/diagnose_failure_thresholds.py:41` defaults `--test` to a TEST
   path.** `default=root / "data" / "model_data" / "df_test_final.parquet"`.
   Running the script with no arguments would open TEST. Already recorded at
   `models/runs/PHASE0_EVIDENCE_RECOVERY.md:663` as a *"latent risk"*.
2. **`scripts/generate_r2_confirmation_report.py:52–53` hardcodes the
   `data/model_data/versions/` path segment as a literal f-string relative to
   `ROOT`**, bypassing `src.paths`. `ACADEMIC_ADVISOR_DATA_DIR` cannot redirect it,
   unlike `scripts/r2_coverage_rescore.py:47`, which composes the same version
   directory from `MODEL_DATA_VERSIONS_DIR`.
3. **`scripts/build_concurrent_group_features.py:106, 292–297` gate on the
   archived `concurrent_44` contract.** `EXPECTED_LEGACY_MODEL_POSITION = 35` and
   `len(CONCURRENT_44_FEATURES) == 44` are enforced by `_assert_contract`, which
   raises at line 313, even when the build is for `baseline_41` or `concurrent_43`.
   CLAUDE.md section 4 lists `concurrent_44` as archived: *"never use in new runs."*
4. **`scripts/build_concurrent_group_features.py:90` resolves
   `CURRENT_VERSION.txt` from `MODEL_DATA_DIR` and `--rebuild-root` does not
   redirect it.** Every other input of that script is redirectable by
   `--rebuild-root`; this one and the raw CRG source are not. For the raw source
   that is documented as deliberate (docstring lines 14–17); for the
   current-version marker no such statement exists in the file.
