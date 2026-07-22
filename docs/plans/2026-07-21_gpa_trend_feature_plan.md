# PLAN — GPA trend feature (direction of the last two valid prior semester GPAs)

> **Deliverable of this task:** this is the single new file
> `docs/plans/2026-07-21_gpa_trend_feature_plan.md`. It is a **plan**; it authorizes no
> implementation on its own. All line numbers, excerpts, and coverage numbers below were
> read from the working tree on 2026-07-21. The coverage tables were replaced with the
> executed pre-split audit numbers on 2026-07-21 (checklist 7).

---

## Context — why this change

M1 (pass classifier) and M2 (grade regressor) already see the **latest** valid prior
semester GPA through `last_valid_gpa_before_current_semester` and the repaired
`prev_gpa_points_clean`. They do **not** see the *previous* valid GPA before that one.
Because LightGBM splits on individual columns, the model cannot reconstruct the
**direction** of recent performance (improving vs declining) from a single level column.

Goal: add exactly one isolated feature-engineering signal — the delta between a student's
last two *valid* prior semester GPAs, strictly before the current prediction semester.
Illustrative (on the real `gpa_points` scale, which is **0–4.0**, verified below):
t‑2 = 2.5, t‑1 = 3.0 → delta = +0.5 (improvement); t‑2 = 3.5, t‑1 = 3.0 → −0.5 (decline).

Scope is **change class B (feature engineering) only** — no hyperparameters, thresholds,
class weights, calibration, other features, or unrelated cleanup. The later experiment
must isolate this feature.

---

## Section 1 — Current valid-GPA history construction

**File:** `src/feature_engineering.py`, function `build_semester_history`
(def at **line 404**). The valid-GPA history is built at **semester grain** at
**lines 495–507**:

```python
495  # Shift valid GPA history so the current semester's GPA never leaks into
496  # features used to predict that same semester.
497  valid_gpa_for_history = gpa_points.where(
498      gpa_points.gt(0).fillna(False) & semester_df["is_interruption_semester"].eq(0)
499  )
500  semester_df["last_valid_gpa_before_current_semester"] = (
501      valid_gpa_for_history.groupby(
502          [semester_df[column] for column in STUDENT_DEGREE_KEY],
503          sort=False,
504          dropna=False,
505      )
506      .transform(lambda values: values.ffill().shift(1))
507  )
```

`is_interruption_semester` is defined just above at **lines 471–473**:

```python
471  semester_df["is_interruption_semester"] = (
472      reg_credits.gt(0) & pass_credits.eq(0) & gpa_points.eq(0).fillna(False)
473  ).astype(int)
```

Interpretation:

- **What counts as a valid GPA:** `gpa_points > 0` **AND** `is_interruption_semester == 0`
  (line 497–498). A zero GPA and any interruption semester are excluded from history.
- **Grouping keys:** `STUDENT_DEGREE_KEY = ["university_id", "student_id", "degree_id"]`
  (**lines 36–40**). `university_id` is included so identical student/degree IDs from
  different campuses never share a timeline.
- **Grain:** this runs on `semester_df`, which is **semester grain** — one row per
  `SEMESTER_KEY = ["university_id","student_id","degree_id","part_id"]` (**lines 30–35**),
  built by aggregating course rows earlier in `build_semester_history`
  (groupby+agg at **lines 431–456**; uniqueness asserted at **lines 460–461**).
- **How prior GPA reaches the current semester:** `valid_gpa_for_history` is NaN on every
  non-valid row; `ffill()` carries the last valid value forward; `.shift(1)` makes it
  **strictly before** the current semester row. The result is later merged back to every
  course row by `merge_semester_history` (def **line 535**).

The new feature reuses this exact construction and adds the *second-to-last* valid value.

---

## Section 2 — Existing previous-GPA indicator pattern

**File:** `src/feature_engineering.py`, function `repair_previous_gpa_chain`
(def **line 617**). The `_missing` / `_zero` / `_invalid_zero_case` indicators are at
**lines 631–635**, and the `_clean` / `_fill_source` outputs at **lines 639–665**:

```python
631  df["prev_gpa_points_missing"] = raw_prev_gpa.isna().astype(int)
632  df["prev_gpa_points_zero"] = raw_prev_gpa.eq(0).fillna(False).astype(int)
633  df["prev_gpa_invalid_zero_case"] = (
634      raw_prev_gpa.eq(0).fillna(False) & df["is_first_active_semester"].ne(1)
635  ).astype(int)
...
660  df["prev_gpa_points_clean"] = clean
661  df["prev_gpa_fill_source"] = source.astype("string")
```

A parallel course-grain missing-indicator pattern also appears as
`course_difficulty_missing`, `requirement_type_missing`, and
`degree_requirement_credits_count_missing` (built in `add_requirement_features`,
**lines 810–821**).

**Which parts the new feature follows:** the **value + paired `_missing` indicator**
convention — one continuous value column plus one 0/1 indicator, derived at course grain
from `value.isna()`, exactly like `course_difficulty_missing`.

**Which parts it does NOT need:** the multi-source fallback chain
(`raw → last_valid → start_agpa → zero_fallback`, **lines 642–658**) and the
`_fill_source` / `_invalid_zero_case` / `_replaced_due_to_invalid_zero` machinery. The
trend feature has a **single source** (the valid `gpa_points` sequence) and **no fallback
imputation** — when t‑2 is unavailable the value stays NaN and the indicator is set. There
is no "clean" repaired variant and no zero-fallback.

---

## Section 3 — Timeline ordering

**File:** `src/feature_engineering.py`, function `_sort_semester_frame`
(def **line 359**). The sort is at **lines 369–377**:

```python
369  semester_df[part_sort_key] = pd.to_numeric(semester_df["part_id"], errors="coerce")
370  semester_df[part_sort_text] = semester_df["part_id"].astype("string")
371
372  sorted_df = semester_df.sort_values(
373      ["university_id", "student_id", "degree_id", part_sort_key, part_sort_text],
374      kind="mergesort",
375      na_position="last",
376  ).reset_index(drop=True)
```

- **Enrollment identity keys:** `university_id`, `student_id`, `degree_id`
  (= `STUDENT_DEGREE_KEY`). `student_id` alone is **not** sufficient — verified by the
  regression test `test_timeline_grouping_is_university_aware`
  (`tests/test_feature_engineering.py:123–152`), where the same student/degree on two
  `university_id`s must **not** share GPA history.
- **Semester ordering key:** numeric `part_id` first, with the string form of `part_id` as
  a deterministic tiebreaker; `kind="mergesort"` (stable).
- **Is `part_id` used?** Yes — `part_id` **is** the semester ordering key (it is also the
  4th component of `SEMESTER_KEY`).
- **Stability with multiple course rows in one semester:** not applicable at this point —
  the frame is already collapsed to one row per `SEMESTER_KEY` before sorting, so there is
  exactly one row per semester. Course-row multiplicity is handled by the later
  many-to-one merge (Section 4).
- **Is there a single monotonic integer semester-order key?** **No.** Ordering is
  established only by this sort plus `groupby(..., sort=False)` preserving order; `part_id`
  is a dotted string, and `part_year` / `part_semester` (built later at **lines 698–709**)
  are row-wise course-grain columns, not a per-enrollment order index. **Consequence:** the
  recommended implementation (Option B below) must **not** assume a monotonic key; it relies
  on the same sort + grouped `ffill().shift()` idiom the existing `last_valid` already uses.
  (Option A / `merge_asof` would first require deriving such a key.)

---

## Section 4 — Semester grain vs course grain

Confirmed grain flow inside `run_feature_engineering_job` (def **line 1016**):

1. `build_semester_history` (**line 1040**) collapses course rows to **one row per
   `SEMESTER_KEY`** (groupby+agg **lines 431–456**; duplicate-key assertion
   **lines 460–461**). All timeline features, including the new trend, are computed here.
2. `merge_semester_history` (**line 1041**) joins those semester features back to **all**
   course rows with `validate="many_to_one"` (**lines 591–597**) and prefixes/renames to
   avoid collisions.

Empirically verified on the current final splits (read-only): after deduping to
`SEMESTER_KEY`, **0** semester groups have more than one `gpa_points` value — i.e. semester
GPA is genuinely constant across a semester's course rows, and computing the sequence
directly on course rows would repeat each semester GPA once per registered course. The plan
**requires** computing the trend on the unique semester timeline, then joining back.

The plan requires:

- **Unique semester key:** `SEMESTER_KEY = ["university_id","student_id","degree_id","part_id"]`.
- **Join keys:** the same four keys (via the temp normalized keys `merge_semester_history`
  already builds, **lines 539–555**).
- **Join cardinality:** many-to-one (many course rows ← one semester row), enforced by
  `validate="many_to_one"` (already present, **line 595**).
- **Row-multiplication guard:** right-side uniqueness assertion already present
  (**lines 562–564**); plus the existing row-count invariant that `merge` is left-only. The
  implementation must assert course-row count is unchanged before/after the join
  (checklist item 5).

---

## Section 5 — Meaning and status of `gpa_points`

- **Per-semester outcome:** `gpa_points` is one of the `SEMESTER_AGGREGATION_COLUMNS`
  (**line 47**) collapsed to one value per semester; it is the semester GPA outcome, not a
  course-level field.
- **Derive-only:** it is a hard leakage column. `LEAKAGE_COLUMNS` includes `"gpa_points"`
  (**line 118**) under the explicit comment at **lines 113–115**:

  ```python
  113  # Columns that must NEVER enter the model matrix X.
  114  # gpa_points and semester_pass_credits are used ONLY to derive shifted, past-only
  115  # history features inside this job; they are leakage only if placed in X.
  ```

- **Never in X:** `assert_no_leakage_columns` (def **lines 153–164**) raises if any
  `LEAKAGE_COLUMNS` member appears in X; it is called inside `prepare_X_y`
  (`src/model_training.py:263`) and again on all three splits before training
  (**lines 767–772**). Regression test `test_assert_no_leakage_columns_raises_on_gpa_points`
  (`tests/test_feature_engineering.py:251–256`) locks this. `gpa_points` is also **not** in
  `MODEL_FEATURES` (`src/model_training.py:63–113`).

The new columns are the *only* additions to X; `gpa_points` itself remains excluded.

---

## Section 6 — Interruption semesters (observed on real data, read-only)

Read-only inspection of `data/model_data/df_{train,valid,test}_final.parquet`:

- `gpa_points` is **never null** in the final splits (0 nulls across all 716,570 course
  rows) and ranges **0.0 → 4.0** (mean ≈ 2.14).
- **Forward direction (proven):** every interruption semester carries `gpa_points == 0.0`
  exactly — of 15,698 interruption course-rows, **all 15,698** are `0.0` (none null, none any
  other value).
- **Reverse direction (does NOT hold — measured):** not every zero-GPA row is an
  interruption. Across the splits there are 16,211 course-rows with `gpa_points == 0`, of
  which **513** have `is_interruption_semester == 0` (all 513 in **train**; 0 in valid/test).
  At semester grain: 6,190 zero-GPA semesters, **107** of them non-interruption. These 513
  rows / 107 semesters are non-interruption **"invalid zeros."**
- Corrected statement: *every interruption semester has `gpa_points == 0`; the reverse does
  **not** hold (513 course rows / 107 semesters are zero-GPA yet non-interruption). Both
  interruption zeros and non-interruption invalid zeros are excluded from valid history by
  the `gpa_points > 0` condition regardless of the interruption flag.*

Interaction with validity: interruptions are excluded from the valid subsequence but still
occupy a row in the ordered timeline, which is precisely what makes the naïve `shift(2)`
(below) corrupt interrupted students.

---

## Section 7 — Current baseline contract

**Baseline run named in the original task:** `2026-07-16_1008__new-difficulty-logic`. It
**exists**: `models/runs/2026-07-16_1008__new-difficulty-logic/feature_contract.json`.

- `CURRENT_N = n_features = 39` (contract line 3; `features` array lines 4–44).
- `dropped_features` has **8** entries (contract lines 70–79):
  `is_interruption_semester`, `model_prev_gpa`, `prev_gpa_actual_zero_performance`,
  `start_level_missing`, `difficulty_fallback_level`, `part_year`, `start_year`,
  `difficulty_group_support_count`. **This list must stay unchanged.**
- Expected post-change contract: **`CURRENT_N + N_new = 39 + 2 = 41`** (N_new per the D3
  decision below).

**Baseline decision (Shrko, 2026-07-21):** the comparison run is
**`2026-07-16_1025__new-difficulty-logic`**, not `_1008`. Three sibling runs
(`_1008`, `_1025`, `_1439__…-0-85`) share identical M1 valid metrics (AUC 0.80856,
fail‑AP 0.3222) per `leaderboard.csv`; `_1025` is the one that recorded its **own**
`--compare-to 2026-07-12_1513__remove-dead-const`, so selecting it keeps the comparison
chain unbroken. `CURRENT_N` is the same (39) across all three, so this choice does not
change the expected 39 → 41 contract move. The later training run uses
`--compare-to 2026-07-16_1025__new-difficulty-logic`.

---

## The rejected approach — `shift(2)` on the forward-filled series (MUST be rejected)

```python
ffilled_valid_gpa.groupby(enrollment_keys).shift(2)   # WRONG — do not implement
```

Why it is wrong:

- After `ffill()`, the series holds **repeated** values. `shift(2)` returns the value two
  **rows** back on the *filled* series, not the second-to-last **valid** observation.
- When an interruption semester or an invalid zero sits between two real observations,
  `shift(2)` returns the same value as `shift(1)`, producing **delta = exactly 0.0** that
  looks like a legitimate "no change".
- The failure is **silent**: the value is non-null and in-range, so it never trips a
  null-count check or an assert. A `0.0` produced this way is **indistinguishable** from a
  true flat trend, so the bug would never surface in null counts or asserts. It corrupts
  precisely the students with interrupted timelines — the segment the product cares about.

**Measured cost of the bug (executed pre-split audit):** the `shift(2)` series differs
from the correct value on **14,749** model-matrix course rows, and **every one** of those
miscomputations lands on an exact wrong `0.0` (train 10,989 / valid 1,855 / test 1,905).
That is the number of rows doing it correctly protects.

---

## The recommended correct approach

Operate on the **sequence of valid observations itself**, not on the forward-filled series.
Required properties (all satisfied below): runs on the unique semester timeline (Section 4);
filters to valid rows first (`gpa_points > 0 & is_interruption_semester == 0`, Section 1);
within the valid-only subsequence, position `k` gives t‑1 and `k-1` gives t‑2; both values
are propagated onto every semester row referring **strictly** to semesters *before* the
current one; the current semester's own GPA never appears.

### Recommended: **Option B — align-then-shift** (no monotonic order key needed)

Recommended because Section 3 confirms **no monotonic integer order key exists**, and this
reuses the exact `ffill().shift(1)` idiom already at lines 500–507, making t‑1 identical to
`last_valid_gpa_before_current_semester` **by construction**. Added inside
`build_semester_history`, right after the existing `last_valid` block, on the already-sorted
`semester_df`:

```python
# valid GPA on valid rows only, NaN elsewhere (same mask as last_valid)
valid_gpa_for_history = gpa_points.where(
    gpa_points.gt(0).fillna(False) & semester_df["is_interruption_semester"].eq(0)
)
grp = valid_gpa_for_history.groupby(
    [semester_df[c] for c in STUDENT_DEGREE_KEY], sort=False, dropna=False
)

# t-1 at every row == existing last_valid (strictly-before via .shift(1) after ffill)
t1 = grp.transform(lambda v: v.ffill().shift(1))          # == last_valid_gpa_before_...

# "previous valid GPA AT a valid row" = ffill().shift(1) read on valid rows only
prev_valid_at_valid = t1.where(valid_gpa_for_history.notna())

# propagate that onto every row, then make it strictly-before with one more shift(1)
t2 = (
    prev_valid_at_valid
    .groupby([semester_df[c] for c in STUDENT_DEGREE_KEY], sort=False, dropna=False)
    .transform(lambda v: v.ffill().shift(1))
)

semester_df["gpa_trend_delta"] = t1 - t2     # NaN wherever t-2 is undefined
```

Strictly-before mechanism (must be stated): the **final `.shift(1)` after `ffill()`** — the
same mechanism the existing `last_valid` uses. `t1`/`t2` at row *k* are drawn only from valid
observations with order strictly less than *k*; the current semester never contributes. The
`gpa_trend_missing` indicator is derived at **course grain after the merge-back**, from
`gpa_trend_delta.isna()` (mirrors `course_difficulty_missing`).

### Named alternative: **Option A — `merge_asof`** (NOT recommended here)

Correct in principle, and the recommended shape *if* a monotonic `order_key` existed:

```python
sem = timeline.sort_values(enrollment_keys + [order_key])
is_valid = sem["gpa_points"].gt(0) & sem["is_interruption_semester"].eq(0)
obs = sem.loc[is_valid, enrollment_keys + [order_key, "gpa_points"]].copy()
g = obs.groupby(enrollment_keys, sort=False)
obs["v_t1"] = obs["gpa_points"]
obs["v_t2"] = g["gpa_points"].shift(1)
sem = pd.merge_asof(
    sem.sort_values(order_key),
    obs.sort_values(order_key)[enrollment_keys + [order_key, "v_t1", "v_t2"]],
    on=order_key, by=enrollment_keys,
    direction="backward", allow_exact_matches=False,   # <-- enforces "strictly before"
)
```

Here the strictly-before guarantee is `allow_exact_matches=False`. **Trade-off:** Option A
needs a derived monotonic `order_key` (extra code + an alignment pitfall — `merge_asof`
returns a reindexed frame, so results must be joined back on keys, never positionally), and
it does not give the free `t1 == last_valid` identity. Option B is preferred.

**Required consistency assertion (either option):** on rows where both are defined, the
produced t‑1 must equal the existing `last_valid_gpa_before_current_semester`. Under Option B
this is true by construction; a read-only reconstruction from the final splits already
matched at **98.11%** (the 1.89% gap is **likely** the over-policy / population-filtered
semesters that are absent from the final splits but present in the pre-split audit frame —
which is *why* the feature must be built inside `build_semester_history`, before any split;
attribution verified during implementation, checklist item 8).

---

## Decisions (resolved by Shrko, 2026-07-21)

**D1 — Definition of "last two semesters." → DECIDED: valid-GPA definition.**
Use the last two semesters with a **valid GPA** (`gpa_points > 0` and
`is_interruption_semester == 0`), consistent with the existing `last_valid` definition —
not two strictly-chronological prior rows. Rationale: validity-aware matches the model's
existing GPA history semantics and avoids treating interruption zeros as performance.

**D2 — Delta basis. → DECIDED: per-semester `gpa_points`.**
Use the per-semester **`gpa_points`** valid sequence (t‑1 minus t‑2), not cumulative
`start_agpa_points` at two points. Rationale: per-semester GPA expresses **recent**
direction; cumulative AGPA is a slow-moving lifetime average whose two-point difference
mostly reflects credit-weighted inertia — a different, already-partly-present signal, not
"did the student just improve or decline."

**D3 — Output columns (YAGNI). → DECIDED: keep both. `N_new = 2`.**
Minimal set: one continuous `gpa_trend_delta` + one `gpa_trend_missing` indicator
(→ contract 39 → **41**). No sign bucket, magnitude bucket, or raw t‑2 level column: a tree
extracts sign and magnitude from the continuous delta via ordinary threshold splits, and the
raw t‑2 level adds nothing the pair (`last_valid` = t‑1, `delta`) does not already span.
The indicator is **kept** (Shrko) even though LightGBM handles NaN natively — for
consistency with the established value+`_missing` pattern and because "trend undefined"
aligns with product-relevant segments (first/second semester, interruptions).

**D4 — Missing policy. → DECIDED: NaN value + indicator, no imputation.**
Value = **NaN** when t‑2 (or both observations) is unavailable, paired with
`gpa_trend_missing = 1`. **No** cross-student fill, **no** medians, **no** zero-as-neutral.
Rationale: a **true** flat trend is `delta == 0.0` (1.10% of computable train rows). Using
`0.0` as the missing sentinel would be indistinguishable from a real flat trend — the same
silent collision that dooms the `shift(2)` approach. NaN cannot be confused with any real
delta and lets LightGBM branch on missingness explicitly.

**D5 — Naming and registration. → DECIDED.**
Names `gpa_trend_delta` (continuous) and `gpa_trend_missing` (0/1 indicator), per
`docs/naming_plan.md` (names say what they are; snake_case; value + `_missing`). Register in
exactly these places:
- `src/feature_engineering.py`: add `"gpa_trend_delta"` to `SEMESTER_FEATURE_COLUMNS`
  (**lines 62–71**) so the merge carries it to course grain; **extend the nullable
  exclusion at line 604** so `gpa_trend_delta` is *not* `fillna(0).astype(int)` (it must
  stay nullable exactly like `last_valid_gpa_before_current_semester`); derive
  `gpa_trend_missing` at course grain after the merge — using the **same construction as the
  existing `_missing` indicators** (`course_difficulty_missing`, `requirement_type_missing`),
  so its persisted dtype matches theirs rather than being fixed here; add both to
  `FEATURE_COLUMNS` (**lines 75–111**).
- `src/model_training.py`: add both to `MODEL_FEATURES` (**lines 63–113**); bump
  `EXPECTED_FEATURE_COUNT` **39 → 41** (**line 142**). `TRAINING_DATA_COLUMNS`
  (**lines 151–156**) then picks them up automatically.
- `feature_contract.json` is an **output** of the training run
  (`_save_feature_contract`, **lines 631–645**) — it regenerates to 41 features and records
  the **actual** dtypes automatically; do **not** hand-edit it. Keep `DROPPED_FEATURES`
  (**lines 116–125**) unchanged.
Rejected names `gpa_delta` / `gpa_change`, and registering only in the model allowlist:
`gpa_trend_*` states the concept, and both the diagnostic catalog and the model allowlist
must know the column or the merge/contract checks drift.

**D6 — Scale & drift. → DECIDED: raw delta, no guardrail.**
No rescaling/normalization — feed the raw delta on the real 0–4.0 scale (trees are
scale-invariant). The drift question is closed: a difference of two GPAs has no calendar
component (unlike `part_year` / `start_year`), so no drop-list guardrail is needed.

---

## Leakage constraints

- **Strictly-before:** only semesters before the current prediction semester contribute.
  Mechanism = the final **`.shift(1)` after `ffill()`** in Option B (or
  `allow_exact_matches=False` in Option A). Named explicitly, matching the existing
  `last_valid` guarantee.
- **Pure per-student window — no train-only fitting:** the computation reads only a single
  student-degree's own past GPAs. It learns **no** statistic from other students, so it
  needs **no** train-only fit step and introduces **no** train/valid/test coupling. This is
  explicitly **unlike** the course-difficulty statistics, which *do* require train-only
  computation. Because it is per-student and past-only, it is correct to compute it once in
  `build_semester_history` on the **full pre-split audit frame** (this is also why the
  coverage reconstruction from post-split files diverged by 1.9%).
- **`gpa_points` stays derive-only:** it remains in `LEAKAGE_COLUMNS` (**line 118**) and out
  of `MODEL_FEATURES`. The only additions to X are `gpa_trend_delta` and `gpa_trend_missing`.
- **Leakage gate:** `assert_no_leakage_columns` (**lines 153–164**) checks that no
  `LEAKAGE_COLUMNS` member is in X (called at `model_training.py:263` and 767–772). The two
  new columns are derived, past-only, and not in `LEAKAGE_COLUMNS`, so they pass.
- **Dropped list:** the new columns are **not** added to `dropped_features` and **not**
  removed from it; the 8-entry list is unchanged.

---

## Coverage analysis (executed pre-split build)

**Both tables below are final audit values from the true pre-split frame.** The feature was
computed on all 163,130 unique semesters before policy exclusion, then mapped to the locked
train/valid/test course rows. The produced t‑1 matched the existing `last_valid` value on
all 163,130 semester rows. Full details and five hand/oracle checks are in
`data/model_data/versions/2026-07-21_gpa_trend_feature/gpa_trend_audit/`.

**Table 1 — course-row grain (model-matrix rows):**

| Split | N (course rows) | delta computable (has t‑2) | t‑1 only (no t‑2) | neither (no t‑1) | wrong `0.0` under `shift(2)` |
|-------|----------------:|---------------------------:|------------------:|-----------------:|------------------------------:|
| train | 450,465 | 330,768 (73.43%) | 53,772 | 65,925 | 10,989 |
| valid | 156,097 | 124,675 (79.87%) | 14,990 | 16,432 | 1,855 |
| test  | 110,008 | 87,831 (79.84%)  | 6,705  | 15,472 | 1,905 |

**Table 2 — unique enrollment-semester grain (dedup on `SEMESTER_KEY`; the grain the feature
is actually computed on before the join-back):**

| Split | n semesters | delta computable (has t‑2) | t‑1 only | neither |
|-------|------------:|---------------------------:|---------:|--------:|
| train | 96,477 | 75,205 (77.95%) | 10,043 | 11,229 |
| valid | 34,293 | 28,247 (82.37%) |  2,934 |  3,112 |
| test  | 22,772 | 18,690 (82.07%) |  1,241 |  2,841 |

- **Share the feature can affect at all:** ~73% (course grain, train) to ~80% (valid/test)
  of rows get a computable delta; ~78–82% at semester grain. This is a **large** share, so a
  real (if modest) metric movement is plausible — not a niche feature. The *incremental*
  signal beyond the existing `last_valid` (= t‑1) applies to the has‑t‑2 rows; t‑1-only and
  first-semester rows gain only the `_missing` indicator.
- **`first_semester` rows** (`is_first_active_semester == 1`) can **never** have a delta:
  train 65,121 (14.46%) / valid 14,732 (9.44%) / test **15,447** (14.04%). ✔ matches the
  reference test n. (Note: `is_first_active_semester == 1` differs slightly from timeline
  ordinal 1 below because of the known first-semester-concept mismatch for transfer/
  advanced-standing students, e.g. train 65,121 vs 64,135.)
- **Second-semester rows — corrected claim:** a second active semester can **never** have
  t‑2. It has **t‑1 only if the first semester was valid**, otherwise **neither**. Measured
  by **timeline semester ordinal** (position in `SEMESTER_KEY` order), course grain:

  | Split | 1st sem | 2nd sem | 3rd+ sem | 2nd → t‑1 only | 2nd → neither | 3rd+ → t‑2 | 3rd+ → t‑1 only | 3rd+ → neither |
  |-------|--------:|--------:|---------:|---------------:|--------------:|-----------:|----------------:|---------------:|
  | train | 64,135 | 53,389 | 332,941 | 52,020 | 1,369 | 330,768 | 1,752 | 421 |
  | valid | 16,156 | 14,920 | 125,021 | 14,701 |   219 | 124,675 |   289 |  57 |
  | test  | 15,393 |  6,590 |  88,025 |  6,538 |    52 |  87,831 |   167 |  27 |

  So the generic "t‑1 only" column of Table 1 is **not** the second-semester count: most
  t‑1-only rows are second semesters, but a small tail of 3rd+ rows are t‑1-only because an
  interruption/invalid-zero broke their early history (train 1,752; valid 289; test 167).
- **Interruption-heavy timelines:** **14,749** course rows total (train 10,989 / valid 1,855
  / test 1,905) would have received a **wrong `0.0`** delta under the rejected `shift(2)`.
  This is the concrete value of the correct implementation.
- **Delta distribution on train (correct):** n = 330,768; min −3.13, max +3.08, mean
  +0.0096; quantiles 5% −1.08 / 25% −0.38 / 50% +0.01 / 75% +0.40 / 95% +1.09; **1.11%**
  exactly 0.0 (true flat). Range is plausible for the 0–4.0 `gpa_points` scale.
- **Reconstruction gap (verified):** rebuilding t‑1 from primary-only history mismatched
  2,596 / 158,835 semesters (1.63%), affecting 10,696 model-matrix course rows (train 6,745 /
  valid 2,672 / test 1,279). **All 2,596 semester mismatches were attributed to a valid
  over-policy semester** present in the true pre-split audit timeline and intentionally absent
  from the primary/final splits.

---

## Verification checklist for the implementation task (gate the later run against this)

1. **Manual spot-check of 5 students** with ≥3 semesters — including one with an
   interruption semester and one retake-heavy timeline. Hand-compute expected t‑1, t‑2,
   delta; compare to output. Show the table in the report. (The existing test
   `test_interruption_semester_does_not_become_valid_previous_gpa`,
   `tests/test_feature_engineering.py:87–121`, is the fixture template.)
2. **Brute-force reference oracle (REQUIRED).** Implement a naïve per-semester
   reference — for each enrollment-semester, collect valid GPAs from *strictly earlier*
   semesters, take the last two, and diff — and assert **exact equality** with the Option B
   output on synthetic fixtures covering all of: (i) an interruption semester **between** two
   valid observations, (ii) a non-interruption invalid zero between valids, (iii) a single
   valid observation, (iv) a first-row valid semester, and (v) the two-university
   same-student collision case. This is required **in addition to** item 3, because the
   `t1 == last_valid` identity does **not** exercise t‑2 at all.
3. **Assertion:** produced t‑1 == `last_valid_gpa_before_current_semester` wherever both
   are defined (free check; identity under Option B). Does not validate t‑2 — see item 2.
4. **Assertion:** no row's t‑1 or t‑2 comes from the current semester (strictly-before).
5. **Row-count assertion** before/after the join-back to course grain: identical count, no
   duplicate matches (`validate="many_to_one"` already enforced, lines 591–597).
6. **Null + indicator counts per split**, plus train delta distribution (min/max/mean/
   quantiles); confirm range plausible for the 0–4.0 scale.
7. **Recompute on the real pre-split build.** Recompute the null/indicator counts and
   **both** coverage tables (course grain and enrollment-semester grain) on the true
   pre-split frame, and **replace** the preliminary reconstruction numbers in this plan's
   coverage section with the run-report numbers.
8. **Verify the reconstruction-gap cause.** Confirm the actual cause of the 1.89%
   t‑1-vs-`last_valid` mismatch against the exact pre-split input frame (expected:
   over-policy / population-filtered semesters), and record it in the run report.
9. **Contract update:** 39 → 41; `dropped_features` unchanged; **record the actual persisted
   dtypes** the run writes for `gpa_trend_delta` and `gpa_trend_missing` (match the existing
   `_missing` indicator columns; do not assert a fixed dtype in advance).
10. **New versioned dataset folder** under `data/model_data/versions/` (rebuild flows
    through feature engineering → splits → difficulty → bucketing).
11. **Exactly one training run**, `--compare-to 2026-07-16_1025__new-difficulty-logic`
    (baseline decision, Section 7).
12. **Selection on VALID only:** `fail_avg_precision`, `auc`, `brier`, `train_valid_auc_gap`.
    TEST stays descriptive.
13. **Reporting threshold — ZERO threshold-code changes.** Do **not** touch `evaluate_pass`'s
    0.85 binarization (`src/model_training.py:454`) or `_THRESHOLDS` (line 544). The 0.80
    fail-class P/R is produced **after** training by running the existing
    `scripts/diagnose_failure_thresholds.py`, whose `THRESHOLDS` sweep already includes 0.80
    (`scripts/diagnose_failure_thresholds.py:25`). It is a readability aid only — **not** a
    selection metric, **not** a product decision.
14. **Segment check:** report valid AUC for `first_semester` and `retake_attempt`
    separately (`collect_segment_auc`, lines 507–530). **`first_semester` is expected NOT to
    improve** (feature undefined there) — state this in advance so a flat result is not
    misread as failure; `retake_attempt` is where most rows have t‑2 and direction may help.

**Tests / scripts that will break and must be updated when this plan is executed:**
- `tests/test_model_training.py:30` — hardcoded `assertEqual(EXPECTED_FEATURE_COUNT, 39)` → 41.
- `scripts/build_b2_temporal_course_stats.py:421–426` — the "MODEL_FEATURES remains the same
  39-column contract" gate references a 39-feature reference contract; it will need the new
  baseline/threshold. `scripts/migrate_legacy_baseline.py` hardcodes 39 but is a one-off
  legacy migration (likely not re-run — confirm).
- Per `docs/pipeline_rules.md`, run `scripts/parity_check.py` after the change and report the
  diff before declaring done.

---

## Execution record (2026-07-21)

- **Implementation:** Option B was added to `src/feature_engineering.py`; the two features
  were registered in the diagnostic catalog and 41-feature model allowlist. The rejected
  `shift(2)` method was not used, and threshold code was unchanged.
- **Automated verification:** all 27 repository tests passed, including the new brute-force
  valid-history oracle fixture. The notebook JSON and Python syntax checks also passed.
- **Versioned data:** `data/model_data/versions/2026-07-21_gpa_trend_feature/` was published
  without overwriting the live split files. All B2 row/order, lineage, null, feature-delta,
  and read-back gates passed.
- **Persisted data dtypes:** `gpa_trend_delta=float64` and `gpa_trend_missing=int64` in the
  versioned parquet files. The training contract records both as `float64` after the model's
  standard numeric casting, consistent with the existing numeric missing indicators.
- **Contract:** the comparison run regenerated a 41-feature contract; the eight-entry
  `dropped_features` list is unchanged.
- **Exactly one training run:** `2026-07-21_1224__gpa-trend-feature`, compared with
  `2026-07-16_1025__new-difficulty-logic`.
- **VALID-only selection metrics:** fail-AP 0.32216 → 0.32198 (−0.00018); AUC 0.80856 →
  0.80919 (+0.00063); Brier 0.08071 → 0.08078 (+0.00007, worse); train-valid AUC gap
  0.06221 → 0.05538 (−0.00682, better). Evidence is mixed rather than a clean win.
- **Segments:** valid first-semester AUC 0.7366 → 0.7329 (no improvement was expected because
  trend is undefined); valid retake-attempt AUC 0.6726 → 0.6768 (+0.0042).
- **0.80 diagnostic only:** VALID fail precision/recall/F1 = 0.3307/0.4195/0.3698; TEST
  descriptive = 0.2771/0.3480/0.3085. No threshold was selected or changed.
- **Parity check:** 46/51 legacy checks passed. The five failures are existing repository
  state outside this feature's scope: two missing sentinel artifacts, a pre-existing hash
  drift for `final/without_outliers.parquet`, a missing `archive/pre_7c`, and one hardcoded
  path in `note_books/debug/try_new_course.ipynb`. The versioned build's own parity/read-back
  gates all passed.

## Open items

All prior open questions were resolved by Shrko on 2026-07-21 and are baked into the body:
baseline → `2026-07-16_1025__new-difficulty-logic` (Section 7); reporting threshold →
diagnostic-only via `diagnose_failure_thresholds.py`, no threshold-code changes
(checklist 13); indicator kept, `N_new = 2` (D3); Option B (D5/approach); raw delta, no
rescaling (D6); drift guardrail not needed (D6). No open questions remain.

## Assumptions made
- The implementation lives in `build_semester_history` (pre-split, full audit frame), not in
  any per-split notebook, because the feature is per-student and past-only.
- The dataset rebuild (feature engineering → base splits → difficulty → bucketing) will
  regenerate the final splits so the new columns exist before training; a new
  `data/model_data/versions/` generation is created rather than overwriting in place.
- `feature_contract.json` is regenerated by the training run, not hand-edited.

## Discrepancies found between the original task prompt and the repo
- Prompt line refs 497–506 / 621–663 / ~372 map to current **497–507 / 631–665 (context
  617–691) / 369–377** — verified and quoted above.
- Prompt's "test n was 15,447" — confirmed as the **test** `first_semester` count (14.04%);
  note the reference run's `metrics.json` persists only **valid** segments (valid
  `first_semester` n = 14,732), so 15,447 is reproduced here by read-only reconstruction, not
  read from a stored artifact.
- Memory note "repo `data/` is an empty skeleton" (dated 2026-07-11) is **stale**: the final
  splits exist locally under `data/model_data/` (written 2026-07-12) and were used for the
  read-only coverage analysis above.
