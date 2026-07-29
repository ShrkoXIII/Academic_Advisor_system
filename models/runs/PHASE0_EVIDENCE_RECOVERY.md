# Phase 0 — evidence recovery before the degree/curriculum lineage diagnostic

**Status: READ-ONLY. Nothing was trained, rescored, mapped, rebuilt or promoted.**

| | |
|---|---|
| HEAD at run time | `afcd2d4` (`Verify 67 course identity candidates by degree`) |
| Working tree at start | one untracked file: `tests/test_degree_course.ipynb` |
| Frozen dataset version | `2026-07-26_batched_fixes__registration_roster_concurrent` |
| Splits read | `df_train_final.parquet`, `df_valid_final.parquet` |
| TEST parquet | not read, not globbed, not stat-ed, not path-constructed by this task |
| VALID outcome columns | `final_mark` never loaded (asserted in `scripts/phase0_evidence_recovery.py:main`) |
| Models loaded / trained / rescored | 0 / 0 / 0 |
| Datasets or mappings written | none |
| Artifacts written | `models/runs/PHASE0_EVIDENCE_RECOVERY.md`, `models/runs/phase0_pair_divergence.csv`, `scripts/phase0_evidence_recovery.py` |

New computation for Q2/Q4/Q5/Q6 was produced by
[`scripts/phase0_evidence_recovery.py`](../../scripts/phase0_evidence_recovery.py).
It imports `src.course_difficulty` (the feature module that produced the on-disk
columns, TRAIN-only, trains nothing) and `scripts/course_identity_investigation.py`
(to reuse `degree_lineage` verbatim). It imports no training code.

---

## Verdict per question

| Q | Subject | Verdict |
|---|---|---|
| Q1 | Strict 67-pair degree verification script | **VERIFIED with a caveat** — script and outputs exist and reproduce; degree sets are **catalog-only**, and a sensitivity check shows `0/67` becomes **12/67** under `catalog ∪ enrolment` |
| Q2 | Reconciling classifications A and B | **CORRECTED** — B is an *independent* re-classification, not a reconciliation of A; B's 67 contain **4 of 5**, not 5 of 5, of A's `confirmed_equivalent` |
| Q3 | Coverage figures and the covered definition | **VERIFIED** — every figure reproduces; `min_support = 20` in force and never changed |
| Q4 | Degree multiplicity of the 67 pairs | **VERIFIED** — multiplicity confirmed from source, it is a **catalog** fact; 0 pairs are 1:1; 33 distinct set-transitions |
| Q5 | Existing `degree_lineage()` state | **VERIFIED** — 25 VALID-only degrees, 2 with zero predecessors; naming convention does encode versions |
| Q6 | Size of a predecessor prior's effect | **VERIFIED** — row-weighted mean \|Δ pass rate\| = **0.0633** |
| Q7 | Model decision artifacts | **CORRECTED** — items 1 and 3 verified exactly; item 2 (a standalone *M2* five-seed R2 experiment) **does not exist** as its own artifact; 8 legacy runs record **no threshold** |
| Q8 | TEST access ledger | **CORRECTED** — the prompt's claim about the reconciliation script is literally accurate, but `TEST = closed_not_read` is **not** literally accurate for the repository as a whole |

---

## Q1 — the strict degree-level verification script

**1. Does the script exist?** Yes.
[`scripts/course_identity_67_degree_verification.py`](../../scripts/course_identity_67_degree_verification.py),
committed at `afcd2d4` (2026-07-28).

**2. Does the output CSV exist?** Yes — three of them:

| Artifact | Path | Content |
|---|---|---|
| all candidate pairs | `models/runs/COURSE_IDENTITY_67_DEGREE_VERIFICATION.csv` | every (new, old) pair, 53 columns |
| best match per course | `models/runs/COURSE_IDENTITY_67_BEST_MATCH_PER_COURSE.csv` | 67 rows |
| **the review file quoted in the prompt** | `models/runs/COURSE_IDENTITY_67_HUMAN_REVIEW.csv` | 67 rows, exactly the 20 columns quoted |

The 20-column list in the prompt matches `REVIEW_COLUMNS` at
[`course_identity_67_degree_verification.py:134-155`](../../scripts/course_identity_67_degree_verification.py#L134-L155)
exactly, in order. The headline counts quoted in the prompt are reproduced verbatim in
`models/runs/COURSE_IDENTITY_67_DEGREE_VERIFICATION.md` lines 9-18 and 56-57:
same university 67/67, same exact `degree_id` 0/67, strict same-degree renumbering
0/67, all 67 classified `SAME_UNIVERSITY_DIFFERENT_DEGREE`, enrolment overlap
True/False = 64/3.

**3. How `old_degree_id` and `new_degree_id` were populated — CATALOG ONLY.**

This is the decisive answer. The degree sets come from the **catalog**
(`V_ACD_DEGREE_COURSE`), never from enrolments:

```python
551  new_catalog = catalog.loc[catalog["course_id"].eq(new_id)].copy()
552  old_catalog = catalog.loc[catalog["course_id"].eq(old_id)].copy()
553  new_model = all_model.loc[all_model["course_id"].eq(new_id)]
554  old_model = all_model.loc[all_model["course_id"].eq(old_id)]
555
556  new_degrees = values_for(new_catalog, "degree_id", id_field=True)
557  old_degrees = values_for(old_catalog, "degree_id", id_field=True)
```
— [`course_identity_67_degree_verification.py:551-557`](../../scripts/course_identity_67_degree_verification.py#L551-L557)

`catalog` is built at lines 466-495 by merging the canonical
`clean_v_acd_degree_course.parquet` with the raw `v_acd_degree_course.parquet` at
`degree_course_id` grain — no enrolment rows enter it. The TRAIN/VALID frames
(`new_model` / `old_model`, lines 553-554) are used **only** for `university_id`
cross-check (`course_universities`, lines 512-529), for the temporal signal
(`temporal_summary`, line 590) and for row counts (lines 625-630). No union of
catalog and enrolment degrees is ever formed.

The report itself states this explicitly and contrasts it with the earlier
diagnostic, which *did* pool enrolment degrees with catalog degrees —
`COURSE_IDENTITY_67_DEGREE_VERIFICATION.md:37`:

> "The earlier diagnostic built a course-level degree set by pooling `degree_id`
> values from enrolment rows with catalog rows. … This verification uses every
> actual catalog row and compares full normalized `degree_id` strings directly."

**The prompt's concern is well founded, but the mechanism is the reverse of the one
it names.** The sets are catalog-derived, not enrolment-derived. However the
`0 / 67` is **not** robust to the choice of definition — and this task tested it
rather than assuming it.

**Sensitivity check (new computation, does not modify the committed verification).**
Recomputing `same_degree` with the union `catalog ∪ TRAIN enrolment ∪ VALID
enrolment` on each side:

| Degree-set definition | Pairs with a non-empty intersection |
|---|---:|
| catalog only (**what the script did**) | **0 / 67** |
| catalog ∪ enrolment (both splits) | **12 / 67** |

The 12 pairs are `1175.111`, `1172.111`, `1174.111`, `1179.111`, `1173.111`,
`1422.111`, `1183.111`, `1184.111`, `1429.111`, `1164.111`, `1425.111`,
`1426.111` — 18 (pair, shared degree) links in total. Tracing each link to its
source is what settles the interpretation:

| Where the shared degree comes from | Links | Pairs |
|---|---:|---:|
| old course's **VALID enrolment** × new course's **catalog** listing | 17 | 11 |
| old course's **TRAIN enrolment** × new course's **catalog** listing | 1 | 1 |
| old course's catalog × new course's catalog | **0** | **0** |

So in 11 of the 12 pairs the "shared degree" means *the old course was still being
taught inside the new degree during 2022-2023* — a teach-out coexistence, not a
shared catalog placement. That does not weaken the verification's conclusion; it
points the same way as the 64/67 enrolment-overlap finding: old and new run
side by side. Exactly **one** pair has a shared degree that predates VALID —
`1172.111` ↔ `431.111` via degree `49.111`, a TRAIN degree the new course is
catalogued in.

**Net effect on the blocker.** `0/67` is a statement about *catalog* degree
placement and is correct as such. It is not a statement that no old/new pair ever
shares a degree in any sense: under the broader definition 12 pairs do, and 1 of
those 12 does so on evidence older than VALID. Whether that reopens the blocker is
the human's decision; this report changes nothing.

**4. `same_degree` and `enrolment_overlap`.**

`same_degree` — non-empty exact intersection of the two normalized full
`degree_id` string sets:

```python
242      shared = old_degrees & new_degrees
…
268          "same_degree": bool(shared),
```
— [`course_identity_67_degree_verification.py:242,268`](../../scripts/course_identity_67_degree_verification.py#L242-L268)

Normalization is `normalize_id_series` from `src/cleaning_utils.py`, applied via
`values_for(..., id_field=True)` (lines 218-227). Names, faculty and the dotted
university suffix cannot make it true (JSON `definitions.same_degree`).

`enrolment_overlap` (column `overlap_in_active_enrolment`) — the two course IDs
appear in at least one common semester across concatenated TRAIN+VALID:

```python
288      old_parts = normalize_id_set(old_rows["part_id"])
289      new_parts = normalize_id_set(new_rows["part_id"])
…
313      shared_parts = old_parts & new_parts
314      overlap = bool(shared_parts)
```
— [`course_identity_67_degree_verification.py:288-314`](../../scripts/course_identity_67_degree_verification.py#L288-L314)

`all_model` is TRAIN ∪ VALID (lines 1063-1066). VALID ends at `20233`, so the
overlap statistic is censored at that boundary — the same limitation the 2025
reconciliation was built to relax.

**5. Deterministic and re-runnable today?** Yes, with one caveat.

- All five inputs exist: candidate CSV, prior MD/JSON, canonical catalog, raw
  catalog, TRAIN, VALID (verified on disk).
- No RNG, no wall-clock, no environment-dependent ordering: every set is
  rendered through `sorted_join` (line 183-185) and every sort passes
  `kind="stable"` (lines 660-670, 1097-1101).
- It hashes all seven inputs before and after and raises if any changed
  (lines 1153-1159).
- **Caveat, not re-run here:** `main()` writes five artifacts
  (`OUT_ALL`, `OUT_BEST`, `OUT_REVIEW`, `OUT_JSON`, `OUT_MD`, lines 1214-1222),
  all of which are already committed. Re-running is outside this task's write
  allowance, so re-runnability is asserted from the code, not demonstrated.

---

## Q2 — reconciling the two classifications

**1. Which artifact produced which.**

| | Classification A | Classification B |
|---|---|---|
| Report | `models/runs/COURSE_IDENTITY_INVESTIGATION.md` | `models/runs/COURSE_IDENTITY_DIAGNOSTIC.md` |
| Script | `scripts/course_identity_investigation.py` | `scripts/course_identity_diagnostic.py` |
| CSV | `models/runs/course_identity_candidates.csv` | `models/runs/COURSE_IDENTITY_CANDIDATES.csv` |
| Commit / date | `0a9f346`, 2026-07-28 | `c6a9656`, 2026-07-28 |
| Buckets | `confirmed_equivalent` 5 / `likely_equivalent_needs_review` 82 / `genuinely_new` 7 / `unresolved` 88 | `likely_renumbered_needs_review` 67 / `genuinely_new` 11 / `unresolved` 104 |

The prompt's "A" collapses A's 82 + 88 into "pending 170" — that matches
`COURSE_IDENTITY_INVESTIGATION.md` lines 46-49 exactly. The prompt's "B" matches
`COURSE_IDENTITY_DIAGNOSTIC.md` lines 65-68 exactly (67 / 13,686 rows;
11 / 216 rows; 104 / 11,725 rows).

> **Artifact-integrity finding, not asked for but material.** Commit `c6a9656`
> replaced `models/runs/course_identity_candidates.csv` with
> `models/runs/COURSE_IDENTITY_CANDIDATES.csv`. On Windows these are the **same
> path**. Classification A's CSV therefore no longer exists on disk; only git
> holds it (`git show 0a9f346:models/runs/course_identity_candidates.csv`, which
> is how this section recovered it). A live consequence:
> `scripts/course_identity_reconciliation_2025.py:75` still declares
> `PRIOR_CSV = .../course_identity_candidates.csv`, which today silently resolves
> to classification **B**'s file. That script would not reproduce its own report
> if re-run now.

**2. Is B a later reconciliation of A?** **No — B is an independent
classification.** Three pieces of evidence:

- **B disagrees with A on the top predecessor for 67 of 182 courses.** Example:
  for `1423.111` (اللغة الإنكليزية – 1, 979 rows) A's top candidate is `830.111`
  and B's is `391.111`. Neither report cites the other's pick.
- **B's taxonomy is not A's.** B has no `confirmed_equivalent` bucket at all
  (`COURSE_IDENTITY_DIAGNOSTIC.md:65` records `confirmed_equivalent 0`), and
  B's rule is score-based (`>= 0.85` name similarity, score `>= 55`, two
  structural/temporal signals — `COURSE_IDENTITY_DIAGNOSTIC.md:59`) while A's is
  clause-based and requires a degree-lineage link
  (`COURSE_IDENTITY_INVESTIGATION.md:172`).
- **The actual reconciliation of A is a different artifact.**
  `COURSE_IDENTITY_RECONCILIATION_2025.md` (`c70c661`) explicitly re-classifies
  A's 170 pending courses and outputs 13 / 74 / 7 / 88 — a third set of numbers
  that is neither A nor B. Chronologically B (`c6a9656`) was committed *after*
  that reconciliation and does not reference it.

**3. Overlap between A and B.** Cross-tabulation over all 182 courses:

| A bucket | B status | Courses |
|---|---|---:|
| `confirmed_equivalent` | `likely_renumbered_needs_review` | 4 |
| `confirmed_equivalent` | `unresolved` | **1** |
| `likely_equivalent_needs_review` | `likely_renumbered_needs_review` | 56 |
| `likely_equivalent_needs_review` | `unresolved` | 26 |
| `unresolved` | `likely_renumbered_needs_review` | 7 |
| `unresolved` | `genuinely_new` | 4 |
| `unresolved` | `unresolved` | 77 |
| `genuinely_new` | `genuinely_new` | 7 |

**The 67 in B do NOT contain all 5 of A's `confirmed_equivalent`.** They contain
**4**: `1172.111`, `1174.111`, `1305.111`, `1379.111`. The fifth, **`1201.111`**,
is `unresolved` in B. B's 67 is assembled as 4 (from A-confirmed) + 56 (from
A-likely) + 7 (from A-unresolved).

**4. Is the 182 set identical in both?** **Yes** — set equality holds exactly;
A-only and B-only are both empty.

---

## Q3 — coverage figures against current artifacts

All eight figures **reproduce**. Source of record:
`models/runs/DIFFICULTY_COVERAGE_DIAGNOSTIC.md` (commit `a32f20c`), produced by
`scripts/difficulty_coverage_diagnostic.py`.

| Claim | Found | Where | Computing code |
|---|---|---|---|
| VALID total rows 156,097 | 156,097 | `DIFFICULTY_COVERAGE_DIAGNOSTIC.md:86` | `coverage_summary`, [`difficulty_coverage_diagnostic.py:217`](../../scripts/difficulty_coverage_diagnostic.py#L217) |
| uncovered rows 26,882 | 26,882 | `…md:92, 96-98` | [`:226`](../../scripts/difficulty_coverage_diagnostic.py#L226) `int((~confident).sum())` |
| never_in_train 25,627 (95.33%) | 25,627 / 95.33% | `…md:96` | `decompose_uncovered`, [`:239, 248`](../../scripts/difficulty_coverage_diagnostic.py#L239-L248) |
| thin_history 1,255 (4.67%) | 1,255 / 4.67% | `…md:97` | [`:240-244, 249`](../../scripts/difficulty_coverage_diagnostic.py#L240-L249) |
| TRAIN confident coverage 89.61% | 89.61% | `…md:85` | [`:227`](../../scripts/difficulty_coverage_diagnostic.py#L227) |
| VALID confident coverage 82.78% | 82.78% | `…md:86` | [`:227`](../../scripts/difficulty_coverage_diagnostic.py#L227) |
| student-semesters ≥1 uncovered 23.55% | 8,076 / 34,293 = 23.55% | `…md:224, 239` | `set_level_prevalence`, [`:372, 396-399`](../../scripts/difficulty_coverage_diagnostic.py#L372-L399) |
| affected majority-uncovered 73.45% | 5,932 / 8,076 = 73.45% | `…md:237` | [`:373-375, 401-404`](../../scripts/difficulty_coverage_diagnostic.py#L373-L404) |

The student-semester grain is `university_id, student_id, degree_id, part_id`
(`src/feature_engineering.py:28-35`), and "majority" is
`uncovered_course_count * 2 > course_count` (line 373-375) — strict majority.

**Covered definition in force — confirmed verbatim** at
[`src/course_difficulty.py:578-590`](../../src/course_difficulty.py#L578-L590):

```python
578  course_is_new = (~supports[1].notna() & ~supports[2].notna()).astype("int64").to_numpy()
579  course_low_support = (
580      (course_history > 0) & (course_history < int(state.config.min_support))
581  ).astype("int64")
…
588  feature_values["course_difficulty_missing"] = (
589      (course_is_new == 1) | (course_low_support == 1)
590  ).astype("int64")
```

`covered ⇔ course_difficulty_missing == 0` is the definition used for every
covered/uncovered split (`difficulty_coverage_diagnostic.py:215`;
`DIFFICULTY_COVERAGE_DIAGNOSTIC.md:54`).

**`min_support` = 20 in the CURRENT config, and no run used a different value.**

- Declared at [`src/course_difficulty.py:79`](../../src/course_difficulty.py#L79)
  — `min_support: int = 20`.
- Persisted with the frozen frames: the version manifest
  `data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent/phase7_registration_roster_report.json`
  records `inputs.difficulty_state.config.min_support = 20`,
  `shrinkage_k = 20.0`, `target_threshold = 50.0`,
  `version = b2_temporal_course_stats_v1`.
- Recomputed live in this task: the state refit from TRAIN reports
  `min_support_in_effect = 20`, `shrinkage_k_in_effect = 20.0`.
- Every builder CLI defaults to 20
  (`scripts/build_b2_temporal_course_stats.py:520`,
  `scripts/build_gpa_trend_dataset.py:234`), and
  `git log -L 79,79:src/course_difficulty.py` shows the default has had exactly
  one revision — its introduction at `decf675`.
- The **only** non-20 value anywhere is `tests/test_course_difficulty.py:51`
  (`min_support=2`), a synthetic unit-test fixture that touches no project data.

---

## Q4 — degree multiplicity structure of the 67 pairs

**1. Multiplicity confirmed from the source data,** not from the review CSV. Degree
sets were recomputed directly from `data/raw/v_acd_degree_course.parquet` (grouped
by course, `answer_q4` in `scripts/phase0_evidence_recovery.py`), and separately
from TRAIN+VALID enrolments.

| Measure | Value |
|---|---:|
| Pairs where old and/or new course spans >1 degree | **67 / 67** |
| Pairs where **both** sides span >1 degree | 43 |
| Pairs that are strictly one-degree-to-one-degree | **0** |
| Largest new-side degree count | **23** |
| Largest old-side degree count | **29** |

The prompt's example reproduces exactly: `391.111 → 1423.111`
(اللغة الإنكليزية 1, 979 VALID rows) is 1 old degree → **23** new degrees.

**2. Catalog fact, not an enrolment artifact.** The multiplicity is present in the
catalog and is *larger* there than in enrolment:

| Check | Pairs (of 67) |
|---|---:|
| new-side enrolment degree set == catalog degree set | 9 |
| new-side enrolment degree set ⊆ catalog degree set | **65** |
| old-side enrolment degree set == catalog degree set | 35 |
| old-side enrolment degree set ⊆ catalog degree set | **44** |

Stronger evidence still: of the 35 distinct new-side catalog degrees across the 67
pairs, **9 appear in neither TRAIN nor VALID enrolment at all**
(`38.111`, `51.111`, `57.111`, `58.111`, `59.111`, `60.111`, `61.111`, `64.111`,
`65.111`). A degree with zero enrolment rows cannot be an enrolment artifact — the
same course is simply *listed in many degree plans*.

The exceptions run the other way and are decision-relevant: in **23** pairs the
old course's enrolment degree set is **not** a subset of its catalog set, and in
**2** pairs the new course's is not — students were enrolled in the course under a
degree the catalog does not list for it. This catalog-completeness gap is exactly
what produces the 12/67 result of the Q1 sensitivity check: for 11 pairs the old
course carries a **VALID** enrolment under a degree that the new course is
catalogued in. Catalog-only comparison cannot see those memberships.

**3. Per-pair old-degree count, new-degree count and mapping shape** — all 67 rows
are in `models/runs/phase0_pair_divergence.csv`, columns
`n_old_degrees_catalog`, `n_new_degrees_catalog`, `n_old_degrees_enrolment`,
`n_new_degrees_enrolment`, `mapping_shape`, `old_degree_ids_catalog`,
`new_degree_ids_catalog`. Distribution of shapes:

| Shape | Pairs |
|---|---:|
| N:M | 43 |
| 1:N | 24 |
| N:1 | 0 |
| **1:1** | **0** |

**4. Distinct old-degree-set → new-degree-set transitions: 33.**
Nine of the 33 transitions carry more than one pair, with sizes
**12, 9, 7, 5, 2, 2, 2, 2, 2** (43 pairs); the remaining **24 transitions are
singletons**. The largest is the informatics transition
`21.111|3.111|4.111|5.111|6.111 → 26.111|27.111|29.111|30.111|31.111|45.111|46.111|47.111|48.111|49.111`
(12 pairs). Column `degree_set_transition_id` in the CSV assigns each pair to its
transition.

So the 67 pairs are really 33 lineage questions, and two thirds of those are a
single pair each — the shared structure is concentrated in a handful of programme
migrations, not spread evenly.

**This confirms the prompt's structural point:** a pair-level
`old_degree_id → new_degree_id` model does not fit — there is no pair in the data
for which such a scalar mapping is even well-defined.

---

## Q5 — state of the existing `degree_lineage()` function

Recomputed by calling
[`scripts/course_identity_investigation.py::degree_lineage`](../../scripts/course_identity_investigation.py#L525-L583)
verbatim on the frozen TRAIN/VALID and the raw catalog. The prompt's description of
the function is accurate: migration `>= 5` students (line 565), name similarity
`>= NAME_SIM_PLAUSIBLE = 0.60` (lines 89-90, 570), `\b20\d{2}\b` stripped and a
`family/` prefix reduced to its tail (lines 543-550).

**1. VALID-only (new) degree IDs: 25.** (TRAIN has 20 distinct degrees, VALID 43;
43 − 18 shared = 25 VALID-only.)

**2. Per new degree** — `mig` = predecessors reaching the ≥5-student migration bar;
`name` = TRAIN degrees at name similarity ≥ 0.60; `linked` = the union the function
returns:

| New degree | Name | VALID rows | mig | name | linked | Linked predecessors |
|---|---|---:|---:|---:|---:|---|
| `26.111` | الهندسة المعلوماتية/هندسة البرمجيات ونظم المعلومات | 5,678 | 3 | 4 | 6 | 21.111, 3.111, 4.111, 18.111, 49.111, 6.111 |
| `29.111` | الهندسة المعلوماتية/هندسة أمن النظم والشبكات الحاسوبية | 3,303 | 3 | 1 | 3 | 21.111, 3.111, 6.111 |
| `41.111` | دكتور في الطب 2023 | 3,125 | 0 | 4 | 4 | 2.111, 15.111, 1.111, 16.111 |
| `27.111` | الهندسة المعلوماتية/هندسة الذكاء الصنعي وعلوم البيانات | 2,966 | 2 | 2 | 3 | 4.111, 21.111, 18.111 |
| `42.111` | إجازة دكتور في طب الأسنان 2023 | 2,765 | 0 | 4 | 4 | 1.111, 16.111, 2.111, 15.111 |
| `44.111` | الصيدلة و الكيمياء الصيدلية 2023 | 2,149 | 0 | 1 | 1 | 13.111 |
| `40.111` | العلوم الإدارية (اختصاص عام) 2023 | 1,430 | 0 | 1 | 1 | 22.111 |
| `34.111` | إدارة الموارد البشرية | 1,339 | 0 | 2 | 2 | 8.111, 19.111 |
| `30.111` | هندسة التحكم والروبوت | 1,313 | 2 | 5 | 7 | 21.111, 5.111, 49.111, 6.111, 20.111, 18.111, 4.111 |
| `36.111` | المحاسبة | 1,305 | 0 | 1 | 1 | 11.111 |
| `31.111` | هندسة الاتصالات | 924 | 1 | 7 | 7 | 6.111, 49.111, 20.111, 4.111, 21.111, 5.111, 3.111 |
| `39.111` | العلوم الإدارية (اختصاص عام) | 874 | 0 | 1 | 1 | 22.111 |
| `45.111` | هندسة التحكم والروبوت 2023 | 862 | 0 | 5 | 5 | 49.111, 6.111, 20.111, 18.111, 4.111 |
| `48.111` | الهندسة المعلوماتية/هندسة أمن النظم والشبكات الحاسوبية 2023 | 862 | 0 | 1 | 1 | 6.111 |
| `47.111` | الهندسة المعلوماتية/هندسة الذكاء الصنعي وعلوم البيانات 2023 | 855 | 0 | 2 | 2 | 4.111, 18.111 |
| `50.111` | إجازة في هندسة البترول2023 | 855 | 0 | 2 | 2 | 24.111, 20.111 |
| `46.111` | الهندسة المعلوماتية/هندسة البرمجيات ونظم المعلومات 2023 | 706 | 0 | 4 | 4 | 3.111, 18.111, 49.111, 6.111 |
| `55.111` | المحاسبة 2023 | 496 | 0 | 1 | 1 | 11.111 |
| **`35.111`** | **التسويق** | **454** | **0** | **0** | **0** | **—** |
| `33.111` | إدارة الأعمال | 448 | 0 | 3 | 3 | 19.111, 22.111, 8.111 |
| `53.111` | إدارة الموارد البشرية 2023 | 436 | 0 | 2 | 2 | 8.111, 19.111 |
| `37.111` | التمويل والمصارف | 351 | 0 | 1 | 1 | 7.111 |
| `54.111` | إدارة الأعمال2023 | 154 | 0 | 3 | 3 | 19.111, 22.111, 8.111 |
| **`56.111`** | **التسويق2023** | **149** | **0** | **0** | **0** | **—** |
| `52.111` | التمويل والمصارف 2023 | 125 | 0 | 1 | 1 | 7.111 |

This table reproduces `COURSE_IDENTITY_INVESTIGATION.md:142-166` exactly, and adds
the name-similarity predecessor counts, which that report did not tabulate.

**Note on signal strength:** only **5 of 25** new degrees have *any* migration
link (`26.111`, `27.111`, `29.111`, `30.111`, `31.111` — all engineering). For
the other 20, the lineage rests entirely on degree-name similarity.

**3. New degrees with ZERO linked predecessors: 2** — `35.111` التسويق (454 VALID
rows) and `56.111` التسويق2023 (149 rows). 603 VALID rows total.

**4. Version-encoding in degree names — the convention is real.** Of 58 catalog
degrees:

- **16 carry a `20\d{2}` token**, in every case the literal `2023`, always as the
  final token: `40.111`, `41.111`, `42.111`, `44.111`, `45.111`, `46.111`,
  `47.111`, `48.111`, `49.111`, `50.111`, `51.111`, `52.111`, `53.111`, `54.111`,
  `55.111`, `56.111`. Two spacing variants occur — with a space
  (`المحاسبة 2023`) and without (`التسويق2023`, `إدارة الأعمال2023`,
  `إجازة في هندسة البترول2023`).
- **12 carry a `family/specialisation` prefix**, across **2 distinct families**:
  `الهندسة المعلوماتية` (6 degrees: `26.111`, `27.111`, `29.111`, `46.111`,
  `47.111`, `48.111`) and `هندسة الذكاء الاصطناعي` (6 degrees: `57.111`,
  `58.111`, `59.111`, `60.111`, `61.111`, `64.111`).
- The two patterns co-occur: `46/47/48.111` are the `2023` re-issues of
  `26/27/29.111` and carry both markers. **The university does encode curriculum
  versions in the naming convention**, and the existing
  `degree_key()` strip logic (lines 543-550) is what turns those pairs into
  name-similarity links.

**5. Coverage of the 67 pairs' new degree IDs.** The 67 pairs' new-side catalog
degree sets span **35 distinct degrees**:

| Status | Count | Degrees |
|---|---:|---|
| VALID-only **and** linked by `degree_lineage()` | **23** | the 23 non-blank rows of the table above that appear in the pairs |
| VALID-only but **zero** linked predecessors | **2** | `35.111`, `56.111` |
| A TRAIN degree — outside the function's scope by construction | 1 | `49.111` |
| **In no enrolment at all** (catalog-only) — invisible to the function | **9** | `38.111`, `51.111`, `57.111`, `58.111`, `59.111`, `60.111`, `61.111`, `64.111`, `65.111` |

So 23 of 35 are covered; **12 are not**, for two structurally different reasons.
`degree_lineage()` iterates over `valid_degrees - train_degrees` (line 539), so
catalog degrees with no enrolment can never enter it.

---

## Q6 — how much would a predecessor prior actually change the estimate?

Computed on the 67 best-match pairs (`COURSE_IDENTITY_67_BEST_MATCH_PER_COURSE.csv`),
**13,686 VALID rows**. Full per-pair table:
[`models/runs/phase0_pair_divergence.csv`](phase0_pair_divergence.csv).

**Method and its guarantees.** The difficulty state was refit with
`fit_difficulty_state(TRAIN)` — the exact frozen state VALID scoring uses
(`src/course_difficulty.py:410-429`). Before use, the refit was validated against
the on-disk frozen columns: **0 `course_history_count` mismatches over 156,097
VALID rows** (verdict: exact). "Current" values are the frozen
`course_pass_rate_historical` / `course_avg_mark_historical` already on the VALID
rows. "Predecessor" values are the TRAIN estimate the same hierarchy would produce
after substituting the predecessor `course_id` into the row's degree-course key —
Level 1 if that substituted key exists in TRAIN, else Level 2. **Both sides are
TRAIN-derived. VALID `final_mark` was never loaded; no pair was ranked, filtered or
selected by anything derived from a VALID label.**

**Where these rows sit today.** They are not in a void, and they are not at the
global fallback either:

| Current `difficulty_fallback_level` | Rows |
|---|---:|
| 1 (degree+course) | 0 |
| 2 (course across degrees) | 0 |
| 3 (degree+req+credits) | 0 |
| **4 (faculty+req+credits)** | **2,858** |
| **5 (req+credits)** | **10,828** |
| 6 (global) | 0 |

All 13,686 rows currently take a *structural* estimate at Level 4 or 5. Level 3 is
never reached because these are VALID-only degrees, and Level 6 is never reached
because the Level-5 key always exists.

**What the prior would replace it with.** For **66 of 67 pairs** the predecessor
estimate is the Level-2 (course-across-degrees) TRAIN value; for 1 pair a subset of
rows also hits a substituted Level-1 key. Predecessor TRAIN support (Level 2):
median **223**, min **1**, max **9,254**; no pair has zero support. Two pairs rest
on almost nothing — `448.111` (1 TRAIN row) and `445.111` (3 rows) — where the
"prior" would be ~95% shrinkage toward the global TRAIN mean.

### The aggregate — stated plainly

**A predecessor prior would move `course_pass_rate_historical` by a row-weighted
mean of 0.0633 in absolute value.**

| `course_pass_rate_historical`, \|current − predecessor\| | Value |
|---|---:|
| rows | 13,686 |
| **row-weighted mean** | **0.0633** |
| median | 0.0615 |
| p75 | 0.0760 |
| p90 | 0.1201 |
| max | 0.2615 |
| pair-level mean (unweighted over 67 pairs) | 0.0706 |

| `course_avg_mark_historical`, \|current − predecessor\| (marks, 0-100) | Value |
|---|---:|
| **row-weighted mean** | **4.09** |
| median | 3.10 |
| p75 | 5.54 |
| p90 | 8.54 |
| max | 21.94 |

**Direction is not symmetric.** The row-weighted *signed* delta
(predecessor − current) is **−0.0264**: the predecessor prior is systematically
*harsher* than the structural estimate these rows carry now. By pair count only
20 of 67 predecessors are harsher, but those 20 include almost every high-volume
pair, so they dominate the row weighting.

**Two anchors for reading 0.0633, both from repository record and neither a
recommendation:**

- The concurrent difficulty scalar is exactly `d = 1.0 − course_pass_rate_historical`
  (`src/concurrent_group_features.py:29,208-209`), so 0.0633 in pass rate is
  0.0633 in `d` — for these rows, one-for-one.
- The recorded TRAIN/VALID means of `peer_difficulty_mean` are 0.186 and 0.134
  (`scripts/build_concurrent_group_features.py:141-145`), a difference of 0.052.
  The prior's typical move on these rows (0.063) is **larger than the entire
  recorded TRAIN→VALID shift** in that feature's mean.

**The distribution matters as much as the mean.** Half the rows move less than
0.062, and the shape is tight up to p75 (0.076) then opens out: the p90 is 0.120
and the tail reaches 0.261. The pairs at the tail are small-volume
(`1271.111`/`417.111`, 13 rows, Δ 0.261) or thin-support, while the largest-volume
pairs sit near the middle of the distribution. Per-pair figures — including
`predecessor_train_support_l2` and the min/max spread of the current estimate
within each pair — are in the CSV.

---

## Q7 — model decision artifacts

**1. Five-seed R2 covered/uncovered rescore → `KEEP_DEFAULT_127_FOR_M1`.** **EXISTS.**

- `models/runs/R2_COVERED_UNCOVERED_FIVE_SEED_REPORT.md` + `.json`, produced by
  `scripts/r2_coverage_rescore.py`, commit `0ce2160` (2026-07-28).
- Pre-registered in `docs/EXPERIMENT_R2_COVERAGE_DECISION_PLAN.md` at `6fa053e`.
- Headline: **`KEEP_DEFAULT_127_FOR_M1`** (`…REPORT.md:3`). "R2 did not meet the
  burden of proof" (`Decisions_Log.md:741`). Clause outcomes: clause 1 FAIL,
  clause 2 PASS, clause 3 FAIL, clause 4 FAIL, clause 5 FAIL. All five parity
  pairs passed 22 checks each.
- `Decisions_Log.md:758` records the M2 side of the same decision: *"M2 remains
  `concurrent_43` with `num_leaves=127`; R2 is not applied to M2."*

**2. "The M2 five-seed experiment concluding R2 is not adopted."** **CORRECTED —
no such standalone artifact exists.** There is no M2-specific five-seed R2
experiment. M2's R2 outcome is a *secondary readout* inside the M1-targeted
confirmation, `models/runs/R2_CONFIRMATION_5SEED_REPORT.md` (commit `653e7f1`,
pre-registered at `docs/EXPERIMENT_R2_CONFIRMATION_PLAN.md`, `235a1db`):

> "**M2 impact: HARMED_WITHIN_NOISE** — VALID MAE worsened in >=4 of 5 seeds in
> both arms; every five-seed mean is inside the band." (`…REPORT.md:18, 311`)

Per-seed VALID MAE deltas (R2 − control), `baseline_41` / `concurrent_43`
(`…REPORT.md:315-324`): seed 42 +0.031470 / +0.029485; seed 52 +0.013610 /
+0.007601; seed 62 +0.052116 (*outside band, harmful*) / +0.024177; seed 72
+0.026957 / +0.027113; seed 82 −0.023698 / +0.045199. Nine of ten are judged
`inside_band`. The verdict there is "harmed but within noise", not "not adopted";
the *non-adoption* for M2 is a decision sentence
(`R2_COVERED_UNCOVERED_FIVE_SEED_REPORT.md:7-8`, `Decisions_Log.md:758`), not a
separate experiment.

**3. Covered/uncovered performance figures.** **ALL FOUR PAIRS VERIFIED EXACTLY**,
in `models/runs/DIFFICULTY_COVERAGE_DIAGNOSTIC.md:189-190` (M1) and `:210-211` (M2):

| Claim | Recorded | Match |
|---|---|:--:|
| M1 covered AUC ~0.8169 | 0.816886 | yes |
| M1 uncovered AUC ~0.7645 | 0.764525 | yes |
| M1 covered Brier ~0.0749 | 0.074945 | yes |
| M1 uncovered Brier ~0.1088 | 0.108818 | yes |
| M2 covered MAE ~9.21 | 9.210392 | yes |
| M2 uncovered MAE ~11.35 | 11.347178 | yes |
| M2 covered RMSE ~12.41 | 12.410075 | yes |
| M2 uncovered RMSE ~14.84 | 14.844096 | yes |

Sources: M1 = frozen seed-42 `baseline_41` binary
`models/runs/2026-07-26_1551__baseline-41-gpa-trend-control/m1_pass_model.lgbm`;
M2 = frozen seed-42 `concurrent_43` binary
`models/runs/2026-07-27_1327__seed42-concurrent-43-drop-dead-missing-flag/m2_grade_model.lgbm`.
n = 129,215 covered / 26,882 uncovered. **Note the base rates:** fail rate is
9.54% covered vs 14.19% uncovered, and mean final mark 69.77 vs 68.32 — the
covered/uncovered gap is partly a population difference, as that report states at
line 220. *(No model was loaded by this task; these are the recorded numbers.)*

**4. Reporting threshold recorded per metrics file.** Scanned all 40
`models/runs/*/metrics.json`:

| Group | Runs | `run_settings.reporting_threshold` |
|---|---:|---|
| All 2026-07-26 and 2026-07-27 runs (the frozen-version era) | 32 | **0.8**, recorded in three places: `run_settings`, `m1_pass_classifier.train`, `m1_pass_classifier.valid` |
| Legacy runs 2026-07-12 → 2026-07-21 | **8** | **ABSENT — no threshold field of any kind** |

**Flagged: 8 files do not record which threshold produced their confusion matrix**
— `2026-07-12_1208__baseline-39f`, `…__02`, `2026-07-12_1215__add-diploma-signals`,
`2026-07-12_1513__remove-dead-const`, `2026-07-16_1008__new-difficulty-logic`,
`2026-07-16_1025__new-difficulty-logic`,
`2026-07-16_1439__new-difficulty-logic-0-85`,
`2026-07-21_1224__gpa-trend-feature`. Their P/R/F1/confusion figures are not
comparable to the 0.80 era, consistent with CLAUDE.md §6. Two further run
directories, `2026-07-16_1424__new-difficulty-logic-0-85` and
`2026-07-16_1433__new-difficulty-logic-0-85`, contain **no `metrics.json` at all**.

---

## Q8 — TEST access ledger

**1. The reconciliation script's declaration — CONFIRMED from code.**

```python
69   EXTENDED_PATH = ROOT / "data" / "final" / "without_outliers.parquet"
70   # Only these columns are ever loaded from the extended file (see module docstring).
71   EXTENDED_COLUMNS = ["course_id", "part_id", "degree_id", "student_id"]
…
162      ext = pd.read_parquet(EXTENDED_PATH, columns=EXTENDED_COLUMNS)
```
— [`scripts/course_identity_reconciliation_2025.py:69-71, 162`](../../scripts/course_identity_reconciliation_2025.py#L69-L71)

`EXTENDED_COLUMNS` is the only column list passed to that read; there is no second
read of that path. `final_mark` appears nowhere in the file's extended-history path.
The docstring (lines 20-37) and the report (`COURSE_IDENTITY_RECONCILIATION_2025.md`
§0, lines 27-37) both declare the conflict rather than hiding it, and the report
records coverage `20051 → 20252`, 727,852 rows, 4 columns loaded.

**2. Every script that reads a TEST-window file.**

| Script | File | Columns loaded | Outcome column? | Could an artifact carry TEST labels? |
|---|---|---|---|---|
| `scripts/course_identity_reconciliation_2025.py:162` | `data/final/without_outliers.parquet` (20051→20252) | `course_id, part_id, degree_id, student_id` | **no** | no — only per-course enrolment counts derived |
| `scripts/parity_check.py:239` | `final/without_outliers.parquet` | **none** — SHA-256 of the file only, as a drift sentinel | no | no |
| `scripts/diagnose_failure_thresholds.py:41` | `--test` **default** is `data/model_data/df_test_final.parquet` | would load TEST if run | **would be yes** | **latent risk** — the default points at TEST; no artifact from it is in the frozen-version era |

No other script in the repository reads a file spanning the TEST window. All eight
report-generation and rescore scripts assert TEST closure
(`generate_r2_confirmation_report.py:1050`,
`generate_regularization_screening_report.py:921`: `assert not
provenance["test_path_exists"]`), and every persistent training run passed a
nonexistent `--test` path.

**But the ledger is not empty of TEST outcomes.** Eight legacy run directories
contain `metrics.json` files with **populated `test` blocks holding real TEST
metrics**:

| Run | M1 TEST AUC | M2 TEST MAE |
|---|---|---|
| `2026-07-12_1208__baseline-39f` (and `__02`) | 0.7692 | 10.6514 |
| `2026-07-12_1215__add-diploma-signals` | 0.7722 | 10.5282 |
| `2026-07-12_1513__remove-dead-const` | 0.7722 | 10.5282 |
| `2026-07-16_1008__new-difficulty-logic` | 0.7880 | 10.3434 |
| `2026-07-16_1025__new-difficulty-logic` | 0.7880 | 10.3434 |
| `2026-07-16_1439__new-difficulty-logic-0-85` | 0.7880 | 10.3434 |
| `2026-07-21_1224__gpa-trend-feature` | 0.7907 | 10.3404 |

These predate the `closed_not_read` policy and a different dataset generation, but
they are committed artifacts in `models/runs/`, and they are TEST-outcome-derived.

**3. Is "TEST remains closed_not_read" literally accurate?** **No.** It is accurate
in the sense the project actually means — no TEST outcome has informed any decision
in the current modeling phase — but as a literal statement about the repository it
is false on two counts. The accurate restatement is:

> **TEST outcomes have never been read in the current (frozen-version) modeling
> phase and have never informed any decision in it; TEST-window enrolment
> *metadata* was read once — `course_id, part_id, degree_id, student_id` from
> `data/final/without_outliers.parquet` — declared and contained; and seven
> historical pre-policy runs (2026-07-12 → 2026-07-21, an earlier dataset
> generation) hold TEST metrics in their committed `metrics.json`.**

The prompt's own proposed restatement ("TEST outcomes never read; TEST-window
enrolment metadata read once, declared and contained") is correct for the
reconciliation but omits the legacy `metrics.json` TEST blocks.

---

## Claims in the prompt that the repository evidence CONTRADICTS

1. **"Do the 67 in B contain all 5 confirmed_equivalent from A?"** — the framing
   presumes containment. They contain **4 of 5**. `1201.111` is `unresolved` in B.
   (Q2.3)
2. **"Is B a later reconciliation of A?"** — B is **not** a reconciliation of A. It
   is an independent classification that disagrees with A on the top predecessor
   for **67 of 182** courses and uses a different rule and taxonomy. The actual
   reconciliation of A is a third artifact
   (`COURSE_IDENTITY_RECONCILIATION_2025.md`, 13/74/7/88). (Q2.2)
3. **"If the sets came from enrolment only, a shared degree could be missed
   whenever the pairing is catalog-level."** — half contradicted. The sets came
   from the **catalog**, not enrolment
   (`course_identity_67_degree_verification.py:551-557`), so the named mechanism
   did not occur. But the underlying worry — that the answer depends on how the
   sets were assembled — is **correct**: the mirror-image omission did occur.
   Catalog-only sets miss enrolment-only degree memberships, and adding them takes
   the result from `0/67` to `12/67`. (Q1.3)
4. **"The five-seed R2 covered/uncovered rescore … The M2 five-seed experiment
   concluding R2 is not adopted"** — the first exists; the second **does not exist
   as a standalone experiment**. M2's R2 result is a secondary section inside the
   M1 confirmation report, verdict `HARMED_WITHIN_NOISE`; non-adoption for M2 is a
   decision sentence, not an experiment. (Q7.2)
5. **"`scripts/course_identity_reconciliation_2025.py` … without touching
   `df_test_final.parquet`"** — true for that script, but the implied conclusion
   that TEST is untouched repo-wide is contradicted by **7 legacy runs with
   populated TEST metric blocks** and by
   `scripts/diagnose_failure_thresholds.py:41`, whose `--test` default points
   directly at `df_test_final.parquet`. (Q8)
6. **"the largest cases span 10 to 23 new degree IDs"** — understated on the old
   side: the largest **old**-side span is **29** degrees (`967.111`, `1021.111`,
   `1019.111`, `1018.111`, `1015.111`). New-side max 23 is correct. (Q4)

Nothing found contradicts a locked decision. M1 = `baseline_41` /
`num_leaves=127`, R2 rejected for M1, M2 = `concurrent_43` / `num_leaves=127`,
`concurrent_44` archived — all four are confirmed by the artifacts read here and
none is disturbed.

---

## Claims that cannot be verified because the producing artifact is missing

| Claim | Why unverifiable | What would be needed |
|---|---|---|
| Classification A's candidate CSV as an **on-disk** artifact | `models/runs/course_identity_candidates.csv` was overwritten by commit `c6a9656` — on Windows it is the same path as `COURSE_IDENTITY_CANDIDATES.csv` | nothing new: the content is intact in git at `0a9f346` and was recovered from there for this report. To make it usable again, one of the two files needs a genuinely distinct name |
| Re-running `scripts/course_identity_reconciliation_2025.py` to reproduce its numbers | its `PRIOR_CSV` (line 75) now resolves to classification **B**'s CSV, so a re-run would read the wrong input | rename the case-colliding file, or pin `PRIOR_CSV` to a git blob |
| The strict verification's re-runnability, **demonstrated** rather than asserted | `main()` writes five already-committed artifacts, outside this task's write allowance | explicit approval to re-run and byte-compare, or a `--dry-run` flag |
| Reporting threshold for 8 legacy runs | the field is absent from their `metrics.json`; no other record states it | re-derive from each run's saved binary + VALID predictions, which requires loading models — out of scope here |
| Any metrics for `2026-07-16_1424__…` and `2026-07-16_1433__…` | no `metrics.json` exists in either directory | re-running those runs; not proposed |
| Inherited Level-1 coverage 93.6% / 76.2% (CLAUDE.md §9) | **already corrected on record** — the recomputation gives 94.37% / 77.42% (`DIFFICULTY_COVERAGE_DIAGNOSTIC.md:85-88`). Not re-litigated here | — |
| The 44.7% Level-1 TEST coverage figure | would require reading TEST | not obtainable under the current policy |

---

## Scope confirmations

- TEST parquet: **not read**, not globbed, not stat-ed, not path-constructed.
- VALID outcome columns: **not loaded** — `final_mark` is absent from
  `VALID_COLUMNS` and its absence is asserted at runtime.
- No pair was ranked, filtered or selected using anything derived from a VALID
  label.
- Models: **0 loaded, 0 trained, 0 rescored**. Q7's figures are quoted from
  existing report artifacts.
- Datasets, parquets, mappings, defaults, `CURRENT_VERSION.txt`, promotion
  markers, `src/`: **unchanged**.
- Locked decisions: **unchanged**. No contradiction found.
- Nothing was committed and nothing was pushed.

**Committing note.** `.gitignore:46-52` un-ignores `models/runs/*.json` and
`models/runs/*.md` at the top level but **not** `models/runs/*.csv` (CSVs are only
un-ignored one level deeper, inside run folders). So this report and the script
appear as untracked normally, while
`models/runs/phase0_pair_divergence.csv` needs `git add -f`, as the existing
top-level `COURSE_IDENTITY_*.csv` artifacts evidently did.
