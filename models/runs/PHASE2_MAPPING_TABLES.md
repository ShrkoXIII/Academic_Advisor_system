# Phase 2 (REVISED) — proposal tables

**Status: READ-ONLY. Nothing was trained, rescored, mapped, rebuilt or promoted.
All four proposal tables were generated and validated; one significant, newly
discovered coverage gap is documented below rather than silently patched.**

| | |
|---|---|
| Frozen dataset version | `2026-07-26_batched_fixes__registration_roster_concurrent` |
| Splits read | `df_train_final.parquet`, `df_valid_final.parquet` |
| TEST parquet | not read, not globbed, not stat-ed, not path-constructed |
| VALID outcome columns | `final_mark` never loaded from VALID; asserted at runtime in `scripts/phase2_mapping_tables.py::load_valid` |
| Models loaded / trained / rescored | 0 / 0 / 0 |
| `src.course_difficulty` | imported READ-ONLY, once, only inside `validate_course_history_count` (reuses `fit_difficulty_state`/`apply_difficulty_state` — the same method and result Phase 0 already validated at 0/156,097) |
| Artifacts written | `models/runs/PHASE2_MAPPING_TABLES.md`, `models/runs/course_split_candidates.csv`, `models/runs/degree_lineage_proposed.csv`, `models/runs/course_link_proposed.csv`, `models/runs/course_difficulty_stats_prototype.csv`, `scripts/phase2_mapping_tables.py` |
| Every proposal row | `approval_status = "pending"` throughout; no row was written with any other value |

---

## Normalization gate

**66 of 67, and the sole non-match is exactly the pair the revised task
predicted: `510.111` (`بنيان الحواسيب`) ↔ `1183.111` (`بنيان الحواسيب1`).**

```
pairs_total            : 67
matched                : 66  (expected: 66)
mismatches             : 1   (expected: 1)
non-match is 510.111/1183.111 : True
gate_pass              : True
```

This is now understood correctly: it is not a normalization defect, it is a
**split** (Table 0), confirmed below.

---

## Table 0 — split and merge candidates

**5 split candidates, 4 merge candidates**, detected catalog-wide (no TRAIN or
VALID rows used for detection itself) using the `course_id` numeric proxy
boundary `1150` (same boundary Phase 1 Q7 already used to separate "old" from
"recent" catalog courses) and connected-component clustering by shared
`degree_id`.

| Direction | Stem | Old course(s) | Old credits | New course(s) | New credits | Credit Δ | Generations |
|---|---|---|---|---|---:|---:|---:|
| split | بنيان حواسيب | `510.111` بنيان الحواسيب | 3.0 | `1183.111\|1192.111` بنيان الحواسيب1/2 | 6.0 | **+3.0** | 2 |
| split | بنيان حواسيب | `510.111` بنيان الحواسيب | 3.0 | `1451.111\|1454.111` بنيان الحواسيب1/2 | 6.0 | +3.0 | 2 |
| split | دارات كهرباييه | `384.111` دارات كهربائية | 3.0 | `1180.111\|1220.111` الدارات الكهربائية1/2 | 6.0 | +3.0 | 2 |
| split | دارات كهرباييه | `384.111` دارات كهربائية | 3.0 | `1447.111\|1539.111` الدارات الكهربائية1/2 | 6.0 | +3.0 | 2 |
| split | محاسبه تكاليف | `179.111` محاسبة تكاليف | 4.0 | `1380.111\|1381.111` محاسبة تكاليف1/2 | 6.0 | +2.0 | 1 |
| merge | دارات الكترونيه | `430.111\|459.111` الدارات الإلكترونية 1/2 | 6.0 | `1586.111` الدارات الإلكترونية | 3.0 | −3.0 | 1 |
| merge | كيمياء حيويه | `209.111\|210.111` كيمياء حيوية (1)/(2) | 6.0 | `1578.111` الكيمياء الحيوية | 3.0 | −3.0 | 2 |
| merge | كيمياء حيويه | `1110.111\|1115.111` الكيمياء الحيوية 1/2 | 6.0 | `1578.111` الكيمياء الحيوية | 3.0 | −3.0 | 2 |
| merge | مهارات حاسوب | `150.111\|173.111` مهارات الحاسوب 1/2 | 7.0 | `1422.111` مهارات الحاسوب | 3.0 | −4.0 | 1 |

**`بنيان الحواسيب` confirmation (the stop condition for this task):**
`old_course_ids = 510.111`, `new_course_ids = 1183.111|1192.111`,
`credit_change = +3.0` — matches exactly. Gate passes; the task did not stop.

**Notable structural pattern beyond the single course the revision named**:
the same 2-generation split shape (a single 3-credit old course splitting
into two 3-credit new courses under the AI-family degrees 57–61/64 as a
*second* generation) recurs for `بنيان حواسيب` and `دارات كهرباييه` —
both split once for the 26–31/45–49 degree cohort and again, independently,
for the 57–61/64 cohort. Neither second-generation pair
(`1451.111|1454.111`, `1447.111|1539.111`) carries any VALID rows yet, so
they don't affect Table 2 coverage today, but they are real catalog facts a
reviewer should see.

**A finding surfaced while building Table 0, not asked for but material**:
`مهارات حاسوب` (`1422.111`) is flagged `merged_from` `{150.111, 173.111}`
(support 192+89, one old degree, `19.111`), by the letter of the merge rule
(≥2 old courses with level suffixes → 1 new course without one). But the
*name_stem* also has a **third**, un-suffixed old course, `967.111`
(train_support **9,254**, catalogued under 29 old degrees) — the actual
service-course predecessor by any reasonable reading, and the one the human
review file (`COURSE_IDENTITY_67_HUMAN_REVIEW.csv`) pairs `1422.111` with.
The merge rule as specified only looks at *suffixed* old courses, so it
cannot see `967.111`, and Table 2's exclusion rule ("any course in Table 0 is
excluded from ordinary matching") then hides the dominant real predecessor
entirely — see the Table 2 validation section below. This is not a coding
defect; it is a real gap in Table 0's detection rule for a mixed
suffixed/un-suffixed predecessor pool.

---

## Table 1 — degree lineage

**23 new degrees** (`degree_id` numeric ≥ 40), **35 old-degree candidates**,
top 3 ranked per new degree by `overlap_pct_of_new` (desc), `jaccard` (desc),
`old_degree_id` (asc).

Rank-1 `auto_suggestion` counts:

| STRONG | PLAUSIBLE | WEAK | NONE |
|---:|---:|---:|---:|
| 16 | 5 | 1 | 1 |

This reproduces the task's own prior-measurement expectations exactly: the
16 `STRONG` cases are the 16 literal-`2023` degrees at 75–89% overlap; every
non-`STRONG` case is listed below.

| New degree | Old (rank 1) | overlap_of_new | overlap_of_old | jaccard | name_sim | has_2023 | same_family | suggestion |
|---|---|---:|---:|---:|---:|---|---|---|
| `57.111` هندسة الذكاء الاصطناعي/هندسة البرمجيات... | `26.111` الهندسة المعلوماتية/هندسة البرمجيات... | 0.701 | — | — | — | False | False | PLAUSIBLE |
| `58.111` هندسة الذكاء الاصطناعي/هندسة الذكاء... | `27.111` الهندسة المعلوماتية/هندسة الذكاء... | 0.649 | — | — | — | False | False | PLAUSIBLE |
| `59.111` هندسة الذكاء الاصطناعي/هندسة نظم... | `29.111` الهندسة المعلوماتية/هندسة أمن... | 0.662 | — | — | — | False | False | PLAUSIBLE |
| `60.111` هندسة الذكاء الاصطناعي/هندسة التحكم... | `30.111` هندسة التحكم والروبوت | 0.649 | — | — | — | False | False | PLAUSIBLE |
| `61.111` هندسة الذكاء الاصطناعي/هندسة الاتصالات | `31.111` هندسة الاتصالات | 0.675 | — | — | — | False | False | PLAUSIBLE |
| `64.111` هندسة الذكاء الاصطناعي/الهندسة... | `27.111` الهندسة المعلوماتية/هندسة الذكاء... | 0.377 | — | — | — | False | False | **WEAK** |
| `65.111` هندسة تكنولوجيا البناء والتشييد | `26.111` الهندسة المعلوماتية/هندسة البرمجيات... | 0.153 | — | — | — | False | False | **NONE** |

(Exact `overlap_pct_of_old`, `jaccard`, `degree_name_similarity` per row are
in `degree_lineage_proposed.csv`; omitted here only for table width.)

Matches the task's stated expectation precisely: 5 AI-family degrees (57–61)
land `PLAUSIBLE` at 65–70%, `64.111` lands `WEAK` at 37.7%, and `65.111` —
correctly, per the task's own framing ("a genuinely new programme") —
lands `NONE` at 15.3%.

**A structural point worth flagging directly, because it drives the Table 2
finding below**: three of these seven rank-1 candidates (`26.111`, `27.111`,
`29.111`, `30.111`, `31.111` across the five rows) are themselves
**catalog degrees with `degree_id` numeric < 40** — i.e. under this task's own
Table 1 rule they are *not* "new degrees" and never receive a Table 1 row of
their own, yet they are independently confirmed (Phase 0 Q5) to be
VALID-only, TRAIN-absent degrees — genuinely new-generation programmes that
simply weren't given a ≥40 id. See "What the human must decide," item 1.

---

## Table 2 — course link

**Census: all 182 never-in-TRAIN VALID course IDs present, including `none` rows.**

Scope classification:

| Scope | New courses |
|---|---:|
| `shared` (≥5 degree families) | 33 |
| `specific` (<5 degree families) | 142 |
| `split_or_merge` (Table 0) | 7 |
| **Total** | **182** |

Relationship classification (one row per new course, taking its
`relationship_type`; `consolidated_into`/`successor` rows expand to one row
per predecessor):

| Relationship | New courses | VALID rows covered |
|---|---:|---:|
| `none` | 149 | 17,285 |
| `successor` | 15 | 1,950 |
| `consolidated_into` | 11 | 4,145 |
| `split_from` | 6 | 1,561 |
| `merged_from` | 1 | 686 |
| **Total** | **182** | **25,627** |

Covered (anything but `none`): **8,342 / 25,627 = 32.6%**.

### Validation 1 — do all 67 human-reviewed pairs appear?

**No — 66 of 67 appear correctly (including the split pair, as
`split_from`); 46 do not appear as the reviewed pair.** This was fully
root-caused, not just counted:

| Root cause | Pairs | What it means |
|---|---:|---|
| **A — "specific"-scope lineage gap** | **42** | See below. The single largest, systemic finding of this task. |
| **B — Table 0 merge exclusion side effect** | **1** | `1422.111` ↔ `967.111`. Table 0 correctly (per its literal rule) claims `1422.111` as `merged_from {150.111, 173.111}`; Table 2 then excludes it from ordinary search, so the far larger true predecessor `967.111` (train_support 9,254) never appears for it. See Table 0's finding above. |
| **C — old course below `train_support ≥ 20`** | **2** | `1255.111` ↔ `445.111` (support 3), `1212.111` ↔ `448.111` (support 1). **Correctly excluded, not a defect** — matches Phase 0 Q6's own observation about these exact two pairs ("Two pairs rest on almost nothing"). |
| **D — atypical old-degree placement** | **1** | `1419.111` ↔ `49.111` ("نقود ومصارف" / Money & Banking). Name key matches exactly and `train_support = 48` (eligible), but the course is catalogued only under old degree `10.111`, not under the finance family's usual old degree (`7.111`) that `37.111`/`52.111`'s lineage resolves to. A single-course exception no generic scope rule reaches. |

**Root cause A, in full.** Table 1's own rule (`degree_id` numeric ≥ 40) is
correct and its output matches the task's own expected numbers exactly (see
Table 1 above) — but it is not a complete list of "new" degrees. Phase 0 Q5
independently established, from TRAIN/VALID degree-set membership alone (no
heuristic), that there are **25** VALID-only new-generation degrees; only 14
of those have `degree_id` ≥ 40. The other **11** — `26.111`, `27.111`,
`29.111`, `30.111`, `31.111`, `33.111`, `34.111`, `35.111`, `36.111`,
`37.111`, `39.111` — are genuinely new (zero TRAIN presence) but numerically
under 40, so Table 1 never builds a lineage row for them.

For a `specific`-scope course listed under one of these 11 degrees (directly,
or reached transitively through a `2023`-suffixed sibling whose Table 1
rank-1 candidate happens to *be* one of these 11 — e.g. `45.111`'s rank-1 is
`30.111`, itself one of the 11), Table 2's search scope resolution has
nothing to fall back on: the spec's own designed fallback ("if Table 1 gives
no candidate for the relevant degree, fall back to same-family search") was
implemented here to trigger for *both* "no Table 1 row exists" and
"Table 1's row carries zero overlap" (a necessary broadening of the literal
wording, documented in the script), but the family-text-equality fallback
itself still can't bridge these cases — e.g. `degree_family(26.111)` =
`"هندسه معلوماتيه"` (an umbrella family name) has no exact string match
among the true historical predecessor degrees `21.111`/`3.111`/`4.111`/
`5.111`/`6.111` (each a specific old specialisation name). The name keys
themselves match perfectly in every one of these 42 pairs (verified
individually, e.g. `1175.111`/`502.111` both normalize to
`تحليل رياضي#1`) — **this is purely a search-scope problem, not a
normalization problem.**

**Consequence, stated plainly**: this task's Table 2, run exactly as
specified, correctly finds the 25 pairs whose search scope happens to
resolve, and correctly fails to find the other 42 — not because the
predecessors don't exist or don't match by name, but because the degree
graph needed to find them was never given a way to reach the true
1930s-vintage TRAIN degrees for these 11 "new-but-<40" degrees. Fixing this
requires changing which degrees get a Table 1 row in the first place (see
"What the human must decide," item 1) — a change to a rule the task defined
explicitly, which this report flags rather than makes.

### Validation 2 — `1423.111`

`1423.111` (اللغة الإنكليزية – 1) is `shared` + `consolidated_into`, as
expected:

| Old course | train_support | Qualifies (≥20) | Weight |
|---|---:|:--:|---:|
| `151.111` | 229 | yes | 0.02884 |
| `221.111` | 241 | yes | 0.03035 |
| `391.111` | 149 | yes | 0.01876 |
| `830.111` | 363 | yes | 0.04571 |
| `893.111` | **2** | **no** | — |

**4 of the 5 named service courses qualify** (`893.111` does not —
train_support 2, well under the 20 floor). The other 7 predecessors in this
course's `consolidated_into` group (11 total) come from the remaining old
faculty-specific English-1 offerings this task's background section didn't
enumerate by name; all weights across the 11-member group sum to 1.0 (see
Validation 4).

### Validation 3 — every new course present

**True.** All 182 appear at least once (149 as a single `none` row; 33 as
one or more predecessor rows).

### Validation 4 — weight sums

**True.** 11 `consolidated_into` groups; every group's `weight_hint` values
sum to 1.0 within 1e-6 (asserted in code — the script `SystemExit`s if not).

---

## Table 3 — difficulty stats prototype

**47,685 rows.** Scope: 46 unique old-generation predecessor course IDs drawn
from Table 2's `old_course_id` column (of the 52 distinct predecessor course
IDs total, 46 also occur in TRAIN and are usable for this walk) plus a
`random_state=42` sample of 200 further TRAIN course IDs — 252 courses in
total, each walked across the 55 distinct TRAIN `part_id` semesters at both
Level 1 (degree+course) and Level 2 (course-only, `degree_id = "ALL"`).

`link_used` and `link_weight` are `null` in **every** row (checked: the
columns are constructed as literal `None` for the whole frame, not
conditionally) — the prototype carries the schema, it applies nothing.

**`course_history_count` validation: 0 mismatches over 156,097 VALID rows**,
reusing `fit_difficulty_state`/`apply_difficulty_state` from
`src/course_difficulty.py` (read-only) fit on the complete TRAIN split — the
exact method and exact result Phase 0 already established. Matches the
required bar exactly.

---

## What changed versus Phase 1

| | Phase 1 (name-key layer, threshold 20) | Phase 2 (degree-lineage-scoped) |
|---|---:|---:|
| Unrestricted narrow+wide coverage | 17,814 / 25,627 = 69.5% | n/a — Phase 2 has no "unrestricted" mode |
| Service-course-excluded coverage | 13,390 / 25,627 = 52.3% | n/a |
| **This task's coverage** (anything but `none`) | — | **8,342 / 25,627 = 32.6%** |
| Mechanism | Global name-key search, then a post-hoc HIGH/MEDIUM-risk service-course filter | Degree-family-scoped search *before* matching (shared vs specific), plus explicit split/merge detection |

Phase 2's coverage is **lower** than Phase 1's service-excluded figure, and
that is fully explained by root cause A above: Phase 1 searched the *entire*
old catalog unconditionally for every course, so it always found `502.111`
for `1175.111` etc. regardless of degree scope. Phase 2 deliberately
restricts `specific`-scope courses to a degree-lineage-derived search scope
(a real improvement in precision — it stops a "specific" course from
matching an old course under an unrelated degree just because the names
happen to coincide) — but that precision is currently undermined by Table 1
only covering 14 of the 25 real new-generation degrees. The `shared`-scope
courses (33 of 182, unrestricted global search, same mechanism as Phase 1)
are not affected by this gap at all.

---

## What the human must decide before Phase 3

1. **Table 1's "new degree" rule (`degree_id` numeric ≥ 40) is incomplete.**
   Phase 0 Q5 already establishes the correct set — 25 VALID-only degrees,
   not 23. Extending Table 1 to build a lineage row for all 25 (using the
   same overlap computation, just against a bigger `new_degrees` list) would
   very likely close most of the 42 root-cause-A pairs, since the missing
   11 degrees (`26.111`, `27.111`, `29.111`, `30.111`, `31.111`, `33.111`,
   `34.111`, `35.111`, `36.111`, `37.111`, `39.111`) are exactly the ones a
   `2023`-suffixed sibling's rank-1 candidate keeps resolving to. This is a
   change to a rule the task defined explicitly, so it is not made here —
   it needs your approval, and re-running Table 1 first, then Table 2.
2. **`1422.111`'s Table 0 merge classification hides its dominant real
   predecessor.** Decide whether `967.111` (9,254 TRAIN rows) should be
   added to that merge group manually, or whether Table 0's merge rule
   should be broadened to also pull in a coexisting un-suffixed old course
   sharing the same `name_stem`.
3. **The two second-generation splits with zero current VALID rows**
   (`1451.111|1454.111` for بنيان حواسيب, `1447.111|1539.111` for
   دارات كهرباييه) — confirm they should be treated identically to their
   first-generation counterparts once they do accumulate VALID rows.
4. **`1419.111` ↔ `49.111`** — a single-course exception (course catalogued
   under an atypical old degree). Decide whether this is worth a manual
   override row or is acceptable as a `none` result.
5. **`893.111`** (one of the five named English-1 predecessors) has
   TRAIN support of only 2 and is correctly excluded from `1423.111`'s
   `consolidated_into` group — confirm this is the intended behavior of the
   `train_support ≥ 20` floor, not a data problem to fix upstream.
6. Standard sign-off items, unchanged from the original task framing: review
   and approve/reject rows in all three "proposed"/"candidate" tables
   (`review_decision` columns are empty by design); nothing here is wired
   into the feature pipeline, and Phase 3 (applying any link) is not
   proposed by this report.
