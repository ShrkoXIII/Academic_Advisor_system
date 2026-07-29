# Phase 1 — name-key layer measurement

**Status: READ-ONLY. Nothing was trained, rescored, mapped, rebuilt or promoted.**

| | |
|---|---|
| Frozen dataset version | `2026-07-26_batched_fixes__registration_roster_concurrent` |
| Splits read | `df_train_final.parquet`, `df_valid_final.parquet` |
| TEST parquet | not read, not globbed, not stat-ed, not path-constructed |
| VALID outcome columns | `final_mark` never loaded (asserted at runtime in `scripts/phase1_name_key_layer.py::load_valid`) |
| Models loaded / trained / rescored | 0 / 0 / 0 |
| `src.course_difficulty` | not imported — every statistic here (support counts, pass rates, avg marks) is an independent recomputation from raw TRAIN rows, not the shrinkage-smoothed difficulty state |
| Artifacts written | `models/runs/PHASE1_NAME_KEY_REPORT.md`, `models/runs/phase1_name_key_per_course.csv`, `scripts/phase1_name_key_layer.py` |

**Two interpretive gaps in the task spec, found and resolved during implementation (documented here, not silently patched):**

1. **Neither `df_train_final.parquet` nor `df_valid_final.parquet` contains a `course_name_sl` column** (confirmed against the on-disk parquet schema). The task's "name source rule" (catalog name, else TRAIN row name) therefore has no second branch to fall back to — every name in this report is catalog-sourced or absent. This is stated as fact, not assumed.
2. **`requirement_type_id` is not course-invariant**: 35 of 1,503 catalog courses and 58 of 811 TRAIN courses carry more than one `requirement_type_id` across the degrees they belong to (elective in one degree, required in another). The spec's per-course narrow key needs one `requirement_type_id`; the **modal** value (most frequent among that course's rows, ties broken toward the smaller id) is used. `course_credits` needed no such treatment — it is unique per `course_id` in the catalog (0 of 1,503 exceptions) and matches TRAIN row-level credits exactly everywhere checked (0 of 811 mismatches).

A third choice, not a gap: **"Level 2, TRAIN-only" pass rate / avg mark** (Q2.4, Q4) is computed as a raw, unsmoothed per-`course_id` mean over TRAIN rows — not the shrinkage-smoothed table `src.course_difficulty` would produce for its Level 2. This keeps the whole layer independent of the difficulty module, per the task's instruction not to load the difficulty state.

Full computation: [`scripts/phase1_name_key_layer.py`](../../scripts/phase1_name_key_layer.py). Per-course detail: [`models/runs/phase1_name_key_per_course.csv`](phase1_name_key_per_course.csv) (182 rows, one per never-in-TRAIN VALID course).

---

## Summary verdict

The name-key layer recovers real coverage, but a large share of it is unsafe to use as-is. At `min_support=20` the layer matches 69.5% of the 25,627 never-in-TRAIN VALID rows by *some* key (narrow or wide), and where it matches, the row-weighted estimate movement (0.079 all-matched, mean abs diff in pass rate) exceeds the Phase 0 predecessor-prior reference of 0.0633 — the layer is not a no-op, it changes estimates by a comparable or larger amount than the mechanism it would extend. But 9 of 98 narrow keys and 23 of 87 wide keys spanning ≥5 degree families are HIGH/MEDIUM risk service courses (English, Arabic, Math, Physics, intro Accounting/Management), and they account for the majority of raw narrow coverage: after restricting service-course keys to same-family matches only, **narrow coverage drops from 31.1% to 25.5%** of never-in-TRAIN rows — inside the pre-registered BORDERLINE band, not a clean GO. The other two pre-registered signals pass clearly (row-weighted |Δ pass rate| = 0.060, HIGH_RISK share of covered rows = 17.9%). Verdict: **BORDERLINE — escalate**, driven entirely by signal 1.

---

## Q1 — normalization validation

**66 of 67 pairs have identical normalized names**, not 65. The `ال`-stripping rule in the spec (`re.sub(r'^ال', '', t) if len(t) > 3`) already resolves `علم اجتماع` vs `علم الاجتماع` — both normalize to `علم اجتماع` — so that failure is **closed by the spec as given**, and the expected count moves from 65/67 to **66/67**.

**The one remaining mismatch:**

| old name | new name | old key | new key | rows |
|---|---|---|---|---|
| بنيان الحواسيب | بنيان الحواسيب1 | `بنيان حواسيب` | `بنيان حواسيب#1` | 495 |

This is the digit-suffix case: the new course name appends a bare `1` with no separating space or level marker, so the digit-stripping step correctly pulls out `#1` for the new name but finds no digit at all in the old name. The keys differ by exactly the trailing `#1` token.

Confirmed: `بنيان الحواسيب` / `بنيان الحواسيب1` fails as predicted (495 rows); `علم اجتماع` / `علم الاجتماع` **now matches** — the task's own normalization spec already fixes it, no separate rule addition needed for this one.

---

## Q2 — TRAIN index statistics

**Name coverage.** 811 distinct TRAIN course IDs; **804 catalog-backed, 0 name-from-row** (the column doesn't exist), **7 not catalog-backed at all** (`1029.111`, `146.111`, `393.111`, `551.111`, `555.111`, `557.111`, `769.111`) — these 7 are excluded from the name-key index entirely; no name-key can be built for a course with no name.

**Key cardinality** (804 catalog-backed TRAIN courses):

| | Narrow (name, credits, req_type) | Wide (name only) |
|---|---:|---:|
| Distinct keys | 749 | 666 |
| Multi-course keys | 38 | 95 |
| Key size distribution | 1→711, 2→30, 3→2, 4→4, 5→1, 6→1 | 1→571, 2→70, 3→13, 4→8, 5→2, 6→2 |

Widening from narrow to wide (dropping credits + requirement type) roughly triples the multi-course-key count (38 → 95) — expected, since it pools courses that differ only in credit weight or elective/required status.

**Cost of pooling — pairwise pass-rate spread within multi-course keys** (raw TRAIN-only pass rate, unsmoothed):

| | pairs | mean abs diff | p50 | p90 |
|---|---:|---:|---:|---:|
| Narrow | 85 | 0.159 | 0.117 | 0.372 |
| Wide | 207 | 0.158 | 0.120 | 0.344 |

Both distributions are nearly identical and both are large — a mean pairwise spread of ~0.16 in pass rate is **more than double** the Phase 0 predecessor-prior reference (0.0633). Narrow vs. wide makes almost no difference to this spread, which is the first sign that credits/requirement-type discrimination isn't what's protecting against bad pooling here — degree-family discrimination (Q5) is doing the real work.

---

## Q3 — VALID never-in-TRAIN coverage funnel

182 never-in-TRAIN VALID course IDs cover the full 25,627 rows; **all 182 have a catalog name** (0 courses / 0 rows lost to missing names on the VALID side).

**Threshold 20 (primary, unrestricted):**

```
never-in-TRAIN VALID rows total          : 25,627
  narrow match (support >= 20)           : 7,981 rows  / 52 course IDs
  wide match   (support >= 20)           : 9,833 rows  / 31 course IDs
  no match                               : 7,813 rows  / 99 course IDs
```

Combined narrow+wide coverage: 17,814 / 25,627 = **69.5%**.

**Threshold 50 (secondary):** narrow 7,979 rows / 51 IDs; wide 9,832 rows / 31 IDs; no match 7,816 rows / 100 IDs — almost identical to threshold 20 (one course, one TRAIN match, drops out of the 20–49 support band). The layer's coverage is not sensitive to this particular threshold choice in the 20–50 range.

**Distribution of the best-match TRAIN course's support** (row-weighted, matched rows only, threshold 20): min 47, p25 368, median 984, p75 1,524, max 9,254. Every match rests on real volume — there is no near-zero-support match anywhere in this population.

**Threshold 20, excluding service-course keys** (Q5 exclusion rule applied — see there for the rule):

```
  narrow match, excl. service            : 6,531 rows  / 50 course IDs   (25.5% of 25,627)
  wide match, excl. service              : 6,859 rows  / 26 course IDs   (26.8% of 25,627)
  no match (incl. service exclusions)    : 12,237 rows / 106 course IDs
```

This is the headline drop: unrestricted narrow coverage (31.1%) and the exclusion-restricted narrow coverage (25.5%) differ by 5.6 points, not because most narrow-matched courses are service courses (only 2 of 52 lose their match entirely under exclusion), but because the ones that *are* service courses are disproportionately high-volume (English 1/2, Arabic) — of the 52 narrow-matched courses, 21 lose match under exclusion, carrying 7,539 of the original 7,981 rows.

---

## Q4 — estimate movement (TRAIN-only)

Row-weighted mean absolute difference in pass rate between the current structural estimate (Level 4/5) and the name-key TRAIN estimate:

| Subset | rows | mean | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| All matched (narrow+wide, unrestricted) | 17,814 | **0.079** | 0.058 | 0.106 | 0.192 | 0.364 |
| Narrow only | 7,981 | 0.087 | 0.071 | 0.119 | 0.232 | 0.364 |
| Wide only | 9,833 | 0.073 | 0.056 | 0.083 | 0.163 | 0.273 |
| **Excluding service-course matches** | 13,390 | **0.060** | 0.056 | 0.074 | 0.119 | 0.364 |

All four numbers meet or exceed the Phase 0 reference (0.0633) except the service-excluded figure, which sits just below it (0.060 vs 0.063) — close enough to call comparable, not a clean beat.

**Direction.** Row-weighted signed diff (name-key estimate − current estimate) is **−0.049** (unrestricted, all matched) and **−0.021** (excluding service) — the name-key TRAIN estimate is systematically *harsher* than the current structural estimate, same direction as the Phase 0 predecessor prior (−0.0264), though larger in magnitude before service exclusion.

**Avg mark** (0–100 scale), row-weighted:

| Subset | mean abs diff | signed diff |
|---|---:|---:|
| All matched | 4.55 | −1.85 |
| Narrow only | 5.86 | −2.80 |
| Wide only | 3.48 | −1.08 |
| Excluding service | 3.53 | −0.13 |

Note the near-zero signed diff (−0.13) once service courses are excluded — the harsh bias in avg mark is almost entirely a service-course artifact; outside service courses the name-key estimate is nearly unbiased in mark terms even though pass-rate bias remains.

**Large moves:** unrestricted, 4,516 of 17,814 rows (25.4%) move >0.10 in pass rate, 1,751 (9.8%) move >0.20. Excluding service, 2,307 of 13,390 (17.2%) move >0.10, but only **13** (0.1%) move >0.20 — the extreme tail is almost entirely a service-course phenomenon.

---

## Q5 — service-course risk

**Rule as implemented.** A name key (narrow or wide, checked separately) is flagged if the union of degree families across all TRAIN courses sharing that key spans ≥5 distinct families. Family = catalog `degree_name_sl` with a trailing `2023` token stripped, then only the text before the first `/` kept (dropping the specialisation suffix) — verified against all 58 catalog degree names, every year-suffixed name uses the literal token `2023` and every specialisation name uses `/` as separator. Risk level: spread (max−min raw TRAIN pass rate across the group) `>0.15` → HIGH_RISK, `0.05–0.15` → MEDIUM_RISK, `<0.05` → OK.

A key can span ≥5 families from a **single course** — a course_id in this catalog can itself be offered under many degree_ids (the widest TRAIN course spans 16 families alone) — so most flagged keys are not the multi-course pooling case from Q2; they are individually-ubiquitous courses.

**Counts.** Narrow: 98 keys span ≥5 families, of which 5 HIGH_RISK + 4 MEDIUM_RISK = **9 require exclusion**, 89 are OK (safe despite the family span, spread <0.05). Wide: 87 keys span ≥5 families, 15 HIGH_RISK + 8 MEDIUM_RISK = **23 require exclusion**, 64 OK.

**HIGH_RISK and MEDIUM_RISK narrow keys** (9, all with any never-in-TRAIN VALID rows covered):

| Key (name, credits, req_type) | Families | Spread | Risk | VALID rows covered |
|---|---:|---:|---|---:|
| لغه انكليزيه#1, 3, 1 | 13 | 0.405 | HIGH | 979 |
| لغه انكليزيه#2, 3, 1 | 12 | 0.198 | HIGH | 471 |
| لغه انكليزيه#3, 3, 1 | 8 | 0.226 | HIGH | 0 |
| لغه عربيه, 2, 1 | 18 | 0.627 | HIGH | 0 |
| رياضيات#1, 3, 5 | 7 | 0.312 | HIGH | 0 |
| امن نظم معلومات, 3, 5 | 6 | 0.127 | MEDIUM | 0 |
| رياضيات#2, 3, 5 | 7 | 0.117 | MEDIUM | 0 |
| فيزياء#1, 3, 5 | 7 | 0.098 | MEDIUM | 0 |
| فيزياء#2, 3, 5 | 7 | 0.114 | MEDIUM | 0 |

**HIGH_RISK and MEDIUM_RISK wide keys** (23; showing the ones with nonzero coverage — the rest cover 0 never-in-TRAIN rows because those name-only keys' member courses already have TRAIN history under some credits/req_type combination):

| Key (name only) | Families | Spread | Risk | VALID rows covered |
|---|---:|---:|---|---:|
| فيزياء#2 | 8 | 0.291 | HIGH | 759 |
| تحليل رياضي#2 | 6 | 0.212 | HIGH | 688 |
| مبادي محاسبه#1 | 6 | 0.102 | MEDIUM | 656 |
| مبادي اداره | 6 | 0.083 | MEDIUM | 579 |
| مبادي محاسبه#2 | 6 | 0.229 | HIGH | 292 |

(18 further wide keys — إحصاء والاحتمالات, إدارة مشاريع, إدارة موارد بشرية, أمن نظم معلومات, بحوث عمليات, تفكير علمي, رياضيات#1/#2, علم نفس سلوكي, فيزياء#1, قواعد بيانات#1, لغة إنكليزية#1/#2/#3, لغة عربية, مدخل إلى قانون, مهارات تواصل, مهارات حاسوب — cover 0 never-in-TRAIN rows.)

**High-risk share of covered rows** (unrestricted threshold-20 coverage, HIGH_RISK only, narrow+wide combined): 3,189 / 17,814 = **17.9%** — comfortably under the 30% GO bound.

**Q3/Q4 recomputed excluding service (side by side with all):**

| | All (unrestricted) | Excluding service |
|---|---:|---:|
| Narrow rows covered | 7,981 (31.1%) | 6,531 (25.5%) |
| Wide rows covered | 9,833 (38.4%) | 6,859 (26.8%) |
| Row-weighted mean abs diff pass rate (all matched) | 0.079 | 0.060 |

---

## Q6 — the two unmatched pairs

**1. بنيان الحواسيب → بنيان الحواسيب1 (495 VALID rows).**

- Fuzzy help: yes, trivially — `بنيان الحواسيب1` is exactly `بنيان الحواسيب` (old name) with a bare `1` appended, no separator. `old_key.startswith` is not even needed; `new_key == old_key + "#1"` holds by construction here since `#1` is exactly what the spec's digit-extraction step produces from that trailing `1`.
- New course's own VALID row count: 495 (matches the review CSV exactly).
- Old course's TRAIN support (raw L2): **1,165** rows.
- Pass rate: old (TRAIN, raw) = 0.776; new (VALID, current structural estimate, mean) = 0.839. Diff (old − new) = **−0.063**.
- Confirmed independently: this course gets **no match at all** from the general name-key layer (`match_type = none` in the CSV) — the digit-suffix failure isn't just a Q1 artifact, it silently costs this specific 495-row course its only viable TRAIN match.
- **Proposed minimal fix:** when the normalized name ends in a bare digit token with no preceding separator (i.e., a level number was concatenated directly onto the name with no space, hyphen, or dash), extract it as the level token the same way the spec already does for names that use a separator — in effect, treat `<name><digit>` identically to `<name> <digit>` before the digit-stripping step. One line: insert a space before a trailing digit run that isn't already separated by whitespace, prior to the existing `digits = re.findall(...)` step.

**2. علم اجتماع → علم الاجتماع (87 rows).**

- The `ال`-stripping rule **already handles this** — confirmed `yes`. Both names tokenize to `["علم", "اجتماع"]` after `ال` is stripped from the definite-article form (`الاجتماع` → `اجتماع`, since the token length after stripping is >3 chars, satisfying the rule's `len(t) > 3` gate on the pre-strip token `الاجتماع`), and both keys equal `علم اجتماع`. No fix needed — this pair was miscounted as a failure in the task's own background material; Q1 already reflects the corrected 66/67.
- (Diagnostic side note, not part of the fix: this pair's old course has TRAIN support 2,831 and raw pass rate 0.828 vs. the new course's current VALID pass rate 0.895 — diff −0.067, similar magnitude to the بنيان الحواسيب pair.)

---

## Q7 — full-catalog structural coverage

Catalog courses split at the id=1150 proxy boundary: 989 "old" (id<1150), 514 "recent" (id≥1150).

| | Count | % of 514 |
|---|---:|---:|
| Narrow match | 108 | 21.0% |
| Wide match only (narrow failed) | 70 | 13.6% |
| No match | 336 | 65.4% |

Only **34.6%** of recent catalog courses find any name-key match against the old catalog at all — this is a catalog-wide, TRAIN/VALID-independent structural ceiling. It is lower than the VALID-side unrestricted coverage (69.5%) because VALID enrolment concentrates on a subset of recent courses (the ones students are actually taking), which skew toward the renamed/re-issued core-curriculum courses that *do* have a clean old-catalog counterpart; the long tail of recent catalog courses with zero VALID enrolment yet is disproportionately the genuinely-new ones with no old-catalog match.

---

## Go / No-go signals

| Signal | Threshold | Measured | Verdict |
|---|---|---:|---|
| Never-in-TRAIN rows covered (narrow, support≥20, excl. service) | Go ≥40%, No-go <20% | **25.5%** | **BORDERLINE** |
| Row-weighted mean abs diff in pass rate | Go ≥0.04, No-go <0.02 | **0.060** (excl. service, all matched) | GO |
| HIGH_RISK service-course rows as % of covered | Go <30%, No-go ≥50% | **17.9%** | GO |

**Overall: `BORDERLINE — escalate`.**

Signal 1 is the only one not clearly GO, and it lands inside the pre-registered borderline band (20–40%) rather than crossing either bound — 25.5% is closer to the No-go side (20%) than the Go side (40%). Signals 2 and 3 both clear their Go bounds with room. This report does not recommend proceeding or stopping; per the task's own rule, a between-bounds result on any signal is escalated, not resolved here.
