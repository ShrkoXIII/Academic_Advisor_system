# Data Organization & Pipeline Governance — Phased Plan (Revision 4, corrected)

## 0. Status of this document

This is a **plan only**. Producing it executes nothing. Every phase runs only after explicit user approval, one phase at a time.

**Revision 4 supersedes Revision 3.** It records the architecture change decided after Phase 4 (diploma merge moved upstream to its own stage), the split-generation decision, layered validation, and a set of out-of-band code changes already applied. It keeps every Rev 3 rule not explicitly changed here.

**Corrected 2026-07-07.** This text is the promoted, canonical Revision 4. Before promotion it was reconciled line-by-line against the repository at commit `c6069bc` (per-file evidence in §1.4; corrections listed in §0.1). Where this corrected text and the pre-correction Rev 4 draft disagree, this text governs.

**Honesty rule for this revision:** a decision being *approved* is not the same as it being *implemented*. Where the contract and the current code disagree, this document says so plainly (see §1.2 "Implementation status"). Phase 5 writes contracts (the target); Phase 7 makes the code match them. No item is marked resolved until code verification confirms it.

**State layers.** Every claim in this document belongs to exactly one layer, and the text says which:

1. **Historical audited state** — what Phases 1–3 observed (kept verbatim; superseded findings are marked superseded, never deleted).
2. **Already-applied out-of-band changes** — code changes committed before Phase 7, recorded in §1.3.
3. **Current implemented state** — what the code at `c6069bc` actually does (§1.2, §1.4).
4. **Approved but not yet implemented architecture** — D1/D2 targets the code does not yet fully satisfy.
5. **Phase 7 remediation still pending** — the gap between layers 3 and 4.

### 0.1 Correction log (pre-correction Rev 4 draft → this text)

1. **§4.1 Chain A stage-2 validation rewritten.** The old diploma merge (`merge_diploma.py`, onto `after_fet_eng.parquet`) and the new upstream merge (`02` notebook, onto `merged_add_acd_crg.parquet`) operate on **different populations**. Absolute matched/unmatched-count parity across the two merges is NOT required and must not be asserted. Required instead: left-join row-count invariance, no duplicate expansion, match-status parity on the common comparable population, downstream logical parity after `select` and feature engineering.
2. **`diploma_gpa` median fill re-classified.** Code inspection (`Academic_info_clean.ipynb` cell 7) shows the fill median is computed over the **entire diploma source population** at pre-processing time — upstream of the split, therefore fit on data that includes future validation/test cohorts. This conflicts with the locked train-only leakage-control rule. It is now **surfaced as open conflict D6**, not encoded as an approved exception (the pre-correction draft's "documented, logged exception" wording is withdrawn).
3. **C1 status confirmed, not upgraded.** `merge_diploma.py` has no live caller, but its historical in-place writer (`df_merged.to_parquet(AFTER_FET_ENG_PATH)`, line 262) is still executable code with no superseded docstring. Status remains **REDESIGNED, ENFORCEMENT PENDING**. Additionally recorded: `docs/pipeline_rules.md` ("src/ module ownership") still documents `merge_diploma.py` as a sanctioned extender of `after_fet_eng.parquet` — pre-D1 wording, flagged for reconciliation (see §10).
4. **C2–C4 scope corrected: TWO in-place split rewriters, not one.** Code inspection shows `course_difficulty.ipynb` ALSO saves its enriched frames back onto the shared split paths (`df_train_enriched.to_parquet(TRAIN_PATH)` etc., same `MODEL_DATA_DIR/df_{train,valid,test}.parquet` in and out), in addition to `diploma_type_bucketing.ipynb`. D2 remains approved / NOT implemented; C2–C4 remain active until distinct split generations exist in code.
5. **Phase 5 contract list extended** with contract 14 (train-fitted transformation persistence, including the full diploma-bucketing fit-state spec) and contract 15 (inference consistency / frozen preprocessing state).
6. **`01_merge_crg_add_acd` MERGE_DIR alignment verified as already applied.** Cell 21 writes `MERGE_DIR / merged_add_acd_crg.parquet`. The Phase 7c instruction to "align its output save target to MERGE_DIR" is converted to **verification only** — do not re-schedule. The §1.3 claim "no other reader still points at the old `features/` path" is corrected: no **pipeline** reader does, but two **diagnostic** notebooks still reference it (§1.4, items 10–11).
7. **Feature-engineering pass-through statically verified.** `run_feature_engineering_job` deep-copies its input and passes unknown columns through; no generic column-wide transform (imputation/scaling) touches `diploma_gpa`/`diploma_type_id`; `select.ipynb`'s drop-lists do not name them. The promised explicit survival assert is **not yet in code** (7a item). Runtime pass-through verification remains mandatory before any rebuild (§4.1 pre-check).
8. **Out-of-band sequence wording corrected** (§3): code changes already occurred before Phase 7, so the absolute claim "first disk/code change is Phase 7 only" is replaced with: *no further planned disk/code changes occur before Phase 7; previously applied out-of-band changes are explicitly recorded.*
9. **§1.4 current-code verification table added** — nine files inspected 2026-07-07 at `c6069bc`.
10. **Naming debt updated.** Commit `c6069bc` already renamed `02_final_mergerd_with_dimploma.ipynb` → `02_final_merged_with_dimploma.ipynb` out-of-band ("mergerd" fixed). The residual "dimploma" typo remains: current on-disk name is **`02_final_merged_with_dimploma.ipynb`**, recorded as Phase 6 naming debt. No rename during Phase 5.
11. **Phase 8 tooling corrected.** `scripts/parity_check.py` does not exist in the repository (there is no `scripts/` directory). Phase 8 wording changed from "reuse" to "create".
12. **Decisions_Log entries required by §1.3 have been written** to `obsidian_vault/Decisions_Log.md` (D1–D4, six-notebook path fix, `merged_add_acd_crg.parquet` relocation with superseded-finding note, `select.ipynb` read-source change) — documentation-only change, no code or data touched.

---

## 1. Context

**Completed prior job:** paths cleanup. Every **pipeline** notebook/script resolves paths from `src/paths.py`. This premise was briefly broken and is now restored — see the out-of-band record below. Corrected caveat (2026-07-07): one live hardcoded absolute path survives in a **non-pipeline diagnostic** notebook (`mark_finish_status_disagreement_diagnostic.ipynb`, §1.4 item 11) and several notebooks carry stale hardcoded paths inside markdown "Architectural Notes" cells — both recorded as 7a/7b debt; the final repo-wide re-grep obligation stands.

**NOT complete — part of THIS job:** notebook naming, notebook numbering, DataFrame naming, saved-artifact naming governance and traceability.

**Scope of this job (unchanged from Rev 3):** split integrity; ID/dtype consistency; full lineage; one-owner-per-artifact governance; intra-tree placement; duplicate/loose-file review; notebook naming & numbering; DataFrame naming; artifact naming & traceability; governance contracts; controlled remediation; selective rebuild; byte/logical/expected-delta validation; final freeze.

**Data root (fixed fact):** the active root exists, is populated, and is protected. It resolves via `ACADEMIC_ADVISOR_DATA_DIR=D:/AI/data_clean_academic_advisor/data`, read-only during audit.

**Out of scope:** promotion/migration into `PROJECT_ROOT/data`; retiring/repointing `ACADEMIC_ADVISOR_DATA_DIR`; copying the cleaned tree into the repo shell; changing where the protected data physically lives.

**Wording rule (existence ≠ authority):** artifacts in the active root are *current on-disk artifacts*, never "trusted/authoritative" by default. Authority comes only from Phase 1–3 evidence.

### 1.1 Resolved decisions (D1–D4)

**D1 — Diploma merge moved upstream, resolves conflict C1 by design.**
A dedicated notebook (logical name `02_merge_diploma`; current on-disk filename `02_final_merged_with_dimploma.ipynb` — residual typo flagged for rename, see §6) is inserted between `01_merge_crg_add_acd` and `select`. It reads the CRG+ADD+ACD merge output plus the cleaned diploma source, joins `diploma_type_id`/`diploma_gpa`, and writes a **distinct** artifact under `MERGE_DIR` (`merged_with_diploma.parquet`). It does **not** extend `after_fet_eng.parquet` in place. The old writer `merge_diploma.py` is retired from the intended DAG.

**D2 — Split generations become distinct artifacts, resolves C2–C4 by design.**
`model_data` splits become three distinct generations instead of in-place rewrites: base (`split_diagnostics.ipynb`) → difficulty-enriched (distinct file, `course_difficulty.ipynb`) → final (distinct file, `diploma_type_bucketing.ipynb`). No stage overwrites another stage's file. Concrete filenames deferred to Phase 6.

**D3 — Layered validation for the diploma rebuild chain (Chain A).** See §4.1.

**D4 — No separate pre-Phase-5 preservation gate.** A full backup of both project and data tree already exists (user-confirmed). Preservation of `df_train/valid/test` and the two orphan artifacts stays inside Phase 7 as scoped in Rev 3; the existing backup covers the interim.

### 1.2 Implementation status (contract vs. code — READ THIS)

Verified against code 2026-07-07 at commit `c6069bc`.

| Item | Decision | Code on disk today (post-7a, 2026-07-07) | Status |
|------|----------|--------------------|--------|
| C1 — single writer of the feature-engineered artifact | Resolved by design (D1) | `merge_diploma.py` neutralized (module-level raise); sole writer is `02_feature_engineering.ipynb` → `FEATURES_DIR/feature_engineered_primary.parquet`; the D1 chain ran end-to-end in 7c with runtime guards passing. | **RESOLVED — CODE-ENFORCED + RUNTIME-VERIFIED (7c, 2026-07-08).** Final re-confirmation rides Phase 8. |
| D2 — distinct split generations | Approved | Code AND data: `01_split_diagnostics` → `df_*_base`; `02_course_difficulty` → `df_*_difficulty`; `03_diploma_type_bucketing` → `df_*_final`. All nine generation files exist on disk, validated against the pre-rebuild splits (positional parity). Bare names retired to `archive/pre_7c/`. | **IMPLEMENTED AND BUILT (7c, 2026-07-08).** |
| C2–C4 — split in-place overwrite chain | Removed by D2 | No notebook writes another stage's path; no bare-name writer or reader remains in pipeline code; training-side readers repointed to `df_*_final` (7b). | **RESOLVED (7b/7c, 2026-07-08).** |
| `02_merge_diploma` notebook guards | Required (§4 contract) | All guards exercised at runtime during the 7c rebuild (dtype-string both sides, uniqueness + `many_to_one`, suffix consistency, double-merge guard, row-count invariance) — all passed. The dtype guard had earlier correctly BLOCKED runs against the unnormalized on-disk source (validation record, Findings). | **GUARDS RUNTIME-VERIFIED (7c, 2026-07-08).** |
| `select` diploma survival assert | Promised in §10 | Fired and passed at 7c execution; diploma columns survive selection (validated +2 columns exactly). | **IMPLEMENTED + RUNTIME-VERIFIED (7c).** |
| Data-root guard (contract 12) | Required | `assert_data_root` in `src/paths.py`; wired into the three split-stage notebooks and `02_feature_engineering`; exercised during 7c. | **IMPLEMENTED — partial wiring.** Training/inference entry points (`01_train_lightgbm`, `inference.py`) not yet wired — remaining non-conformance for Phase 8/9 follow-up. |
| Contract 14 — fitted-state persistence | Required | `artifacts/diploma_type_bucket_map.json` written by `03_diploma_type_bucketing` (validated content). `course_difficulty_lookup.parquet` producer: adoption stands, implementation **deferred** — requires new aggregation logic matching `inference.py`'s contract; needs its own gated task (validation record, Findings). `knn_index.pkl` untouched (D7 HOLD). `categorical_levels` persistence pending (training not rerun). | **PARTIALLY IMPLEMENTED (7c).** Open items listed in §10. |

Phase 5 may write the contracts for all of the above (a contract describes the target). No Phase 5 wording may state these are already satisfied.

### 1.3 Out-of-band changes already applied (record, do not re-schedule)

Six notebooks had live hardcoded `D:\` paths bypassing `src/paths.py`, found during Phase 3 cross-check, fixed manually, committed:
`select.ipynb`, `clean_v_crg_student_course.ipynb`, `handeling_outliers.ipynb`, `pipeline_run_judge_test.ipynb`, `add_student_degree_status_clean.ipynb`, `load_preprocessing.ipynb`.
These are already-applied; do NOT re-schedule as Phase 7 work. A final repo-wide re-grep is still required to confirm no live hardcoded data-root path remains (the 2026-07-07 sweep found one further live case in a diagnostic notebook — §1.4 item 11 — which is NEW 7a work, not a re-run of the fix above).

Additional applied changes:
- Diploma-source normalization: `student_id` is now normalized with `cleaning_utils.normalize_id_columns` in the diploma-source cleaning notebook (`Read.ipynb`, renamed by user to `Academic_info_clean.ipynb`), at cleaning time — not patched at merge time. Guards added and verified in code: row count unchanged, unique-key count unchanged, dtype confirmed `string`.
- `01_merge_crg_add_acd` output `merged_add_acd_crg.parquet` relocated from `features/` to `MERGE_DIR`. This **supersedes** the Phase 3 ledger finding that `features/` was its correct home (reason: `features/` is for derived features; this is a merge intermediate). **Verified 2026-07-07:** cell 21 writes `MERGE_DIR / merged_add_acd_crg.parquet`; the notebook also writes the CRG+ADD intermediate `MERGE_DIR / merge_crg_add.parquet` and the unmatched-audit CSV `MERGE_DIR / merge_crg_add_unmatched_add_snapshot.csv` (the CSV's placement in `MERGE_DIR` rather than `AUDIT_DIR` is a folder-ownership tension recorded for contract 6). Corrected reader claim: no **pipeline** stage reads the old `features/` path; two **diagnostic** notebooks still reference it (§1.4 items 10–11) — 7b repoint/cleanup targets.
- `select.ipynb` read-source updated to `MERGE_DIR / merged_with_diploma.parquet`. **Verified 2026-07-07** (cell 3).
- Notebook rename applied in commit `c6069bc`: `02_final_mergerd_with_dimploma.ipynb` → `02_final_merged_with_dimploma.ipynb` ("mergerd" typo fixed; "dimploma" typo remains — Phase 6 debt, §6).
- User edits observed 2026-07-07 (uncommitted at time of recording): `add_student_degree_status_clean.ipynb` — the second write (root pre-drop copy `preprocessed/v_add_student_degree_status_clean.parquet`) removed, resolving the paths-audit "writes twice" quirk at the producer side; `load_preprocessing.ipynb` — unused import + dead CSV path line removed. The stale root pre-drop artifact remains on disk; its ARCHIVE decision stands (`docs/naming_plan.md` 6C).

**Decisions_Log entries (required by this section): WRITTEN 2026-07-07** to `obsidian_vault/Decisions_Log.md`, each separate: the six-notebook path-fix commit; D1; D2; D3; D4; the `merged_add_acd_crg.parquet` relocation with its superseded-finding note; and — logged separately from the path fix — `select.ipynb`'s D1 read-source change (two distinct changes to the same file, not conflated).

### 1.4 Current-code verification record (2026-07-07, commit `c6069bc`)

Static inspection (no execution, no data touched). Layer-3 ground truth for this revision:

| # | File | Verified facts |
|---|------|----------------|
| 1 | `note_books/merge/01_merge_crg_add_acd.ipynb` | Reads cleaned CRG/ADD/ACD from `PREPROCESSED_DIR`; all paths from `src/paths.py`. Writes `MERGE_DIR/merge_crg_add.parquet` (intermediate), `MERGE_DIR/merged_add_acd_crg.parquet` (final, cell 21), `MERGE_DIR/merge_crg_add_unmatched_add_snapshot.csv` (audit CSV). Row-count invariance asserts on both merge steps; `validate='many_to_one'` on both joins. Stale markdown (cell 0) still names `FEATURES_DIR` as output — doc drift only; code governs. |
| 2 | `note_books/merge/02_final_merged_with_dimploma.ipynb` | D1 notebook. Reads `MERGE_DIR/merged_add_acd_crg.parquet` + `PREPROCESSED_DIR/V_add_academic_info/v_add_adcademic_info_cleaned.parquet`; writes distinct `MERGE_DIR/merged_with_diploma.parquet`. Does NOT touch `after_fet_eng.parquet`. Full guard set present (§1.2 row 4). Cell-4 error text still references "Read.ipynb" (renamed) — doc drift, Phase 6/7. |
| 3 | `note_books/feature_eng/select.ipynb` | Reads `MERGE_DIR/merged_with_diploma.parquet` (cell 3); writes `AUDIT_DIR/df_crg_add_acd.parquet` (cell 62). Denylist-based column drops; diploma columns absent from all drop-lists (survive implicitly); **no survival assert**. Stale markdown cell 0 carries old hardcoded paths + old `features/` input claim — markdown only, not live code. |
| 4 | `note_books/feature_eng/handle_gpa.ipynb` | Live writer of `AUDIT_DIR/after_fet_eng.parquet` (cell 27) via `run_feature_engineering_job`. Also still writes debug residue `DATA_DIR/merged.csv` (cell 21, 5-row head) — the producer of a Phase 4 ARCHIVE-OLD residue is live code; 7a item. |
| 5 | `src/feature_engineering.py` | `run_feature_engineering_job` deep-copies input, adds/repairs **named** columns only, passes unknown columns through to `df_primary`; no reference to diploma columns anywhere; no generic column-wide imputation/scaling; module has no hardcoded paths. |
| 6 | `src/merge_diploma.py` | Superseded by D1 but fully executable: reads AND rewrites `AUDIT_DIR/after_fet_eng.parquet` in place (line 262). Guards exist (uniqueness, suffix check, double-merge guard, row-count assert) but the in-place write stands. No superseded docstring yet. No live caller found. |
| 7 | `note_books/model_eng/split_diagnostics.ipynb` | Reads `AUDIT_DIR/after_fet_eng.parquet`; writes base splits `MODEL_DATA_DIR/df_{train,valid,test}.parquet` (index=True) with post-save read-back shape verification. Sole computer of split boundary masks (consistent with Decisions_Log). |
| 8 | `note_books/model_eng/course_difficulty.ipynb` | Reads the three shared split paths; builds train-only difficulty features (6-level fallback, LOO on train); asserts +6 columns; **saves enriched frames back onto the SAME three split paths** — in-place rewrite of another stage's artifact (C2–C4). No lookup-table artifact is persisted by committed code (orphan `course_difficulty_lookup.parquet` remains producer-less). |
| 9 | `note_books/model_eng/diploma_type_bucketing.ipynb` | Reads the three shared split paths; fits bucketing on `df_train` only: top-5 codes kept as raw values, other train codes → rare label `6` (collision-checked), unseen-in-train + nulls → `-1`, explicit `pd.Categorical` category set = sorted(top-5) + [6, −1]; asserts exactly +1 column; **saves back onto the SAME three split paths** (C2–C4). Failure message still instructs running `merge_diploma.py` — stale under D1, 7a fix. **None of the fitted state (codes, mapping, labels, category set) is persisted anywhere** — contract 14 input. |
| 10 | `note_books/debug/trace_student.ipynb` (incidental finding) | Lineage-debug tool; builds paths from `DATA_DIR` but still lists the old `features/merged_add_acd_crg.parquet` location and `final/without_outliers.parquet`. Referencing historical artifacts is partly its purpose (stale-artifact detection); still recorded as a 7b repoint target. |
| 11 | `note_books/pre_processing/V_CRG_STUDENT_COURSE/mark_finish_status_disagreement_diagnostic.ipynb` (incidental finding) | **Live hardcoded absolute path in a code cell**: `D:\AI\Real projects\Academic_Advisor\data\features\merged_add_acd_crg.parquet` — bypasses `src/paths.py`, points at the repo-shell root (not the active env-var root) AND at the old `features/` location. Diagnostic notebook, not a pipeline stage. NEW 7a item. |
| 12 | `note_books/pre_processing/V_ACADEMIC_INFO/Academic_info_clean.ipynb` (incidental finding) | Diploma-source cleaner (renamed from `Read.ipynb`). Reads `RAW_DIR/v_add_academic_info.parquet`; drops rows with null `diploma_type_id`; **fills `diploma_gpa` nulls with the median of the full source population** (cell 7 — D6 conflict, §10); contains a dead `fillna(6.111)` on `diploma_type_id` after the dropna (dead code + a float-literal dotted-ID hazard pattern; the raw `diploma_type_id` is float-typed with a dotted suffix at source); normalizes `student_id` with `normalize_id_columns` under row/unique/dtype asserts; writes `PREPROCESSED_DIR/V_add_academic_info/v_add_adcademic_info_cleaned.parquet`. |

---

## 2. Pre-flight record (already executed and resolved)

Phase 1 opening visibility check ran: env var unset → fell back to empty repo shell → correctly STOPPED; env var set → PASSED (`df_train` 19,748,821 B, `df_valid` 6,729,784 B, `df_test` 4,760,639 B, all mtime 2026-07-02 16:51).

Standing caveats: nonzero size proves availability only; mtimes are diagnostic metadata only (do not infer chronology/authority); `src/paths.py` has an import-time `ensure_dir` side effect — do not re-import to resolve paths during read-only phases; do not assume env-var persistence across command contexts (set/verify inline each run); audit outputs are documentation, never written inside the data tree.

---

## 3. Phase overview

| # | Phase | Mode | Output | Gate |
|---|-------|------|--------|------|
| 1 | Split integrity audit | Read-only | Writer/reader/overwrite map (deltas labeled OBSERVED/INFERRED/UNRESOLVED), base-vs-enriched, reconstruction-capability | Approve before 2 |
| 2 | ID & dtype audit | Read-only | Taxonomy, normalizer inventory, dangerous-cast list, join-key map | Approve before 3 |
| 3 | Ownership & lineage audit | Read-only | Ledger + DAG + conflicts + duplicates (binary/logical) + loose files | Approve before 4 |
| 4 | Placement & action mapping | Planning | Per-artifact action record + validation type + dependencies | Approve before 5 |
| 5 | Governance contracts | Planning | Draft contracts incl. ownership, split-immutability, stage boundaries, rollback, fitted-state persistence | Approve before 6 |
| 6 | Naming & numbering plan | Planning | Notebook + DataFrame + artifact-name proposals | Approve before 7 |
| 7 | Controlled remediation | **Write** | Gated groups; change → group validation → approval → promote | Per-group; 7c distinct |
| 8 | Final cross-pipeline validation | Read-only | Integration + drift screen | Approve before 9 |
| 9 | Final freeze | Read-only + git tag + manifest | Freeze checklist, lineage, dual rollback points | Done |

Phases 1–6 are non-destructive. **No further planned disk/code changes occur before Phase 7. Previously applied out-of-band changes are explicitly recorded (§1.3) and are not re-scheduled.**

**Current position:** Phases 1–8 complete. **Phase 9 (final freeze) executed 2026-07-08** — see `docs/manifests/freeze_phase9_2026-07-08.md`. Job DONE, subject to the carried-forward open items listed there (repo-shell debris left for user disposition, legacy `models/` duplicates left for user disposition, DataFrame naming (6B) not executed, `course_difficulty_lookup.parquet`/`knn_index.pkl` producers still open per D7).

---

## 4. Validation model

**A. Byte parity** — exact copy/move, no rerun. Hash + size equality. Any difference fails.

**B. Logical parity** — rerun after a non-semantic change (variable/notebook rename, path-only edit). Byte equality NOT required (parquet compression/metadata/row-group order aren't byte-stable). Compare: row counts; unique keys + duplicate counts; schema + dtypes; values sort-normalized on stable keys; null patterns; distributions. Requires determinism: fixed seeds, stable sort, declared float tolerance. Any delta beyond these fails.

**C. Expected delta** — intentional fix (ID normalization, dtype fix, overwrite prevention, logic correction). Result may change, but only where predicted, by an amount/type declared *before* the change, with no unexplained downstream drift.

### 4.1 Chain A layered validation (D3) — diploma redesign

Not a fourth type; a named application of A/B/C per stage.

**Population note (governs stage 2).** The old merge (`merge_diploma.py`) joined diploma columns onto `after_fet_eng.parquet` — a population downstream of `select` and feature-engineering exclusions. The new merge (`02_merge_diploma`) joins onto `merged_add_acd_crg.parquet` — the full pre-`select` merged population. These are **different populations**; absolute matched/unmatched totals are not comparable between them and MUST NOT be required to match.

| Stage | Type | Pre-declared expectation |
|-------|------|--------------------------|
| Cleaned diploma source | EXPECTED DELTA | Relative to the prior cleaned artifact, the only intended change is `student_id` schema: double → string. Identical row count vs the prior cleaned artifact; 1:1 mapping (zero ID collisions — asserted in code); suffix preserved; null pattern unchanged. (The notebook's pre-existing row filtering — dropna on `diploma_type_id` — and the `diploma_gpa` median fill are unchanged pre-existing logic; the fill's leakage conflict is D6 and is a separate decision, not part of this delta.) |
| Diploma merge (`02_merge_diploma`) | LOGICAL PARITY (population-aware) | (1) **Row-count invariance of the new left join** — output rows == left-input rows (asserted in code). (2) **No duplicate expansion** — `validate='many_to_one'` + source-uniqueness assert. (3) **Match-status parity on the common comparable population**: for rows/students present in BOTH the old population (`after_fet_eng` grain) and the new one, match status must not change — Phase 3 proved the ~48 unmatched rows / 6 old-cohort students on that population are genuinely missing records, not float corruption; any recovered or newly-lost match **on the common population** is a **STOP**. (4) New matches on rows outside the old population (rows later removed by `select`/exclusions) are expected and are NOT failures. Do NOT compare absolute matched/unmatched totals across the two merges. |
| `select` output | LOGICAL PARITY | Plus required-column survival: `diploma_gpa`, `diploma_type_id` must be present after selection (assert — not yet in code, 7a). |
| Feature-engineered artifact + final splits | LOGICAL PARITY | Downstream logical parity on the common population, unless a separately approved change introduces an explicit expected delta. |

**Mandatory Phase 7c pre-check (before any rebuild runs):** diploma columns now flow *through* `feature_engineering.py` instead of bypassing it. Verify `diploma_gpa`/`diploma_type_id` survive the full chain `02_merge_diploma → select → feature_engineering.py → after_fet_eng` and that no generic column-wide transform (train-only imputation, scaling) touches them. Statically verified 2026-07-07 (§1.4 items 3, 5: pass-through by deep-copy, no generic transforms, absent from drop-lists) — **runtime confirmation still required before any rebuild.** If a transform would touch them, either exclude these columns or declare the delta explicitly. An unexcluded generic transform here is the single most likely silent parity failure in the whole rebuild.

---

## 5. Detailed phase definitions

### Phase 1 — Split integrity audit (read-only) — COMPLETE
As Rev 3. Deltas labeled OBSERVED/INFERRED/UNRESOLVED. HARD STOP only for missing/zero-size split or inability to inspect safely; multiple writers → flag CRITICAL, continue mapping, block remediation. `after_fet_eng.parquet` is a candidate reconstruction source only; authority deferred to Phase 3.

### Phase 2 — ID & dtype audit (read-only) — COMPLETE
As Rev 3. Confirmed CRITICAL: `student_id` float64-sourced/unnormalized on the diploma join path (now fixed at source, per §1.3). Three normalizer implementations confirmed (`cleaning_utils.normalize_id_to_string`/`normalize_id_series`/`normalize_id_columns` family, `feature_engineering._normalize_key_series`, plus ad-hoc notebook casts). Dangerous-cast search is pattern-based, not exhaustive. Addendum 2026-07-07: raw `diploma_type_id` is float-typed with a dotted suffix at source and is suffix-stripped to `Int64` at the merge stage — an ID-class fact for contract 1.

### Phase 3 — Ownership & lineage audit (read-only) — COMPLETE, with one superseded finding
As Rev 3, plus: the `merged_add_acd_crg.parquet` placement finding (`features/` = correct home) is **superseded** by the §1.3 relocation to `MERGE_DIR`. The original finding stays in the record, marked superseded and dated — not deleted.

Duplicate verification is two-tier: (A) exact binary duplicate = same hash+size; (B) logical duplicate = hash differs but passes §4B comparison. A hash mismatch never alone proves two parquets differ. Verdicts: BINARY DUPLICATE / LOGICAL DUPLICATE / DISTINCT / UNRESOLVED.

**Refreshed current-state DAG (post-D1/D2 target):**
```
raw → preprocessing (clean CRG/ADD/ACD/Diploma; normalize student_id at diploma source)
  → 01_merge_crg_add_acd → MERGE_DIR/merged_add_acd_crg.parquet   (implemented)
  → 02_merge_diploma      → MERGE_DIR/merged_with_diploma.parquet (implemented, distinct)
  → select                → selected modeling population (currently AUDIT_DIR/df_crg_add_acd.parquet)
  → handle_gpa / feature engineering → after_fet_eng   (single writer — enforcement pending)
  → split_diagnostics     → base splits (implemented, MODEL_DATA_DIR)
  → course_difficulty     → distinct difficulty-enriched splits   (D2 — NOT built; today: in-place rewrite)
  → diploma_type_bucketing→ distinct final-model splits           (D2 — NOT built; today: in-place rewrite)
  → training
Outlier branch (undecided, D5): selected/feature frame → handeling_outliers.ipynb
  → without_outliers.parquet → no confirmed live training consumer
  (2026-07-07 census: only its producer and the debug tracer reference it).
```

### Phase 4 — Placement & action mapping (planning) — COMPLETE, interpreted through Rev 4
Action model unchanged (primary + ordered secondary + validation type + dependencies). Updates:

- **Merge artifacts** (`merged_add_acd_crg.parquet`, `merged_with_diploma.parquet`, plus intermediate `merge_crg_add.parquet`): Primary KEEP, location `MERGE_DIR`, filenames/ownership verified against implemented code (§1.4 items 1–2).
- **`after_fet_eng.parquet`**: no longer blocked by the C1 ownership gap (D1). Action: MOVE LATER + REPOINT single producer + REPOINT consumers + PRESERVE. Validation byte parity for pure move, logical parity if rerun required. **Precondition:** code verification confirms a single current writer (see §1.2 — pending until `merge_diploma.py` is neutralized).
- **Selected modeling population** (`df_crg_add_acd.parquet`): `select` is no longer an audit step, so its output must leave `AUDIT_DIR`. Action: MOVE LATER + optional RENAME LATER + REPOINT CONSUMERS + PRESERVE. Byte parity.
- **`merge_diploma.py`** (superseded code): action KEEP-IN-PLACE + add a one-line top docstring marking it superseded and not called. **Do NOT move to `ARCHIVE_DIR`** — it is tracked code; git already preserves its history permanently. `ARCHIVE_DIR` is for gitignored data git cannot see (Rev 3 §9). Archiving tracked code adds nothing and blurs that distinction.
- **`course_difficulty.ipynb` and `diploma_type_bucketing.ipynb`**: both save cells still do the C2–C4 in-place overwrite (§1.2, §1.4 items 8–9), and the bucketing failure message still says "run merge_diploma.py". All are Phase 7 code fixes under D2. Flag now; fix in 7.
- **Debug residues** (`merged.csv`, `after_feature_eng_run.csv`): ARCHIVE OLD, subject to archive gate. Addendum: `merged.csv`'s producer is still live code (`handle_gpa` cell 21) — retiring the residue requires also retiring/gating the write (7a).
- **Orphans** (`course_difficulty_lookup.parquet`, `knn_index.pkl`): PRESERVE + USER DECISION (reconstruct a designated producer vs. accept as frozen legacy). Producer reconstruction is nearly free during the D2 rebuild — see §10 open items. Under contract 14 these become mandatory persisted fitted-state artifacts if adopted.

### Phase 5 — Governance contracts (planning) — APPROVED (user, 2026-07-07)
Draft the durable rules. Contract inputs include D1–D4, the §1.2 status table, and the §1.4 verification record. Deliverable: `docs/governance_contracts.md` (DRAFT). Contracts:

1. **ID/dtype contract** — per ID class: dtype + canonical normalizer; dotted-suffix preservation; ban on float joins; `diploma_type_id`'s float-at-raw → suffix-stripped-`Int64` lifecycle made explicit.
2. **Ownership contract** — each artifact path has exactly one writer-owner. A consumer may read upstream and write its own distinct downstream artifact; it must NOT rewrite an upstream path unless designated owner. Encode `after_fet_eng.parquet`'s single writer = the feature-engineering stage (`handle_gpa`), with C1 marked enforcement-pending until §1.2 clears.
3. **Split immutability & distinct split generations** — one owner for base splits; enrichment writes distinct derived artifacts, never in-place (D2). Encode as a rule even though C2–C4 code is still active in **both** enrichment notebooks.
4. **Overwrite policy** — in-place rewrite allowed only if idempotent, documented, by the designated owner; otherwise forbidden.
5. **Stage boundaries** — `raw → clean → merge(CRG+ADD) → merge(CRG+ADD+ACD) → merge(+Diploma) → select → feature-eng → split(base) → difficulty(derived) → bucketing(final) → training → KNN → inference/recommendation`.
6. **Folder ownership** —
   ```
   RAW_DIR          raw extracts only
   PREPROCESSED_DIR cleaned source-table artifacts only
   MERGE_DIR        merge outputs only
   FEATURES_DIR     selected + feature-engineered modeling frames
   AUDIT_DIR        diagnostics, mismatch reports, validation evidence only
   MODEL_DATA_DIR   split artifacts by immutable derivation stage
   ARTIFACTS_DIR    runtime/model-support artifacts with reproducible producers
   ARCHIVE_DIR      approved superseded/obsolete DATA artifacts (not tracked code)
   ```
7. **Artifact naming principles** — encode stage/content/ownership/derivation.
8. **Notebook numbering principles** — reflect real execution order.
9. **DataFrame naming principles** — semantic by source/domain + stage.
10. **Remediation safety** — preserve-before-change; update-consumers-after-ownership-confirmed; no silent overwrite; one group at a time; validate before promote.
11. **Rollback/baseline-manifest** — git for code; manifest + isolated copies for gitignored data (§9). Define manifest format, capture timing, isolation requirement.
12. **Data-root guard** — startup assertion that resolved splits exist and are nonzero; mitigate the `ensure_dir` import side effect. Proposes the code change; implementation is a gated 7a item.
13. **Pre-admission feature availability** — `diploma_type_id`/`diploma_gpa` are known before the target semester; joining them upstream does not leak **as values**. The full-source median fill for `diploma_gpa` was surfaced as conflict **D6** and **resolved by user decision 2026-07-07**: accepted as an explicitly approved, logged exception for this cycle (§10, Decisions_Log). The train-only rule stands unchanged for all future fitted statistics (contracts 14–15).
14. **Train-fitted transformation persistence** — any transformation fitted on train and needed at inference must persist its fitted state as a versioned artifact. For diploma bucketing the persisted state MUST include: the top train codes (`TOP_DIPLOMA_CODES`); the rare-code policy and full code→bucket mapping; the unseen-code policy (unseen + null → unseen label); the reserved labels (rare=6, unseen=−1) and their collision guarantees; the final category set; the fit-source version; and a reference to the train-split manifest it was fitted on. Applies equally to `learn_categorical_levels` output, course-difficulty lookup tables, and the KNN index.
15. **Inference consistency / frozen preprocessing state** — inference must reuse persisted fitted state (contract 14), never refit; preprocessing applied at inference must be the frozen train-time version.

Guardrails: contracts must not weaken locked decisions in `pipeline_rules` (temporal split boundaries, `feature_contract.json` allowlist, train-only leakage control, no `scale_pos_weight`/SMOTE, `finish_status` target). Conflict → surface, don't encode (D6 is exactly such a surfaced conflict; so is the stale `merge_diploma.py` ownership wording in `pipeline_rules.md`).

### Phase 6 — Naming & numbering (planning) — APPROVED (user, 2026-07-07)
Deliverable: `docs/naming_plan.md` (6A notebooks & numbering, 6B DataFrames, 6C artifacts incl. the D2 split-generation filenames). Planning only — no rename executes before its 7b group. Notebooks, DataFrames, saved artifacts. New targets to name: current on-disk `02_final_merged_with_dimploma.ipynb` → fix residual typo ("dimploma"→"diploma"; "mergerd" already fixed out-of-band in `c6069bc`) and assign stage number; `merged_with_diploma.parquet`; the difficulty-enriched and final split filenames (D2); `note_books/model_eng/read.ipynb`, root `read.ipynb`, `note_books/pre_processing/all.ipynb` (meaningless names); stale in-code references to the renamed "Read.ipynb" (e.g. `02` notebook cell 4) and to `merge_diploma.py` (bucketing failure text) follow the renames in 7. Honor: one merge notebook per stage; early merge and diploma merge stay separate; no two files differing only by case.

### Phase 7 — Controlled remediation (WRITE; gated groups)
Per-group flow: change → immediate group validation (type per Phase 4 record) → user approval → promote/retire preserved originals → next. No parallel execution.

- **7a — Code & ownership fixes. EXECUTED 2026-07-07 (user-authorized).** Done: `merge_diploma.py` neutralized (superseded docstring + module-level raise; single-writer verified); **both** `course_difficulty.ipynb` and `diploma_type_bucketing.ipynb` save cells now write distinct generations (D2) and read the previous generation; bucketing stale failure message + stale markdown fixed (incl. a broken stale-column guard string found during the edit); `select` diploma-survival assert added; diploma-source normalization verified-only (already applied); `merged.csv` debug-write cells removed from `handle_gpa`; live hardcoded path in `mark_finish_status_disagreement_diagnostic.ipynb` repointed to `MERGE_DIR`; data-root guard added to `src/paths.py` and wired into the three split-stage notebooks; `pipeline_rules.md` ownership wording reconciled (dated). Deferred out of 7a: fitted-state persistence (contract 14 — lands with the 7c rebuild); guard wiring into training entry points; D6 remediation (none — approved exception); D7 items untouched. §1.2 table is the binding post-7a status.
- **7b — Relocations & renames. EXECUTED 2026-07-08 (user-authorized).** All Phase 6 notebook renames applied via `git mv` (23 files/folders; every `read.ipynb`/`all.ipynb` eliminated; `all.ipynb` archived to `note_books/archive/`); all reader/writer repoints applied and statically validated; stale markdown rewritten; baseline manifest + isolated backups created BEFORE any data change; data-side moves hash-verified (raw diploma corrected name; `merged_add_acd_crg.parquet` byte-parity copy to `MERGE_DIR`; ACD `clean_` name; unmatched CSV → `AUDIT_DIR`; debug residues + pre-drop twin → `archive/pre_7c/`).
- **7c — Selective rebuild. EXECUTED 2026-07-08 (user-authorized).** Chain executed headlessly in order: diploma-source cleaning → `02_merge_diploma` (first materialization of `merged_with_diploma.parquet`) → select → feature engineering → base splits → difficulty generation → final generation + fitted-state persistence. Every §4.1 stage validated against pre-rebuild backups — full record in `docs/manifests/validation_7c_2026-07-08.md`, including the one declared non-semantic delta (index provenance: the old artifact's index was a `pd.merge` accident of the retired script). `01_merge_crg_add_acd` was NOT rebuilt (as planned). Models NOT retrained (Phase 8 scope). Superseded originals retired to `archive/pre_7c/` after validation.

Ordering: 7a → 7b → 7c is default, not law; evidence-driven per dependency chain. The 7c gate applies to every rebuild wherever it lands.

Guardrails: never overwrite a current on-disk artifact without an isolated preserved copy + manifest. Do not modify split logic, saved splits, target definitions, or `feature_contract.json` without explicit approval. Superseded DATA → `archive/` first; permanent deletion needs separate approval. Minimal changes.

### Phase 8 — Final cross-pipeline validation (read-only) — EXECUTED 2026-07-08 (user-authorized)
Not the first validation (each 7-group self-validated). Final integration pass: end-to-end consistency; split boundaries; feature columns vs `feature_contract.json`; target distributions; join integrity; ID suffixes; drift screen vs baseline manifests. `scripts/parity_check.py` **created** this phase (did not exist before) and run: **51/51 checks PASS**. Full record: `docs/manifests/validation_phase8_2026-07-08.md`. No unexplained drift — nothing stopped. Three informational findings carried forward (stale docstring in `src/model_training.py`, duplicate legacy `.lgbm` files in `models/`, guard-wiring gap into training/inference) — none block Phase 8, all are Phase 9/follow-up candidates.

### Phase 9 — Final freeze — EXECUTED 2026-07-08 (user-authorized)
Checklist verification, an out-of-band incident investigation, one closing hardening pass, and the dual rollback points. Full record: `docs/manifests/freeze_phase9_2026-07-08.md`. Summary:

- **Out-of-band incident found and closed.** Two commits landed between Phase 8 and Phase 9 that had not been reviewed (`54eed08`, `4b0d3b1`). Investigation showed: (a) a behavior-preserving `save_parquet()` I/O refactor (verified line-by-line — every call site semantically identical to the prior `.to_parquet()` call); (b) `clean_v_acd_degree_course.ipynb` was re-executed twice outside any governed phase, once failing (no write) and once **succeeding but writing into the wrong root** — the repo-shell `data/` tree, not the active `ACADEMIC_ADVISOR_DATA_DIR` root — because that notebook had no data-root guard. This is the exact "twin-copy drift" risk already on record in §10. **The active data root's canonical artifact was verified byte-identical (sha256) to its pre-incident state — no governed data was affected.** Closed by wiring `assert_data_root` into every writer notebook that lacked it (`00_extract_raw_tables`, all five per-table cleaners, `01_select_model_population`); notebooks that already asserted `path.exists()` on every upstream input (`01_merge_crg_add_acd`, `02_merge_diploma`) were left untouched (already protected, minimal-changes). Re-validated: `scripts/parity_check.py` 51/51 PASS after the edit; 10/10 unit tests.
- **Checklist:** one owner per artifact — PASS (full writer census re-run, zero multi-writer conflicts outside the neutralized `merge_diploma.py`). One owner for base splits — PASS. No silent overwrites — PASS **after** the incident closure above (was a live counter-example minutes before freeze). Stable schema contract — PASS (`feature_contract.json` unchanged, re-verified). Meaningful notebook order/names — PASS (7b). Documented lineage — PASS (this document + Decisions_Log). Reproducible rebuild sequence — PASS (proven twice: 7c and Phase 8). Fitted-state artifacts persisted per contract 14 — PARTIAL, as already recorded (`diploma_type_bucket_map.json` done; `course_difficulty_lookup.parquet` producer and `categorical_levels` persistence remain open, D7 unresolved for `knn_index.pkl`) — **not blocking**, both are new-logic tasks explicitly out of this paths/names job's scope. Clean tree — PARTIAL: two pre-existing, out-of-scope items **found but deliberately NOT touched** (see below) and DataFrame naming (6B) was never executed in 7b — **NOT MET**, carried forward, not blocking (cosmetic, no governance-correctness impact; renaming ~7 notebooks' internal variables risks exactly the kind of unreviewed change this freeze just had to investigate, so it is deferred rather than rushed).
- **Found but deliberately not touched (destructive-action guardrail):** (1) repo-shell `data/` tree now contains gitignored debris (`data/audit/id_dtype_audit/*` — a discovery report from an untracked, out-of-repo script; empty `data/{raw,preprocessed/V_ACD_DEGREE_COURSE,...}` folders from the `ensure_dir` cascade) — harmless (outside `ACADEMIC_ADVISOR_DATA_DIR`, never read by the pipeline), left for user disposition since its exact origin could not be fully reconstructed. (2) `models/grade_model.lgbm`/`pass_model.lgbm` — legacy pre-rename duplicates flagged in Phase 8 — left in place, not archived, same reasoning.
- **Rollback (dual points):** (1) pre-7c baseline — `docs/manifests/baseline_7bc_2026-07-08.json` + `backup_7bc_2026-07-08/`. (2) post-freeze golden state — `docs/manifests/freeze_phase9_2026-07-08.json` + isolated copies in `freeze_phase9_2026-07-08/` (17 canonical artifacts, sha256-verified). Git: commit + annotated tag at the frozen point (see freeze doc for the exact hash/tag name).
- Paused pipeline logic rules remain in `docs/pipeline_rules.md` and git history, as instructed.

---

## 6. Sequential dependencies
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9. No parallel phases. Split integrity before all; ID/dtype before naming/relocation; ownership before placement; planning (4–6) before writes (7); within 7 one group at a time; rebuild from earliest confirmed defect behind its own gate; Phase 8 after all 7-groups; Phase 9 after 8.

---

## 7. Decision gates
KEEP (nothing) · RENAME LATER (7b, Phase 6 name) · MOVE LATER (7b, preserve first) · REBUILD (7c only, own gate, expected-delta) · ARCHIVE OLD (confirmed duplicate/obsolete DATA → `archive/`; deletion separate) · PRESERVE/REPOINT (supporting steps in order) · STOP–USER DECISION (insufficient evidence / naming conflict / two owners).

Evidence-before-action is unconditional: nothing moved/renamed/archived/rebuilt/deduplicated before ownership+lineage support it. No duplicate deleted on resemblance — tier verdict + lineage + own approval required.

---

## 8. Remediation safety rules (intra-tree)
Single resolved root (set/verify env var each context; repo shell stays empty). Preserve before change (copy, retire only after validation + repoint + approval). Ownership before movement. One group at a time. No silent overwrite (collision = STOP). Provenance retained (superseded data → `archive/`). Splits special (base moves/rebuilds only under immutability contract, single owner; enrichment = distinct derived artifact per D2). Rebuild last resort, from earliest confirmed defect, behind 7c gate.

---

## 9. Rollback & data-safety model
Git protects tracked code/notebooks/docs only — NOT gitignored parquet/CSV/split/model-data artifacts. Before any move/rename-with-copy/rebuild/overwrite-prone op on a data artifact, record: original path, size, content hash, schema, row count, key statistics. Preserved copies isolated from the working path (never beside the file they protect); high-value artifacts (base splits) prefer a dedicated backup folder no pipeline/remediation script writes to. Manifest lives with audit docs, never overwritten; Phase 9 snapshots it. Rollback: restore isolated copies, verify against manifest (hash + row count), halt until re-approved.

---

## 10. Risks & open items

**Resolved (see §1.2 for the code/data evidence):**
- C1 multi-writer of the feature-engineered artifact → **RESOLVED** (D1; `merge_diploma.py` neutralized; single writer confirmed by repo-wide grep and Phase 8/9 checks).
- C2–C4 split overwrite → **RESOLVED** (D2; distinct base/difficulty/final generations built and validated in 7c; bare names retired).

**Active risks:**
- **Feature-eng pass-through** — diploma columns now traverse `feature_engineering.py`; an unexcluded generic transform silently shifts splits. Statically clear (§1.4 items 3, 5); runtime confirmation is the mandatory 7c pre-check (§4.1).
- **Orphan fitted state** — `course_difficulty_lookup.parquet`, `knn_index.pkl`: live consumers, no reproducible committed producer; the diploma-bucketing fit state is not persisted at all. PRESERVE + USER DECISION; contract 14 makes persistence mandatory going forward. Nearly free to fix during the D2 rebuild: save the lookup from `course_difficulty`'s train aggregates, persist the bucketing map, and add a small `build_knn_index` script calling the existing `KNNAdvisor.build(df_train_final)`.
- **Dotted-ID corruption** — any float/`to_numeric` on a dotted ID silently breaks joins. Addendum: raw `diploma_type_id` is float-typed with a dotted suffix at source; the dead `fillna(6.111)` in `Academic_info_clean.ipynb` is a reminder of how easily float literals encode dotted IDs.
- **Env-var fragility + import side effect** — unset var → silent write to empty shell; `ensure_dir` masks it. Guard proposed (contract 12), implemented in 7a. The live hardcoded path in §1.4 item 11 points at the repo shell — a concrete instance of this risk class.
- **Twin-copy drift** — accidental unset-var run populates the repo shell. Only the active root is ever written.
- **mtime misreading** — identical split mtimes; never infer order/authority.
- **Doc drift** — stale markdown "Architectural Notes" (old paths/locations) in several notebooks; stale references to `Read.ipynb` and `merge_diploma.py`; `pipeline_rules.md` ownership wording pre-dates D1. Docs never override code; 7a/7b cleanups.
- **Broken diploma-source rebuild read — RESOLVED (7b, 2026-07-08).** The raw artifact was byte-parity-copied to the corrected name `v_add_academic_info.parquet`, the extractor repointed, and Chain A stage 1 executed successfully in 7c. (Historical flag text preserved in git history.) A second latent defect surfaced at 7c preflight and was closed by the stage-1 rebuild: the on-disk cleaned diploma artifact had `student_id` as float64 — the normalization code existed but the artifact predated it; `02_merge_diploma`'s dtype guard had correctly blocked all prior runs, which is why `merged_with_diploma.parquet` had never been materialized before 2026-07-08.
- **Latent extractor SQL bug (flag only — logic fix out of scope).** `extact_all_row_tables.ipynb`, V_ADD_ACADEMIC_INFO query: a missing comma makes `DIPLOMA_TYPE_SL` be aliased as `ACTIVE`; the real `ACTIVE` column is never selected (the `WHERE ACTIVE='A'` filter still applies). Currently harmless — the `active` column is dropped downstream — but the saved raw "active" column actually contains diploma-type text, and `diploma_type_sl` is silently unavailable. Any fix is a data-logic change requiring its own approval; not scheduled in this job. (Same notebook, temp-request cell: a missing comma in an unused `needed_columns` list and a `COURSE_CREITS` typo — inert, recorded only.)
- **Bare split-filename generation risk — RESOLVED (7b/7c, 2026-07-08).** All readers repointed to explicit generations (`_final` for training/results/exploration, `_difficulty` for the fallback diagnostic); bare-named files retired to `archive/pre_7c/`. Any stale reader now fails loudly.

**Open decisions (carry forward, do not resolve here):**
- **D5 — outlier branch — RESOLVED (user decision, 2026-07-07).** The outlier branch (`handeling_outliers.ipynb` / `without_outliers.parquet`) is **exploratory/inactive**: not part of the current production training pipeline and not part of the current rebuild. It may be revisited later as a separate controlled experiment. (2026-07-07 census supported this: no live training consumer.) Naming consequence (naming plan 6A): `handeling_outliers.ipynb` → `explore_outlier_removal.ipynb` in 7b; `without_outliers.parquet` stays a frozen exploratory artifact — optional archive later, no pipeline reader.

**Dispositions recorded 2026-07-07 (user decisions, pre-7a):**
- **`all.ipynb`** — archive as a dead prototype (repo `note_books/archive/`, 7b group); do NOT rename into the production pipeline. Permanent deletion stays a separate approval.
- **Root pre-drop ADD artifact** (`preprocessed/v_add_student_degree_status_clean.parquet`) — archive **only after** a consumer check confirms no live pipeline reader (7b gate; current evidence: only `trace_student` lists it).
- **`course_difficulty_lookup.parquet`** — ADOPT a reproducible committed producer (contract 14). Implementation lands with the D2 rebuild group (7c), where the train aggregates already exist in memory.
- **`knn_index.pkl`** — **HOLD (new open decision D7):** do not adopt or rebuild a producer until its role in the live architecture is confirmed. No 7-group may touch it before D7 is decided.
- **Extractor SQL defect** (missing comma aliasing `DIPLOMA_TYPE_SL` as `ACTIVE`, §10 flag) — tracked as a **separate data-logic defect**; explicitly excluded from naming-only and paths-only changes. Fixing it requires its own approval and an expected-delta declaration.
- **D6 — `diploma_gpa` full-source median fill — RESOLVED (user decision, 2026-07-07).** Surfaced 2026-07-07: `Academic_info_clean.ipynb` cell 7 computes the fill median over the entire diploma source at cleaning time — upstream of the split, fit on a population that includes future validation/test cohorts, conflicting with the letter of train-only leakage control. The user chose option (c): the current fill is **explicitly approved as a logged exception** for this cycle — the fill stays as-is, no remediation scheduled, Decisions_Log entry written. Rationale on record: the filled values are pre-admission facts (contract 13) and the affected count is small (~40 per Phase 3 evidence). The train-only rule stands unchanged for every future fitted statistic (contracts 14–15); this exception does not generalize. Stays closed; not reopened.
- **D7 — `knn_index.pkl` role in the live architecture — STILL OPEN.** HOLD stands; no 7-group touched it; not resolved by this freeze.
- **DataFrame naming (6B) — NOT EXECUTED, carried forward past freeze.** The naming plan's 6B proposals (`df_merge_test`→`df_crg_add`, `dfcrg_acd_add`→`df_crg_add_acd`, the `df`/`df1`/`df2` chains in `01_select_model_population`, `clean_v_acd_degree_course`, `clean_v_add_academic_info`, `clean_v_acs_grade`) were never applied in 7b. Verified still present 2026-07-08. Deliberately deferred rather than rushed at freeze time — bulk variable renaming across ~7 notebooks cannot be validated as behavior-preserving without full re-execution and re-diffing of each one, which is exactly the class of unreviewed change this freeze just had to investigate and close (see Phase 9 §5 incident). Cosmetic only; no governance-correctness impact.
- **Repo-shell `data/` debris and legacy `models/` duplicates — found 2026-07-08, deliberately not touched.** See Phase 9 §5 for full detail. Left for user disposition.
- **select denylist → allowlist** — `select.ipynb` is denylist-based, violating the project's own allowlist principle; diploma columns pass only because they're absent from drop lists. Two follow-ups: (a) the explicit survival assert — promised, **verified still absent from code**, scheduled 7a; (b) log the full conversion as tech debt for a later 7a item. Not part of this rebuild.

---

## 11. Locked objectives → coverage
Split integrity → P1. ID/dtype → P2. Lineage → P3. One-owner → P3/P5/P7a. Placement → P4/P7b. Duplicates/loose → P3/P4. Notebook naming/numbering → P6A/P7b. DataFrame naming → P6B/P7b. Artifact naming/traceability → P5/P6C/P7b. Contracts → P5. Fitted-state persistence & inference consistency → P5 (contracts 14–15)/P7a. Remediation → P7. Selective rebuild → P7c. Byte/logical/expected-delta → §4, applied P3/P7/P8. Freeze → P9.

---

## 12. Verification & approval
Validated against CLAUDE.md guardrails, `docs/pipeline_rules.md` locked decisions, and `docs/paths_audit.md`. Phases 1–9 complete. D1–D4 approved contract inputs. D5 resolved (outlier branch exploratory/inactive, 2026-07-07). D6 resolved (approved logged exception, 2026-07-07). **D7 (KNN index role) remains the sole open decision at freeze** — does not block the freeze (no 7-group ever touched `knn_index.pkl`; it is preserved, untouched, exactly as required by its HOLD status).

**Gate-interpretation change (recorded explicitly, 2026-07-07).** The pre-correction Revision 4 wording made D5 a blocker for a "clean Phase 5" approval. At Phase 5 execution the gate was interpreted differently — open decisions D5/D6 do not block drafting or approving the contracts; they block only the later 7c rebuild scoping — and Phase 5 was approved under that interpretation. This is a deliberate, recorded change of gate interpretation, not an oversight or contradictory wording. Both decisions were subsequently resolved, so the change has no residual effect.

**Full phase authorization history (superseding all earlier partial statements in this section):** Phase 5 approved 2026-07-07. Phase 6 approved 2026-07-07. Phase 7a executed 2026-07-07. Phase 7b and 7c executed 2026-07-08 (user-authorized) — every artifact move/rename/archive/rebuild recorded in §7's per-group entries and validated in `docs/manifests/validation_7c_2026-07-08.md`. Phase 8 executed 2026-07-08 — 51/51 checks, `docs/manifests/validation_phase8_2026-07-08.md`. **Phase 9 (final freeze) executed 2026-07-08** — `docs/manifests/freeze_phase9_2026-07-08.md`. This governance job is DONE, subject to the carried-forward open items recorded in §10 (D7, DataFrame naming 6B, repo-shell debris, legacy `models/` duplicates) — none of which are governance-correctness defects, all of which are explicitly non-blocking by design.
