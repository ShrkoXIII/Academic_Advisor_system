# Data Organization & Pipeline Governance — Phased Plan (Revision 4)

## 0. Status of this document

This is a **plan only**. Producing it executes nothing. Every phase runs only after explicit user approval, one phase at a time.

**Revision 4 supersedes Revision 3.** It records the architecture change decided after Phase 4 (diploma merge moved upstream to its own stage), the split-generation decision, layered validation, and a set of out-of-band code changes already applied. It keeps every Rev 3 rule not explicitly changed here.

**Honesty rule for this revision:** a decision being *approved* is not the same as it being *implemented*. Where the contract and the current code disagree, this document says so plainly (see §1 "Implementation status"). Phase 5 writes contracts (the target); Phase 7 makes the code match them. No item is marked resolved until code verification confirms it.

---

## 1. Context

**Completed prior job:** paths cleanup. Every notebook/script resolves paths from `src/paths.py`. This premise was briefly broken and is now restored — see the out-of-band record below.

**NOT complete — part of THIS job:** notebook naming, notebook numbering, DataFrame naming, saved-artifact naming governance and traceability.

**Scope of this job (unchanged from Rev 3):** split integrity; ID/dtype consistency; full lineage; one-owner-per-artifact governance; intra-tree placement; duplicate/loose-file review; notebook naming & numbering; DataFrame naming; artifact naming & traceability; governance contracts; controlled remediation; selective rebuild; byte/logical/expected-delta validation; final freeze.

**Data root (fixed fact):** the active root exists, is populated, and is protected. It resolves via `ACADEMIC_ADVISOR_DATA_DIR=D:/AI/data_clean_academic_advisor/data`, read-only during audit.

**Out of scope:** promotion/migration into `PROJECT_ROOT/data`; retiring/repointing `ACADEMIC_ADVISOR_DATA_DIR`; copying the cleaned tree into the repo shell; changing where the protected data physically lives.

**Wording rule (existence ≠ authority):** artifacts in the active root are *current on-disk artifacts*, never "trusted/authoritative" by default. Authority comes only from Phase 1–3 evidence.

### 1.1 Resolved decisions (D1–D4)

**D1 — Diploma merge moved upstream, resolves conflict C1 by design.**
A dedicated notebook (`02_merge_diploma`, current on-disk filename `02_final_mergerd_with_dimploma.ipynb` — flagged for rename, see §6) is inserted between `01_merge_crg_add_acd` and `select`. It reads the CRG+ADD+ACD merge output plus the cleaned diploma source, joins `diploma_type_id`/`diploma_gpa`, and writes a **distinct** artifact under `MERGE_DIR` (`merged_with_diploma.parquet`). It does **not** extend `after_fet_eng.parquet` in place. The old writer `merge_diploma.py` is retired from the intended DAG.

**D2 — Split generations become distinct artifacts, resolves C2–C4 by design.**
`model_data` splits become three distinct generations instead of in-place rewrites: base (`split_diagnostics.ipynb`) → difficulty-enriched (distinct file, `course_difficulty.ipynb`) → final (distinct file, `diploma_type_bucketing.ipynb`). No stage overwrites another stage's file. Concrete filenames deferred to Phase 6.

**D3 — Layered validation for the diploma rebuild chain (Chain A).** See §4.1.

**D4 — No separate pre-Phase-5 preservation gate.** A full backup of both project and data tree already exists (user-confirmed). Preservation of `df_train/valid/test` and the two orphan artifacts stays inside Phase 7 as scoped in Rev 3; the existing backup covers the interim.

### 1.2 Implementation status (contract vs. code — READ THIS)

| Item | Decision | Code on disk today | Status |
|------|----------|--------------------|--------|
| C1 — single writer of `after_fet_eng.parquet` | Resolved by design (D1) | `merge_diploma.py` still contains the old in-place write; grep shows `after_fet_eng` referenced by `handle_gpa` (writer), `merge_diploma.py` (retired writer, still on disk), `split_diagnostics` + `diploma_type_bucketing` (readers). No live caller invokes `merge_diploma.py`. | **REDESIGNED, ENFORCEMENT PENDING.** Fully resolved only after `merge_diploma.py`'s write is neutralized and a repo-wide grep confirms exactly one writer. |
| D2 — distinct split generations | Approved | No code reflects it yet. | **PLANNED, NOT IMPLEMENTED.** |
| C2–C4 — split in-place overwrite chain | To be removed by D2 | `diploma_type_bucketing.ipynb` still runs `df_train.to_parquet(TRAIN_PATH)` etc. — same path in, same path out. | **STILL ACTIVE.** This is the current reality D2 has not yet replaced. |
| `02_merge_diploma` notebook guards | Required (§4 contract) | New notebook exists on disk but its guards are not yet independently verified against code. | **UNVERIFIED — verify before Phase 7 claims it done.** |

Phase 5 may write the contracts for all of the above (a contract describes the target). No Phase 5 wording may state these are already satisfied.

### 1.3 Out-of-band changes already applied (record, do not re-schedule)

Six notebooks had live hardcoded `D:\` paths bypassing `src/paths.py`, found during Phase 3 cross-check, fixed manually, committed:
`select.ipynb`, `clean_v_crg_student_course.ipynb`, `handeling_outliers.ipynb`, `pipeline_run_judge_test.ipynb`, `add_student_degree_status_clean.ipynb`, `load_preprocessing.ipynb`.
These are already-applied; do NOT re-schedule as Phase 7 work. A final repo-wide re-grep is still required to confirm no live hardcoded data-root path remains.

Additional applied changes:
- Diploma-source normalization: `student_id` is now normalized with `cleaning_utils.normalize_id_columns` in the diploma-source cleaning notebook (`Read.ipynb`, renamed by user to `Academic_info_clean.ipynb`), at cleaning time — not patched at merge time. Guards added: row count unchanged, unique-key count unchanged, dtype confirmed `string`.
- `01_merge_crg_add_acd` output `merged_add_acd_crg.parquet` relocated from `features/` to `MERGE_DIR`. This **supersedes** the Phase 3 ledger finding that `features/` was its correct home (reason: `features/` is for derived features; this is a merge intermediate). Verified: no other reader still points at the old `features/` path.
- `select.ipynb` read-source updated to `MERGE_DIR / merged_with_diploma.parquet`.

**Decisions_Log entries required (each separate):** the six-notebook path-fix commit; D1; D2; D3; D4; the `merged_add_acd_crg.parquet` relocation with its superseded-finding note; and — logged separately from the path fix — `select.ipynb`'s D1 read-source change (two distinct changes to the same file, do not conflate).

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
| 5 | Governance contracts | Planning | Draft contracts incl. ownership, split-immutability, stage boundaries, rollback | Approve before 6 |
| 6 | Naming & numbering plan | Planning | Notebook + DataFrame + artifact-name proposals | Approve before 7 |
| 7 | Controlled remediation | **Write** | Gated groups; change → group validation → approval → promote | Per-group; 7c distinct |
| 8 | Final cross-pipeline validation | Read-only | Integration + drift screen | Approve before 9 |
| 9 | Final freeze | Read-only + git tag + manifest | Freeze checklist, lineage, dual rollback points | Done |

Phases 1–6 non-destructive. First disk/code change is Phase 7 only.

**Current position:** Phases 1–4 complete. Entering Phase 5 with D1–D4 as contract inputs and the §1.2 status table as binding truth.

---

## 4. Validation model

**A. Byte parity** — exact copy/move, no rerun. Hash + size equality. Any difference fails.

**B. Logical parity** — rerun after a non-semantic change (variable/notebook rename, path-only edit). Byte equality NOT required (parquet compression/metadata/row-group order aren't byte-stable). Compare: row counts; unique keys + duplicate counts; schema + dtypes; values sort-normalized on stable keys; null patterns; distributions. Requires determinism: fixed seeds, stable sort, declared float tolerance. Any delta beyond these fails.

**C. Expected delta** — intentional fix (ID normalization, dtype fix, overwrite prevention, logic correction). Result may change, but only where predicted, by an amount/type declared *before* the change, with no unexplained downstream drift.

### 4.1 Chain A layered validation (D3) — diploma redesign

Not a fourth type; a named application of A/B/C per stage:

| Stage | Type | Pre-declared expectation |
|-------|------|--------------------------|
| Cleaned diploma source | EXPECTED DELTA | `student_id` schema only: double → string. Identical row count. 1:1 mapping (zero ID collisions). Suffix preserved, null pattern unchanged. |
| Diploma merge (`02_merge_diploma`) | LOGICAL PARITY | Same left-side row count; no duplicate expansion; **row-count invariance in new left join,
no duplicate expansion,
parity on common comparable population,
downstream logical parity** (Phase 3 proved the ~48 unmatched rows / 6 old-cohort students are genuinely missing records, not float corruption). Any recovered or newly-lost match is a **STOP**. |
| `select` output | LOGICAL PARITY | Plus required-column survival: `diploma_gpa`, `diploma_type_id` must be present after selection (assert). |
| Feature-engineered artifact + final splits | LOGICAL PARITY | Unless a separately approved change introduces an explicit expected delta. |

**Mandatory Phase 7c pre-check (before any rebuild runs):** diploma columns now flow *through* `feature_engineering.py` instead of bypassing it. Verify no generic column-wide transform (train-only imputation, scaling) touches `diploma_gpa`/`diploma_type_id`. If one would, either exclude these columns or declare the delta explicitly. An unexcluded generic transform here is the single most likely silent parity failure in the whole rebuild.

---

## 5. Detailed phase definitions

### Phase 1 — Split integrity audit (read-only) — COMPLETE
As Rev 3. Deltas labeled OBSERVED/INFERRED/UNRESOLVED. HARD STOP only for missing/zero-size split or inability to inspect safely; multiple writers → flag CRITICAL, continue mapping, block remediation. `after_fet_eng.parquet` is a candidate reconstruction source only; authority deferred to Phase 3.

### Phase 2 — ID & dtype audit (read-only) — COMPLETE
As Rev 3. Confirmed CRITICAL: `student_id` float64-sourced/unnormalized on the diploma join path (now fixed at source, per §1.3). Three normalizer implementations confirmed. Dangerous-cast search is pattern-based, not exhaustive.

### Phase 3 — Ownership & lineage audit (read-only) — COMPLETE, with one superseded finding
As Rev 3, plus: the `merged_add_acd_crg.parquet` placement finding (`features/` = correct home) is **superseded** by the §1.3 relocation to `MERGE_DIR`. The original finding stays in the record, marked superseded and dated — not deleted.

Duplicate verification is two-tier: (A) exact binary duplicate = same hash+size; (B) logical duplicate = hash differs but passes §4B comparison. A hash mismatch never alone proves two parquets differ. Verdicts: BINARY DUPLICATE / LOGICAL DUPLICATE / DISTINCT / UNRESOLVED.

**Refreshed current-state DAG (post-D1/D2 target):**
```
raw → preprocessing (clean CRG/ADD/ACD/Diploma; normalize student_id at diploma source)
  → 01_merge_crg_add_acd → MERGE_DIR/merged_add_acd_crg.parquet
  → 02_merge_diploma      → MERGE_DIR/merged_with_diploma.parquet   (distinct)
  → select                → selected modeling population
  → handle_gpa / feature engineering → after_fet_eng   (single writer — enforcement pending)
  → split_diagnostics     → base splits
  → course_difficulty     → distinct difficulty-enriched splits   (D2 — not yet built)
  → diploma_type_bucketing→ distinct final-model splits           (D2 — not yet built)
  → training
Outlier branch (undecided): selected/feature frame → handeling_outliers.ipynb
  → without_outliers.parquet → no confirmed live training consumer.
```

### Phase 4 — Placement & action mapping (planning) — COMPLETE, interpreted through Rev 4
Action model unchanged (primary + ordered secondary + validation type + dependencies). Updates:

- **Merge artifacts** (`merged_add_acd_crg.parquet`, `merged_with_diploma.parquet`): Primary KEEP, location `MERGE_DIR`, provided filenames/ownership match implemented code.
- **`after_fet_eng.parquet`**: no longer blocked by the C1 ownership gap (D1). Action: MOVE LATER + REPOINT single producer + REPOINT consumers + PRESERVE. Validation byte parity for pure move, logical parity if rerun required. **Precondition:** code verification confirms a single current writer (see §1.2 — pending).
- **Selected modeling population** (`df_crg_add_acd.parquet`): `select` is no longer an audit step, so its output must leave `AUDIT_DIR`. Action: MOVE LATER + optional RENAME LATER + REPOINT CONSUMERS + PRESERVE. Byte parity.
- **`merge_diploma.py`** (superseded code): action KEEP-IN-PLACE + add a one-line top docstring marking it superseded and not called. **Do NOT move to `ARCHIVE_DIR`** — it is tracked code; git already preserves its history permanently. `ARCHIVE_DIR` is for gitignored data git cannot see (Rev 3 §9). Archiving tracked code adds nothing and blurs that distinction.
- **`diploma_type_bucketing.ipynb`**: its save cell still does the C2–C4 in-place overwrite and its failure message still says "run merge_diploma.py". Both are Phase 7 code fixes under D2. Flag now; fix in 7.
- **Debug residues** (`merged.csv`, `after_feature_eng_run.csv`): ARCHIVE OLD, subject to archive gate.
- **Orphans** (`course_difficulty_lookup.parquet`, `knn_index.pkl`): PRESERVE + USER DECISION (reconstruct a designated producer vs. accept as frozen legacy). Producer reconstruction is nearly free during the D2 rebuild — see §10 open items.

### Phase 5 — Governance contracts (planning) — ENTERING NOW
Draft the durable rules. Contract inputs now include D1–D4. Draft:

1. **ID/dtype contract** — per ID class: dtype + normalizer; dotted-suffix preservation; ban on float joins.
2. **Ownership contract** — each artifact path has exactly one writer-owner. A consumer may read upstream and write its own distinct downstream artifact; it must NOT rewrite an upstream path unless designated owner. Encode `after_fet_eng.parquet`'s single writer = the feature-engineering stage (`handle_gpa`), with C1 marked enforcement-pending until §1.2 clears.
3. **Split immutability** — one owner for base splits; enrichment writes distinct derived artifacts, never in-place (D2). Encode as a rule even though C2–C4 code is still active.
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
7. **Artifact naming** — encode stage/content/ownership/derivation.
8. **Notebook numbering** — reflect real execution order.
9. **DataFrame naming** — semantic by source/domain + stage.
10. **Remediation safety** — preserve-before-change; update-consumers-after-ownership-confirmed; no silent overwrite; one group at a time; validate before promote.
11. **Rollback/baseline-manifest** — git for code; manifest + isolated copies for gitignored data (§9). Define manifest format, capture timing, isolation requirement.
12. **Data-root guard** — startup assertion that resolved splits exist and are nonzero; mitigate the `ensure_dir` import side effect. Proposes the code change; implementation is a gated 7a item.
13. **Pre-admission-fact clause** — `diploma_type_id`/`diploma_gpa` are known before the target semester; joining them upstream does not leak. The full-source median fill for `diploma_gpa` (40 values) is a documented, logged exception.

Guardrails: contracts must not weaken locked decisions in `pipeline_rules` (temporal split boundaries, `feature_contract.json` allowlist, train-only leakage control, no `scale_pos_weight`/SMOTE, `finish_status` target). Conflict → surface, don't encode.

### Phase 6 — Naming & numbering (planning)
Notebooks, DataFrames, saved artifacts. New targets to name: `02_final_mergerd_with_dimploma.ipynb` → fix typos ("mergerd"→"merged", "dimploma"→"diploma") and assign stage number; `merged_with_diploma.parquet`; the difficulty-enriched and final split filenames (D2). Honor: one merge notebook per stage; early merge and diploma merge stay separate; no two files differing only by case.

### Phase 7 — Controlled remediation (WRITE; gated groups)
Per-group flow: change → immediate group validation (type per Phase 4 record) → user approval → promote/retire preserved originals → next. No parallel execution.

- **7a — Code & ownership fixes.** Neutralize `merge_diploma.py`'s in-place write (superseded docstring; confirm single writer of `after_fet_eng`); fix `diploma_type_bucketing.ipynb` save cell to distinct paths (D2) and its stale failure message; the diploma-source normalization is already applied (verify only); guard assertions; data-root guard.
- **7b — Relocations & renames.** Move `after_fet_eng` and the selected-population artifact out of `AUDIT_DIR`; apply Phase 6 names. Copy never cut; repoint consumers only after ownership confirmed; no silent overwrite.
- **7c — Selective rebuild.** Distinct approval gate. Run the Chain A pre-check (§4.1) FIRST. Rebuild starts from the earliest actually-affected stage: diploma-source preprocessing → `02_merge_diploma` → downstream. Do NOT rebuild `01_merge_crg_add_acd`'s logic — it reads only cleaned CRG/ADD/ACD, unaffected by the diploma-source fix — but DO align its output save target to `MERGE_DIR` (folder change, not logic change). Validation per §4.1.

Ordering: 7a → 7b → 7c is default, not law; evidence-driven per dependency chain. The 7c gate applies to every rebuild wherever it lands.

Guardrails: never overwrite a current on-disk artifact without an isolated preserved copy + manifest. Do not modify split logic, saved splits, target definitions, or `feature_contract.json` without explicit approval. Superseded DATA → `archive/` first; permanent deletion needs separate approval. Minimal changes.

### Phase 8 — Final cross-pipeline validation (read-only)
Not the first validation (each 7-group self-validated). Final integration pass: end-to-end consistency; split boundaries; feature columns vs `feature_contract.json`; target distributions; join integrity; ID suffixes; drift screen vs baseline manifests. Reuse `scripts/parity_check.py`. Any unexplained drift → STOP.

### Phase 9 — Final freeze
Checklist: one owner per artifact; one owner for base splits; no silent overwrites; clean tree; stable schema contract; meaningful notebook order/names; meaningful DataFrame names; documented lineage; reproducible rebuild sequence. Rollback: git tag for code; baseline manifest + isolated copies for data. Keep paused logic rules in `docs/pipeline_rules.md` and git.

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

**Resolved by design (enforcement/build pending — see §1.2):**
- C1 multi-writer of `after_fet_eng` → redesigned (D1), enforcement pending (`merge_diploma.py` write still on disk; single-writer grep still required).
- C2–C4 split overwrite → to be removed by D2; **still active in code** until `diploma_type_bucketing.ipynb` writes distinct files.

**Active risks:**
- **Feature-eng pass-through** — diploma columns now traverse `feature_engineering.py`; an unexcluded generic transform silently shifts splits. Mandatory 7c pre-check (§4.1).
- **Orphan artifacts** — `course_difficulty_lookup.parquet`, `knn_index.pkl`: live consumers, no reproducible committed producer. PRESERVE + USER DECISION. Nearly free to fix during the D2 rebuild: add a save of the lookup from `course_difficulty`'s train aggregates, and a small `build_knn_index` script calling the existing `KNNAdvisor.build(df_train_final)`.
- **Dotted-ID corruption** — any float/`to_numeric` on a dotted ID silently breaks joins.
- **Env-var fragility + import side effect** — unset var → silent write to empty shell; `ensure_dir` masks it. Guard proposed (contract 12), implemented in 7a.
- **Twin-copy drift** — accidental unset-var run populates the repo shell. Only the active root is ever written.
- **mtime misreading** — identical split mtimes; never infer order/authority.

**Open decisions (carry forward, do not resolve here):**
- **D5 — outlier branch** (`handeling_outliers.ipynb` / `without_outliers.parquet`): exploratory / abandoned / adopt into live training. Recommendation on record: declare exploratory for this rebuild, log it, revisit as a separate post-freeze experiment. **Awaiting user confirmation — this is the last decision blocking a clean Phase 5.**
- **select denylist → allowlist** — `select.ipynb` is denylist-based, violating the project's own allowlist principle; diploma columns pass only because they're absent from drop lists. Two follow-ups: (a) add the explicit survival assert now (cheap), (b) log the full conversion as tech debt for a later 7a item. Not part of this rebuild.

---

## 11. Locked objectives → coverage
Split integrity → P1. ID/dtype → P2. Lineage → P3. One-owner → P3/P5/P7a. Placement → P4/P7b. Duplicates/loose → P3/P4. Notebook naming/numbering → P6A/P7b. DataFrame naming → P6B/P7b. Artifact naming/traceability → P5/P6C/P7b. Contracts → P5. Remediation → P7. Selective rebuild → P7c. Byte/logical/expected-delta → §4, applied P3/P7/P8. Freeze → P9.

---

## 12. Verification & approval
Validated against CLAUDE.md guardrails, `docs/pipeline_rules.md` locked decisions, and `docs/paths_audit.md`. Phases 1–4 complete. D1–D4 are approved contract inputs; the §1.2 status table is binding — C1 enforcement-pending, D2 not implemented, C2–C4 still active. Before Phase 5 writes are approved, confirm D5 (outlier branch). Phase 5 drafts contracts only; no code/data touched.

**STOP — awaiting D5 decision, then Phase 5 approval.**