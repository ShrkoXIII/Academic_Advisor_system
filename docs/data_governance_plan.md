# Data Organization & Pipeline Governance — Phased Plan (Revision 3)

## 0. Status of this document

This is a **plan only**. Producing it executes nothing: no audit runs, no file is moved, renamed, rebuilt, archived, or deleted, no code or data is modified. Every phase runs only after explicit user approval, one phase at a time.

**Revision 3 supersedes all earlier revisions.** It adds six final corrections: binary-vs-logical duplicate verification, OBSERVED/INFERRED/UNRESOLVED delta labeling, candidate-only status for the reconstruction source, a clarified ownership contract, corrected env-var wording, and isolated preserved-copy rollback for high-value artifacts.

---

## 1. Context (corrected)

**Completed prior job:** paths cleanup (tracked in `docs/paths_audit.md`). Every notebook/script now resolves paths from `src/paths.py`. That job touched **paths only**.

**NOT complete — part of THIS job:**
- Notebook naming
- Notebook numbering
- DataFrame naming
- Saved-artifact naming governance and traceability

**Scope of this job:**
1. Split integrity and overwrite-chain audit
2. ID and dtype consistency audit
3. Full pipeline lineage
4. One-owner-per-artifact governance
5. Intra-tree file placement cleanup
6. Duplicate and loose-file review
7. Notebook naming and numbering
8. DataFrame naming
9. Saved-artifact naming and traceability
10. Governance contracts
11. Controlled remediation
12. Selective rebuild only when evidence requires it
13. Byte-parity / logical-parity / expected-delta validation
14. Final pipeline freeze

**Data root (fixed fact, not a question):**
The active data root exists, is populated, and is **protected**. It resolves via `ACADEMIC_ADVISOR_DATA_DIR=D:/AI/data_clean_academic_advisor/data`. The governance audit operates on this resolved active root, read-only.

**Explicitly OUT of scope for this job:**
- Promotion or migration of data into `PROJECT_ROOT/data`
- Retiring or repointing `ACADEMIC_ADVISOR_DATA_DIR`
- Copying the cleaned tree into the repo `data/` shell
- Changing where the protected data physically lives

External-vs-project data-root relocation is a **separate future task** and appears nowhere in this plan's phases, gates, or migration rules.

**Wording rule (existence ≠ authority):**
Artifacts in the active root are called **current on-disk artifacts** (or *current real artifacts*). They are never called "trusted" or "authoritative" by default. Authority is established only through Phase 1–3 evidence.

---

## 2. Pre-flight record (already executed and resolved)

The mandatory Phase 1 opening visibility check has already run:

- With the env var unset, `src.paths` fell back to the empty repo `data/` shell → correctly STOPPED.
- With `ACADEMIC_ADVISOR_DATA_DIR` set to the active root, the check **PASSED**: `df_train.parquet` 19,748,821 B, `df_valid.parquet` 6,729,784 B, `df_test.parquet` 4,760,639 B (all mtime 2026-07-02 16:51).

Standing caveats recorded from this event:

1. **Nonzero size proves availability only** — never correctness, ownership, or integrity.
2. **mtimes are diagnostic metadata only.** Do not infer chronology, authority, or execution order from mtime alone.
3. **`src/paths.py` has an import-time write side effect**: `ensure_dir(...)` creates directories on import. That import is what created the empty repo shells. Therefore, during all read-only phases: **do not re-import `src.paths` to resolve paths.** Reuse the already-recorded resolved root string. If any future step genuinely requires code execution with filesystem side effects, report it and request approval instead of silently doing it.
4. **Do not assume environment-variable persistence across separate command or tool execution contexts.** Each audit command context must explicitly verify or set `ACADEMIC_ADVISOR_DATA_DIR` before accessing the active data root.
5. Audit outputs (reports, ledgers, tables) are documentation deliverables (chat or `docs/`), **never written inside the data tree**.

---

## 3. Phase overview

| # | Phase | Mode | Expected output | Gate |
|---|-------|------|-----------------|------|
| 1 | Split integrity audit | Read-only | Split writer/reader/overwrite map (deltas labeled OBSERVED/INFERRED/UNRESOLVED), base-vs-enriched verdict, reconstruction-capability finding | Approve before 2 |
| 2 | ID & dtype audit | Read-only | ID-class taxonomy, normalizer inventory, dangerous-cast list, join-key consistency map | Approve before 3 |
| 3 | Artifact ownership & lineage audit | Read-only | Full artifact ledger + execution DAG + conflicts + duplicates (binary/logical verified) + loose files | Approve before 4 |
| 4 | Placement & action mapping | Planning only | Per-artifact action record: primary action + ordered secondary actions + validation type + dependencies | Approve before 5 |
| 5 | Governance contracts | Planning only | Draft contracts incl. clarified ownership contract and artifact rollback / baseline-manifest contract | Approve before 6 |
| 6 | Naming & numbering plan | Planning only | Notebook proposals + DataFrame proposals + artifact-name proposals (concrete target names for Phase 4 rename actions) | Approve before 7 |
| 7 | Controlled remediation | **Write** | Gated groups; each group: change → immediate group validation → approval → promote | Per-group gates; 7c has its own distinct gate |
| 8 | Final cross-pipeline validation | Read-only | Integration validation + unexplained-drift screen across the whole pipeline | Approve before 9 |
| 9 | Final freeze | Read-only + git tag + manifest snapshot | Freeze checklist, documented lineage, code rollback point (git) + data rollback point (baseline manifest + isolated preserved copies) | Job complete |

Phases 1–6 are non-destructive. The first change to disk or code happens only inside Phase 7, only after Phases 1–6 are approved.

---

## 4. Validation model (used by Phase 7 group gates, Phase 8, and duplicate verification)

Three validation types. Every Phase 4 action record declares which type applies.

**A. Byte parity** — when an exact file copy/move is expected (no rerun involved). Test: file hash equality (plus size). Any difference is a failure.

**B. Logical parity** — when code is rerun after a *non-semantic* change (DataFrame variable rename, notebook rename, path-only code adjustment). Byte equality is NOT required (parquet compression, metadata, row-group layout, and write timestamps are not byte-stable). Compare instead:
- row counts
- unique keys and duplicate counts
- schema and dtypes
- values, compared **sort-normalized on stable keys**
- null patterns
- relevant distributions

Determinism requirement: logical parity is only meaningful if the rerun is deterministic — fixed seeds, stable sort before comparison, and declared float tolerance. A non-semantic change that shows any delta beyond these rules is a failure.

This same comparison method is the standard for **logical-duplicate determination** in Phase 3 (Section 5, Phase 3).

**C. Expected delta** — for intentional fixes (ID normalization, dtype fixes, overwrite prevention, logic corrections). The result may change, but only:
- where predicted,
- by an approved amount/type declared **before** the change,
- with no unexplained downstream drift.

A delta that does not match the pre-declared expectation is a failure.

---

## 5. Detailed phase definitions

### Phase 1 — Split integrity audit (read-only; highest priority)

**Scope.** Establish whether `df_train/df_valid/df_test` are intact, who writes them, who reads them, and whether the on-disk files are base or enriched.

**Opening visibility check:** ✅ DONE — PASSED (Section 2). Not repeated. Path resolution reuses the recorded root; no `src.paths` import.

**Evidence required:**
1. **Original writer** of each split. Declared owner per `pipeline_rules`: `split_diagnostics.ipynb` — verify against code, don't assume.
2. **All later writers.** Known candidates to verify (neither assumed to be an actual writer until code inspection confirms it):
   - `course_difficulty.ipynb`
   - `diploma_type_bucketing.ipynb`
   Additionally, search the full codebase for **any other direct or indirect writer** of `df_train.parquet`, `df_valid.parquet`, `df_test.parquet` (direct `to_parquet` calls, save helpers, scripts, utility wrappers).
3. **All readers** (training notebooks, `results_analysis.ipynb`, `read.ipynb`, KNN build, etc.).
4. **Overwrite chain:** does the declared owner produce base splits that a later notebook then rewrites in place? If so, the current files are enriched, not base.
5. **Row / column / dtype deltas** between base-write stage and current on-disk state. **Every claimed delta must be labeled:**
   - **OBSERVED** — directly verified against an existing on-disk artifact or a preserved baseline (e.g., a parquet schema footer read of the current file describes the current state and is OBSERVED for that state only).
   - **INFERRED** — derived from code reading (e.g., "this notebook adds column X before saving") with no preserved baseline to confirm it actually happened on the current files.
   - **UNRESOLVED** — evidence insufficient to classify.
   Code-inferred row/column/dtype changes are **never presented as directly observed facts** when no preserved baseline exists. Method for current-state observation: parquet schema footer reads (`pyarrow.parquet.read_schema` — metadata-only, no data loaded); nothing is rerun.
6. **Reconstruction capability (candidate-source status only):** `after_fet_eng.parquet` is treated strictly as a **candidate** reconstruction source. Phase 1 may assess whether it is *technically capable* of reproducing the temporal split deterministically (boundary Train 2005–2021 / Valid 2022–2023 / Test 2024 + 2025S1) — but must **not** declare it authoritative. Final reconstruction authority depends on Phase 3 ownership and lineage evidence; note that this file itself carries an unresolved multi-writer conflict pending Phase 3.

**Output.** Split integrity report: writer/reader map, overwrite chain, base-vs-enriched determination, delta summary with OBSERVED/INFERRED/UNRESOLVED labels, reconstruction-capability finding (candidate status, authority deferred).

**Stop conditions:**
- **HARD STOP** only for: a required split file missing, a required split file zero-size, or inability to inspect further safely.
- **Multiple writers discovered** → flag **CRITICAL**, do NOT stop. Continue the safe read-only mapping, identify all writers/readers/overwrite paths completely, and **block remediation** on the conflict until it is resolved in Phases 4–7.

**Guardrails.** Read-only. Do not execute any split-writing notebook. Do not touch split parquets or `feature_contract.json`. Existence ≠ authority; mtime ≠ chronology.

### Phase 2 — ID & dtype audit (read-only)

**Scope.** Classify ID columns and every cast applied to them, so later remediation never breaks join identity. Dotted IDs (e.g. `15.111`) are string identity, not decimals, normalized via `cleaning_utils.normalize_id_series`.

**Evidence required:**
1. Classify each flagged key — `student_id`, `course_id`, `degree_id`, `faculty_id`, `student_course_id`, `student_status_id`, `part_id`, `requirement_type_id`, `diploma_type_id`, semester keys, categorical codes, surrogate keys — into: dotted identity ID / semester key / categorical code / surrogate key. One dtype or one normalizer does not fit all.
2. Inventory all normalizers: `normalize_id_series` and any ad-hoc implementations.
3. Dangerous casts: grep real code for `astype(...)`, `pd.to_numeric(...)`, float/int/string conversions on any key column — flag anything that could turn a dotted ID into a float or drop the suffix.
4. Join-key consistency: for every merge, confirm both sides share dtype and normalization end-to-end.

**Output.** ID-class taxonomy + normalizer inventory + ranked dangerous-cast list + join-key consistency map.

**Stop condition.** A confirmed float/decimal cast on a dotted identity ID on a live join path → flag **CRITICAL**, record as a defect for Phases 4/7, continue the audit. Do not fix in place.

**Guardrails.** Read-only. Trust executed code over docstrings; report disagreements.

### Phase 3 — Artifact ownership & full lineage audit (read-only)

**Scope.** Complete ledger of every artifact: exact current path, semantic meaning, producing notebook/script, all readers, all writers, overwrite behavior, upstream source, downstream consumers, and whether its current folder matches its role.

**Known lineage clues to verify (not conclude):**
- `merge/01_merge_crg_add_acd.ipynb` → writes `preprocessed/merge/merge_crg_add.parquet` and `features/merged_add_acd_crg.parquet`.
- `select.ipynb` → reads `features/merged_add_acd_crg.parquet`, writes `audit/df_crg_add_acd.parquet`.
- `handle_gpa.ipynb` → reads `audit/df_crg_add_acd.parquet`, writes root `merged.csv` and `audit/after_fet_eng.parquet`.
- `handeling_outliers.ipynb` → reads `audit/df_crg_add_acd.parquet`, writes `final/without_outliers.parquet`.
- `split_diagnostics.ipynb` → reads `audit/after_fet_eng.parquet`, writes `model_data/` splits.
- `course_difficulty.ipynb` → writes `artifacts/course_difficulty_lookup.parquet` **and is a candidate split writer (Phase 1)**.
- KNN build → `artifacts/knn_index.pkl`.

**Suspicious items (evidence before verdict — do NOT pre-decide location):**
- Multiple-writer conflict on `after_fet_eng.parquet`: declared owner `feature_engineering.py` vs. actual writers `handle_gpa.ipynb` and `merge_diploma.py`. How many real writers, which is canonical, is `audit/` the right home.
- `audit/` holding pipeline inputs, not reports: `audit/df_crg_add_acd.parquet` and `audit/after_fet_eng.parquet` are consumed downstream.
- `features/merged_add_acd_crg.parquet` placement vs `preprocessed/merge/`.
- Loose root files: `merged.csv` (written by `handle_gpa.ipynb`) and `after_feature_eng_run.csv` (owner unknown) — owner, purpose, obsolete/duplicate status.
- Duplicate ADD clean output: `preprocessed/v_add_student_degree_status_clean.parquet` vs `preprocessed/V_ADD_STUDENT_DEGREE_STATUS/clean_v_add_student_degree_status.parquet` — which is live, is the other dead.
- `audit/` subfolders `matching_status_course/`, `V_ADD_STUDENT_DEGREE_STATUS/` — writer, purpose, report-only vs input.

**Duplicate verification method (two tiers):**

- **A. Exact binary duplicate** — same content hash (and same size where applicable). Proves duplication outright.
- **B. Logical data duplicate** — hash differs, but the artifacts are: schema-compatible, same row count, same stable keys, same values after deterministic key-based normalization (sort-normalized on stable keys, declared float tolerance), same null patterns — i.e., they pass the logical-parity comparison of Section 4B.

**A hash mismatch does not prove two parquet artifacts are logically different** — compression, row-group layout, and writer metadata differ between writes of identical data. Every candidate pair receives one verdict: **BINARY DUPLICATE / LOGICAL DUPLICATE / DISTINCT / UNRESOLVED**. "Looks similar" is never sufficient evidence for any action.

**Output.** Artifact ledger (one row per artifact) + execution-order DAG + multiple-writer conflict list + duplicate set with tier verdicts + loose-file set.

**Stop condition.** Two genuine independent writers with no documented exception → flag **CRITICAL** governance defect for Phase 5, continue the audit.

**Guardrails.** Read-only. Inspect real code, never infer from filename/title. Existence of a file is not proof of where it belongs. Logical-duplicate comparison loads data read-only and writes nothing.

### Phase 4 — Placement & action mapping (planning only)

**Scope.** Turn Phase 1–3 evidence into a per-artifact decision map. No disk changes.

**Action model (multiple ordered actions allowed):**

Each artifact record contains:
- **Primary action**
- **Optional ordered secondary actions**
- **Required validation type** (byte parity / logical parity / expected delta, per Section 4)
- **Dependency notes** (what must happen before/after, and in which remediation group)

Action vocabulary: `KEEP · MOVE LATER · RENAME LATER · REBUILD · ARCHIVE OLD · DUPLICATE REVIEW · PRESERVE · REPOINT CONSUMERS · USER DECISION REQUIRED`.

Valid combinations include, for example:
- `MOVE LATER + RENAME LATER`
- `REBUILD + ARCHIVE OLD`
- `PRESERVE + REPOINT CONSUMERS`
- `DUPLICATE REVIEW + USER DECISION`

**Handoff rule:** Phase 4 assigns action *classes* only. Concrete target names for every `RENAME LATER` come from Phase 6. Both feed Phase 7b together.

**Guardrails.** Planning only — nothing is authorized to run. Nothing becomes `ARCHIVE OLD` without a confirmed duplicate/obsolete finding (tier verdict from Phase 3); nothing becomes `REBUILD` without a confirmed upstream defect. Current on-disk artifacts are not automatically authoritative because they exist.

**Stop condition.** Any artifact whose correct home or action cannot be decided from evidence → `USER DECISION REQUIRED`, no guessing.

### Phase 5 — Governance contracts (planning only)

Draft the durable rules the remediated pipeline will hold to:

1. **ID/dtype contract** — per ID class: required dtype and normalizer; dotted-suffix preservation; ban on float joins.
2. **Artifact ownership contract (clarified)** — each artifact **path** has exactly one writer-owner. A consumer may read an upstream artifact and write its own **distinct downstream artifact**. A consumer must **not** rewrite the upstream artifact path unless explicitly designated as that artifact's owner under an approved contract. Readers never rewrite.
3. **Split immutability policy** — one owner for base `df_train/valid/test`; enrichment (e.g. diploma bucketing, course difficulty) writes a distinct derived artifact instead of rewriting base in place, if Phase 1 confirms in-place rewriting; how base splits are preserved/versioned.
4. **Overwrite policy** — when in-place rewrite is allowed (idempotent, documented, by the designated owner only) vs. forbidden.
5. **Stage boundaries** — clean → merge(CRG+ADD+ACD) → feature select → feature eng → diploma merge → split → course difficulty → diploma bucketing → training → KNN → recommendation; which folder each stage owns.
6. **Artifact naming convention** — names encode stage/content/ownership/derivation.
7. **Notebook numbering convention** — numbers reflect real execution order.
8. **DataFrame naming convention** — semantic names by source/domain + stage.
9. **Remediation safety policy** — preserve-before-change, update-consumers-after-ownership-confirmed, no silent overwrite, one group at a time, validate before promote (Section 8).
10. **Artifact rollback / baseline-manifest contract** — because git does not protect gitignored data, define the manifest format (Section 9), when it must be captured, where it lives, and the isolation requirement for preserved copies.
11. **Data-root guard proposal** — a startup assertion that the resolved splits exist and are nonzero, plus removal or mitigation of the `ensure_dir` import-time side effect in `src/paths.py`. This contract *proposes* the code change; implementing it is a Phase 7a item behind its own approval.

**Guardrails.** Planning only. Contracts must not weaken locked decisions in `pipeline_rules` (temporal split boundaries, `feature_contract.json` allowlist, train-only leakage control, no `scale_pos_weight`/SMOTE, `finish_status` as the pass/fail contract).

**Stop condition.** A proposed contract conflicts with a locked decision → surface the conflict, do not encode it.

### Phase 6 — Naming & numbering plan (planning only; three sub-sections)

This is where the naming governance that was **not** completed by the paths job gets planned.

**A — Notebooks.** For each notebook: actual purpose (from code, not title), real pipeline stage, proposed stage number, proposed meaningful name, confidence, conflicts. Honor CLAUDE.md rules: one merge notebook per stage; the early CRG+ADD+ACD merge and the later diploma merge stay separate; no two files differing only by letter case.

**B — DataFrames.** Audit long-lived variables (`df`, `df1`, `df2`, `df_clean`, `df_merge`, `temp`, `data`) and propose semantic names by source/domain + processing stage.

**C — Saved artifacts.** Concrete target filenames for every artifact marked `RENAME LATER` in Phase 4, consistent with the Phase 5 naming contract, with a traceability note (old name → new name → owner → stage).

**Output.** Three proposal tables. Renaming remains a later, separately-gated execution decision.

**Stop condition.** Two notebooks contend for the same stage number/name → `USER DECISION`.

### Phase 7 — Controlled remediation (WRITE; gated groups)

Only phase that changes disk or code. Executes only actions approved in Phases 4–6.

**Per-group flow (validation is inside Phase 7, per group):**

```
change (one group)
  → immediate group-specific validation (type per the Phase 4 record)
  → user approval to promote
  → promote / retire preserved originals only if approved
  → next group
```

No group starts before the previous group is validated and approved. No parallel execution.

**Groups:**
- **7a — Code & ownership fixes.** Collapse multiple-writers to single owners, fix confirmed dangerous ID casts, remove dead writes, implement approved guard assertions. Preserve the source artifact (Section 9) before altering its producer. Validation: expected delta (or logical parity when the fix is declared non-semantic).
- **7b — Intra-tree relocations & renames.** Move/rename only artifacts marked in Phase 4, using target names from Phase 6. Copy, never cut; repoint consumers only after ownership confirmed; no silent overwrite. Validation: byte parity for pure copies/moves; logical parity for reruns after rename-only code changes.
- **7c — Selective rebuild / regeneration.** ⚠️ **Separate explicit approval gate, distinct from 7a/7b — always.** No rebuild, no regeneration, no downstream rerun without it. Rebuild starts from the **earliest confirmed defect stage**, never a default stage:
  - preprocessing ID corruption → rebuild from the affected preprocessing stage
  - feature-engineering defect → rebuild from the feature-engineering stage
  - split-only defect → rebuild from the split stage
  Validation: expected delta, pre-declared per artifact.

**Ordering.** 7a → 7b → 7c is the **default**, not a law. The actual sequence is evidence-driven per dependency chain from the Phase 4 dependency notes. Examples: a confirmed ID fix may require a rebuild before a relocation; an ownership/path fix may require a relocation before a later rebuild. Whatever the order, the 7c approval gate applies to every rebuild wherever it lands in the sequence.

**Guardrails.** Never overwrite a current on-disk artifact without an isolated preserved copy and manifest record. Do not modify split logic, saved split parquets, target definitions, or `feature_contract.json` without explicit approval. Superseded files go to `archive/` (`paths.ARCHIVE_DIR`) first; permanent deletion always requires separate explicit approval. Minimal changes only.

**Stop condition.** Any group validation fails, or any output changes unexpectedly during a supposedly non-semantic operation → stop, roll back that group from the isolated preserved baseline, report.

### Phase 8 — Final cross-pipeline validation (read-only)

Phase 8 is **not** the first validation — every Phase 7 group was already validated at its own gate. Phase 8 is the final integration pass:

- End-to-end pipeline consistency across all remediated groups together
- Cross-artifact integration checks: split boundaries, feature columns vs `feature_contract.json`, target distributions, join integrity, ID suffixes
- Final unexplained-drift screen against the baseline manifests
- Reuse `scripts/parity_check.py` where applicable

**Stop condition.** Any unexplained drift → stop, do not proceed to freeze.

### Phase 9 — Final pipeline freeze

**Freeze checklist:** one owner per artifact; one owner for base splits; no silent overwrites; clean organized tree at the active data root; stable schema contract; meaningful notebook order and names; meaningful DataFrame names; documented lineage; reproducible rebuild sequence.

**Rollback points (two mechanisms, Section 9):**
- **Code/notebooks/docs:** git tag/commit.
- **Data artifacts:** the baseline manifest snapshot plus isolated preserved copies. Git does not protect these.

Do not delete the paused logic rules — they live in `docs/pipeline_rules.md` and git history.

---

## 6. Sequential dependencies

Strictly sequential phases: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9. No parallel phase execution. Hard ordering rules:

1. Split integrity (1) before everything else.
2. ID/dtype correctness (2) before naming or any relocation planning.
3. Ownership & lineage (3) before any placement decision (4).
4. All planning (4–6) before any write (7).
5. Within 7: one group at a time, each group validated and approved before the next; group order defaults to 7a → 7b → 7c but follows the evidence-driven dependency chains from Phase 4.
6. Rebuild (7c) starts from the earliest confirmed defect, never before evidence, always behind its own gate.
7. Phase 8 runs only after every Phase 7 group is promoted; Phase 9 only after Phase 8 passes.

---

## 7. Decision gates

| Gate | Trigger | Authorizes |
|------|---------|------------|
| KEEP | Correct folder, single owner, correct dtype | Nothing — leave as-is |
| RENAME LATER | Right location/owner, unclear name | 7b rename (target name from Phase 6), validated per declared type |
| MOVE LATER | Right owner, wrong folder | 7b relocate (preserve original first), validated per declared type |
| REBUILD | Confirmed upstream defect / ownership break | 7c only, behind its own gate, expected-delta validated |
| ARCHIVE OLD | Confirmed duplicate (BINARY or LOGICAL tier verdict) / confirmed obsolete | Move to `archive/`; permanent deletion = separate explicit approval |
| PRESERVE / REPOINT CONSUMERS | Supporting actions in an ordered chain | The named step within its group, in dependency order |
| STOP — USER DECISION | Evidence insufficient / naming conflict / two legitimate owners | Halt; ask the user before any action |

An artifact may carry a **primary action plus ordered secondary actions**; each action in the chain passes through its matching gate, in the order fixed by the Phase 4 dependency notes.

**Evidence-before-action (unconditional):** no artifact is moved, renamed, archived, rebuilt, or deduplicated before ownership and lineage evidence supports it. No duplicate is deleted merely because two files look similar — a Phase 3 tier verdict (binary or logical) plus lineage evidence is required, and deletion still needs its own approval.

---

## 8. Remediation safety rules (intra-tree)

All cleanup semantics are **intra-tree** within the active data root. No destination outside that tree is planned, invented, or assumed.

1. **Single resolved root.** Each command context explicitly verifies or sets `ACADEMIC_ADVISOR_DATA_DIR` before accessing the active data root. Nothing is ever read from or written to the empty repo `data/` shell; any accidental write there is itself a defect to report.
2. **Preserve before change.** Every relocation copies (never cuts) the source first; the original is retired only after the target is validated, consumers repointed, and retirement approved.
3. **Ownership before movement.** No artifact moves until Phase 3 confirms its single writer-owner and all readers.
4. **One group at a time.** Change → validate → approve → promote → next.
5. **No silent overwrite.** A move never writes over an existing file; a name/placement collision is a STOP.
6. **Provenance retained.** Superseded/duplicate files go to `archive/` with lineage recorded — never straight to deletion.
7. **Splits are special.** Base splits move or rebuild only under the split-immutability contract, with the confirmed single owner as sole writer; any enrichment becomes a distinct derived artifact if Phase 1 confirms in-place enrichment is happening.
8. **Rebuild is last resort.** Prefer move/rename over regeneration; rebuild only from the earliest confirmed defect, behind the 7c gate.

---

## 9. Rollback & data-safety model

**Git protects code only.** Git tags/commits cover tracked code, notebooks, and docs. They do **not** protect gitignored parquet files, CSV artifacts, split files, or model-data artifacts. Every write phase touching data therefore uses a separate artifact rollback mechanism.

**Artifact baseline manifest.** Before any move, rename-with-copy, rebuild, or overwrite-prone operation on an artifact, record:
- original path
- file size
- content hash
- schema (columns + dtypes)
- row count
- key statistics (null rates, unique-key counts, target distribution where relevant)

**Isolation requirement (strengthened).** Preserved copies must be **isolated from the working artifact path** — never stored beside the file they protect, where the same faulty operation could destroy both. For **high-value artifacts such as base splits**, prefer a protected backup location separate from the active overwrite target when practical (a dedicated backup folder that no pipeline notebook or remediation script writes to). The manifest records both the original path and the preserved-copy location.

The isolated preserved copy plus its manifest entry form the rollback point for that artifact. **The preserved baseline is never overwritten.** The manifest lives with the audit documentation (not inside the data tree's working folders), and Phase 9 snapshots the final manifest as the frozen data-state record.

**Rollback procedure.** If a group fails validation: restore from the isolated preserved copies for that group, verify restoration against the manifest (hash + row count), report, and halt until re-approved.

---

## 10. Risks (most important only)

1. **Silent split overwrite.** If `course_difficulty.ipynb` and/or `diploma_type_bucketing.ipynb` rewrite base splits in place, the current `df_train/valid/test` are enriched, not base — a naive move/rebuild could destroy the only base copy or leak enriched columns. Phase 1 must settle base-vs-enriched before anything touches `model_data/`.
2. **Multi-writer `after_fet_eng.parquet`.** Declared owner vs. actual writers (`handle_gpa.ipynb`, extended by `merge_diploma.py`) — choosing the wrong canonical writer corrupts the feature stage feeding the splits. This same unresolved conflict is why the file is only a *candidate* reconstruction source in Phase 1.
3. **Dotted-ID dtype corruption.** A float/`to_numeric` cast on a dotted identity ID silently breaks joins and is hard to detect afterwards. Phase 2 must catch it before any rebuild re-runs a bad cast.
4. **Mistaking `audit/` inputs for reports.** `audit/df_crg_add_acd.parquet` and `audit/after_fet_eng.parquet` are consumed downstream; relocating them as disposable reports breaks the pipeline.
5. **Duplicates/loose files masking authority.** `merged.csv`, `after_feature_eng_run.csv`, and the double ADD-clean write could each be the real artifact or dead weight — archiving the wrong one loses real data. Tier-verdict duplicate evidence (Phase 3) precedes any archive action; a hash mismatch alone never proves the files are different.
6. **Existence-as-authority bias.** Neither a populated file nor an empty folder proves correctness; every decision rests on Phase 1–3 evidence. Delta claims carry OBSERVED/INFERRED/UNRESOLVED labels so inference is never mistaken for observation.
7. **Env-var fragility + import side effect.** With the env var unset, `src.paths` silently falls back to the empty repo shell, and its import-time `ensure_dir` masks the error by creating empty folders. Mitigations: each command context verifies or sets the env var; no `src.paths` import during read-only phases; startup guard assertion proposed in Phase 5 and implemented (with approval) in 7a.
8. **Twin-copy drift.** Any accidental run without the env var would populate the repo shell from a partial pipeline, creating a divergent second copy. Only the active root may ever be written; the repo shell stays empty for the duration of this job.
9. **Git-doesn't-protect-data blind spot.** Relying on a git tag as the only rollback point leaves every data artifact unprotected. The baseline manifest + isolated preserved copies (Section 9) are mandatory for every write group.
10. **mtime misreading.** All current split mtimes are identical (2026-07-02 16:51); inferring write order or authority from them would be false confidence. mtimes stay diagnostic-only.
11. **Co-located backups.** A preserved copy stored next to its working file can be destroyed by the same faulty script it exists to protect against. The Section 9 isolation requirement addresses this, especially for base splits.

---

## 11. Locked objectives → phase coverage

| # | Locked objective | Covered by |
|---|------------------|-----------|
| 1 | Split integrity & overwrite-chain audit | Phase 1 |
| 2 | ID & dtype consistency audit | Phase 2 |
| 3 | Full pipeline lineage | Phase 3 |
| 4 | One-owner-per-artifact governance | Phases 3, 5 (clarified contract), 7a |
| 5 | Intra-tree file placement cleanup | Phases 4, 7b |
| 6 | Duplicate & loose-file review | Phases 3, 4 (binary + logical tier verdicts) |
| 7 | Notebook naming & numbering | Phase 6A, 7b |
| 8 | DataFrame naming | Phase 6B, 7b |
| 9 | Saved-artifact naming & traceability | Phases 5, 6C, 7b |
| 10 | Governance contracts | Phase 5 |
| 11 | Controlled remediation | Phase 7 (per-group gates) |
| 12 | Selective rebuild only on evidence | Phase 7c (own gate, earliest-defect rule) |
| 13 | Byte/logical parity vs expected delta | Section 4, applied in 3, 7 & 8 |
| 14 | Final pipeline freeze | Phase 9 |

---

## 12. Verification & approval

This plan is validated by review against: CLAUDE.md guardrails (paths job done; logic untouched; archive-before-delete; deletion needs approval), `docs/pipeline_rules.md` locked decisions (split boundaries, `feature_contract.json` allowlist, train-only leakage control, ownership rules, `finish_status` target contract), and the completed `docs/paths_audit.md` lineage clues.

Nothing beyond the already-executed and recorded visibility check has run. Phase 1's remaining work (split writer/reader/overwrite mapping, base-vs-enriched, reconstruction-capability assessment) is read-only and **awaits explicit approval** before starting.

**STOP — awaiting approval.**