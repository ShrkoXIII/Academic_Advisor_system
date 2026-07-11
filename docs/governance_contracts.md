# Governance Contracts — Phase 5 Deliverable (APPROVED)

Status: **APPROVED by user 2026-07-07.** Produced 2026-07-07 under `docs/data_governance_plan.md` (Revision 4, corrected; repository state `c6069bc`). Phase 5 is planning only: this document changes no code, no notebooks, no data, no splits, and no `feature_contract.json`. A contract describes the **target**; §"Current state" notes say where today's code does not yet conform. Nothing here claims those gaps are closed — closure is Phase 7 work, each group behind its own gate.

These contracts must not weaken locked decisions in `docs/pipeline_rules.md` (temporal split boundaries, `feature_contract.json` allowlist, train-only leakage control, no `scale_pos_weight`/SMOTE, `finish_status` target). Where a conflict exists it is surfaced, not encoded (see D6 and the `pipeline_rules.md` ownership note below).

---

## Contract 1 — ID/dtype classes and canonical normalizers

**Rule.** Every identifier belongs to a declared ID class with a fixed storage dtype and exactly one canonical normalizer. Dotted university suffixes (e.g. `15.111`) are identity, not decimals.

| ID class | Storage dtype | Canonical normalizer | Notes |
|----------|--------------|----------------------|-------|
| `student_id`, `degree_id`, `course_id`, `part_id`, `faculty_id`, `grade_id`, `student_course_id` (dotted join keys) | pandas `string` | `cleaning_utils.normalize_id_series` / `normalize_id_columns` (element rule: `normalize_id_to_string`) | Normalize at **source cleaning time**, never patched at merge time. |
| `university_id` (derived) | `string` | suffix extraction `r"\.([^.]+)$"` (as in `feature_engineering.ensure_university_id`) | Derived for validation/keying only. |
| `diploma_type_id` | raw: float64 with dotted suffix → cleaned: `Int64` (suffix-stripped) | suffix-consistency check + strip + `Int64` cast at **source cleaning time** via `id_casting.normalize_ids` (schema: `src/schemas.py`); the equivalent check in `02_merge_diploma` remains as a secondary no-op guard (amended by user decision 2026-07-09; previously done at the diploma-merge stage) | Multi-suffix source = STOP (multiple universities). |

**Bans.** No join on float IDs. No `to_numeric`/`astype(float)` on a dotted ID class. No new normalizer implementations — consolidate on `cleaning_utils` (the Phase 2 finding of three parallel implementations is debt, not license). No float literals encoding dotted IDs (the dead `fillna(6.111)` in `Academic_info_clean.ipynb` is the canonical anti-example).

**Current state.** Pipeline join paths conform (verified `c6069bc`). `feature_engineering._normalize_key_series` remains a second live normalizer — consolidation is post-freeze debt; do not swap implementations during this job.

## Contract 2 — One writer per artifact path

**Rule.** Every artifact path has exactly one designated writer-owner. A consumer may read upstream artifacts and write its **own distinct downstream** artifact; it must never rewrite an upstream path unless it is the designated owner. Ownership table (target):

| Artifact path | Sole writer |
|---------------|-------------|
| `MERGE_DIR/merge_crg_add.parquet`, `MERGE_DIR/merged_add_acd_crg.parquet` | `01_merge_crg_add_acd` |
| `MERGE_DIR/merged_with_diploma.parquet` | `02_merge_diploma` |
| selected-population artifact (today `AUDIT_DIR/df_crg_add_acd.parquet`; relocates in 7b) | `select` |
| `after_fet_eng.parquet` | feature-engineering stage (`handle_gpa` via `run_feature_engineering_job`) — **C1 enforcement pending**: `merge_diploma.py`'s executable in-place write must be neutralized (7a) and a repo-wide grep must confirm single-writer before C1 is closed |
| base splits `df_{train,valid,test}` | `split_diagnostics` only |
| difficulty-enriched splits (distinct files, D2) | `course_difficulty` only |
| final-model splits (distinct files, D2) | `diploma_type_bucketing` only |
| fitted-state artifacts (contract 14) | the stage that fits them |

**Conflict surfaced, not encoded:** `docs/pipeline_rules.md` ("src/ module ownership") still describes `merge_diploma.py` as a sanctioned extender of `after_fet_eng.parquet`. That wording pre-dates D1 and now contradicts this contract. `pipeline_rules.md` is a paused/locked document — it is flagged for reconciliation at the 7a gate, not silently edited.

## Contract 3 — Split immutability and distinct split generations

**Rule.** Base splits are written once by their sole owner (`split_diagnostics`) and are immutable thereafter. Enrichment stages read a split generation and write a **new distinct** generation; no stage ever writes to another stage's path, and no stage rewrites its own published generation in place. Target generations (filenames = Phase 6): base → difficulty-enriched → final. Split boundary masks are computed in exactly one place (`split_diagnostics`) — downstream stages may sanity-check boundaries but never recompute or redefine them.

**Current state (non-conforming, known).** C2–C4 active: **both** `course_difficulty.ipynb` and `diploma_type_bucketing.ipynb` still save back onto `MODEL_DATA_DIR/df_{train,valid,test}.parquet`. This contract is the D2 target; the code change is 7a/7c work. C2–C4 remain open until distinct generations exist in code.

## Contract 4 — Overwrite policy

**Rule.** In-place rewrite of an existing artifact is permitted only when ALL hold: (a) the writer is the path's designated owner; (b) the rewrite is idempotent (rerun ⇒ logically identical output per §4B of the plan); (c) the behavior is documented at the write site. Otherwise forbidden. Writing to a path that already exists outside these conditions is a collision = STOP. Remediation writes additionally require the preserve-before-change rule (contract 10).

## Contract 5 — Stage boundaries

**Rule.** The pipeline is the ordered stage chain:
`raw → clean → merge(CRG+ADD) → merge(CRG+ADD+ACD) → merge(+Diploma) → select → feature-eng → split(base) → difficulty(derived) → bucketing(final) → training → KNN → inference/recommendation`.
Each stage reads only upstream artifacts and writes only its own outputs. No stage skips a boundary (e.g. no merge-stage logic inside `select`, no split logic inside enrichment). Cross-stage "convenience" writes are ownership violations under contract 2.

## Contract 6 — Folder ownership

**Rule.**

```text
RAW_DIR          raw extracts only
PREPROCESSED_DIR cleaned source-table artifacts only
MERGE_DIR        merge outputs only
FEATURES_DIR     selected + feature-engineered modeling frames
AUDIT_DIR        diagnostics, mismatch reports, validation evidence only
MODEL_DATA_DIR   split artifacts by immutable derivation stage
ARTIFACTS_DIR    runtime/model-support artifacts with reproducible producers
ARCHIVE_DIR      approved superseded/obsolete DATA artifacts (not tracked code)
```

No loose files at the data root. **Known tensions (7b targets, recorded not fixed):** `after_fet_eng.parquet` and the selected-population artifact live in `AUDIT_DIR` (belong in `FEATURES_DIR`); `01_merge_crg_add_acd` writes its unmatched-audit CSV into `MERGE_DIR` (belongs in `AUDIT_DIR`); `handle_gpa` still writes debug `merged.csv` to the data root (residue whose producer is live code).

## Contract 7 — Artifact naming principles

**Rule.** An artifact name encodes stage + content, and its generation where applicable: `<stage>_<content>[_<generation>].parquet`. Names never lie about lineage (a file named for a stage is written by that stage's owner only). No two artifacts differ only by letter case. Concrete renames are Phase 6 deliverables; this contract fixes the principles Phase 6 must follow.

## Contract 8 — Notebook numbering principles

**Rule.** Numbered notebooks reflect real execution order within their stage folder (`01_`, `02_`, …). A number implies a dependency on all lower numbers in the same chain. Unnumbered notebooks are diagnostics/exploration and must never be required for the pipeline to build. Renames are Phase 6; the residual-typo notebook (`02_final_merged_with_dimploma.ipynb`) keeps its current name until then.

## Contract 9 — DataFrame naming principles

**Rule.** In-notebook DataFrame names are semantic: source/domain + stage (`df_crg_add`, `dfcrg_acd_add` → normalized in Phase 6 to a consistent pattern like `df_<domain>_<stage>`). No positional names (`df1`, `df2`, `a`, `d1`) in pipeline notebooks — permitted only in throwaway diagnostic cells. Phase 6 proposes the concrete renames; no renaming during Phase 5.

## Contract 10 — Remediation safety

**Rule.** Preserve before change (isolated copy + manifest entry before any move/rename/rebuild/overwrite-prone op). Update consumers only after ownership is confirmed. No silent overwrite — collision is STOP. One remediation group at a time; validate (per the plan §4 type declared in Phase 4) before promote; user approval between groups. Rebuild is last resort, from the earliest confirmed defect, behind the 7c gate.

## Contract 11 — Rollback / baseline manifests

**Rule.** Git protects tracked code/notebooks/docs; gitignored DATA artifacts are protected by baseline manifests + isolated copies. Manifest format per artifact: original path, size bytes, content hash, schema (columns+dtypes), row count, key statistics (unique key counts, null counts on key columns), capture timestamp, and the git commit of the producing code. Capture timing: immediately before the first Phase 7 write that could affect the artifact. Isolation: preserved copies never live beside the file they protect; base splits get a dedicated backup folder no pipeline or remediation script writes to. The manifest lives with audit docs, is never overwritten (append new versions), and Phase 9 snapshots it.

## Contract 12 — Data-root guard

**Rule.** Pipeline entry points assert, before any read/write: the resolved `DATA_DIR` is the intended root (env var set and pointing at the active root), and the expected split artifacts exist with nonzero size. The `ensure_dir` import-time side effect in `src/paths.py` must not be able to mask an unset env var by silently materializing an empty shell — the guard must distinguish "root exists and is populated" from "root was just created". This contract proposes the code change; implementation is a gated 7a item. Until then: set/verify the env var inline in every command context; never trust folder existence as evidence of the correct root.

## Contract 13 — Pre-admission feature availability

**Rule.** `diploma_type_id` and `diploma_gpa` are pre-admission facts — known before any target semester. Joining them upstream (D1) does not leak future outcome information **as values**, and no temporal masking is required for them. This clause covers value availability only. It does not itself approve population-level fitted statistics. **D6 resolution (user decision, 2026-07-07):** the existing full-source median fill of `diploma_gpa` in `Academic_info_clean.ipynb` is an **explicitly approved, logged exception** for this cycle — it stays as-is and is recorded in the Decisions_Log. The exception is specific to that fill and does not generalize: every future fitted statistic falls under contracts 14–15 (train-fitted, persisted, frozen at inference).

## Contract 14 — Train-fitted transformation persistence

**Rule.** Any transformation whose parameters are fitted on training data and needed to transform new data at inference time MUST persist its fitted state as a versioned artifact under `ARTIFACTS_DIR`, written by the stage that fits it (contract 2). Refitting at inference is forbidden (contract 15). Session-memory-only fit state is a contract violation.

Minimum persisted state per known fitted transformation:

- **Diploma-type bucketing** (`diploma_type_bucketing.ipynb`): the top train codes (`TOP_DIPLOMA_CODES`); the rare-code policy and the full explicit code→bucket mapping; the unseen-code policy (unseen-in-train and null → unseen label); the reserved labels (`RARE_BUCKET_LABEL = 6`, `UNSEEN_BUCKET_LABEL = -1`) with their collision guarantee (no real code may equal a reserved label — the existing assertion becomes part of the persisted artifact's validity conditions); the final category set (sorted top codes + [6, −1]); the fit-source version (git commit + input artifact identity); and a reference to the train-split manifest (contract 11) it was fitted on. **Current state: none of this is persisted — it exists only in notebook session memory.**
- **Categorical levels** (`model_training.learn_categorical_levels`, currently `requirement_type_id`): allowed level set per column + `UNKNOWN_CATEGORY = -1` policy + fit-source version + train manifest reference. Currently persisted only inside the model metadata dump, if at all — verify at 7a.
- **Course-difficulty lookups** (`course_difficulty.ipynb`, 6-level fallback): all level tables + shrinkage parameters + global fallback scalar + fit-source version + train manifest reference. Current on-disk `course_difficulty_lookup.parquet` is an orphan with no committed producer — adopting a producer is the standing USER DECISION; under this contract a producer becomes mandatory if the feature stays.
- **KNN index** (`knn_advisor.KNNAdvisor`): index artifact + build parameters + train manifest reference (`knn_index.pkl` is currently an orphan — same decision).
- **Any future imputation fitted on train** (e.g. the D6 option (a) `diploma_gpa` median): the statistic value(s), fit population definition, fit-source version, train manifest reference.

## Contract 15 — Inference consistency / frozen preprocessing state

**Rule.** Inference (`inference.py`, `recommendation.py`, KNN advisory) must reproduce the training-time representation exactly by **loading** persisted fitted state (contract 14) — never by refitting, re-deriving, or approximating it from whatever data is at hand. The preprocessing applied to an inference-time row is the frozen train-time version: same normalizers (contract 1), same category sets and reserved labels, same imputation statistics, same difficulty lookups, resolved from versioned artifacts whose fit-source matches the deployed model's training manifest. A model artifact and its fitted-state artifacts form one atomic bundle: mixing versions is forbidden; a missing fitted-state artifact is a hard startup failure, not a silent refit.

---

## Out-of-scope for this document (explicit)

Per the Phase 5 definition: no notebook or script modifications; `merge_diploma.py` not neutralized; `diploma_type_bucketing.ipynb` not rewritten; no distinct splits created; no artifacts moved; no data rebuilt; `feature_contract.json` untouched; Phase 6 not begun.

## Open decisions these contracts do not resolve

- **D5** — outlier branch disposition (plan §10). Blocks 7c scoping, not these contracts.
- ~~**D6** — `diploma_gpa` full-source median fill~~ — **RESOLVED 2026-07-07** as a user-approved logged exception (see contract 13 and plan §10).
- Orphan producers (course-difficulty lookup, KNN index) — USER DECISION under contract 14.
- `pipeline_rules.md` ownership wording reconciliation — 7a gate.

**Gate:** Phase 5 approved by user 2026-07-07; Phase 6 (naming & numbering plan, `docs/naming_plan.md`) has begun. No Phase 7 work is unlocked by this document alone.
