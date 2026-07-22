# Academic Advisor — ML Pipeline

## CURRENT OBJECTIVE (this is the only active job right now)

Reorganize the project so it is clean and consistent. Three tasks, in order:

1. **Paths.** Every notebook and script MUST get its file paths from
   `src/paths.py`. No hardcoded absolute paths anywhere — not in code, not
   in docstrings, not in usage examples. This is the first and main task.
2. **Names.** Rename files and folders to clear, meaningful names.
   No more `read.ipynb` / `Read.ipynb` / `all.ipynb`. Names must say what
   the file does.
3. **Remove duplicate merge notebooks.** There must be exactly ONE merge
   notebook per pipeline stage.Identify dead prototypes during the audit.
Do not delete immediately.
Move approved superseded files to archive/ first.
Permanent deletion requires explicit approval.
Do NOT change any data logic, ML logic, features, or model code during this
job. Paths and names only. Logic fixes are a separate later job.

## How to do the paths task safely

During the read-only audit phase, do not modify src/paths.py.

After explicit approval, update src/paths.py before converting notebooks.
Compare it against the canonical directory layout first; do not assume the
only missing constants are AUDIT_DIR, MODEL_DATA_DIR, ARTIFACTS_DIR, or
MODELS_DIR.
*Convert one file at a time. Change only where it looks for files —
  replace `D:\...` strings with imports from `paths.py`.
After converting a file, first verify resolved input/output paths without
changing data.

Do not execute notebooks or scripts that write pipeline artifacts unless
execution is explicitly approved.

When execution is approved, confirm that the file reads/writes the intended
canonical artifacts and that no data logic changed.. If any output changed, stop and report — a path task
  must never change data.
*Do the merge and model notebooks first: that is where the wrong-folder
  bug lives.

## Canonical paths

* Project root: `D:/AI/Real projects/Academic_Advisor/`
* Canonical data root: `PROJECT_ROOT/data`, exposed through `src/paths.py`.
* Anything under `D:/AI/data_clean_academic_advisor/` is LEGACY. Do not
  read from it or write to it. If a path pointing there is found in code or
  docstrings, that is a bug — report it, do not silently repoint it.

## Naming rules

* One merge notebook per stage. The early merge (CRG + ADD + ACD) and the
  later diploma merge are SEPARATE stages — keep them separate, do not
  combine them. Just remove dead duplicate prototypes.
* Notebook names describe the action: e.g. `clean_v_crg_student_course`,
  `01_merge_crg_add_acd`, `split_diagnostics`.
* No two files with the same name differing only by letter case.

## Environment

Windows. Python 3.11 in `.venv`. VS Code + Jupyter. All pipeline data is
parquet. Source data = Oracle views in `data/raw/`.

## DO NOT TOUCH (guardrails — always apply, even during the paths job)

* Do NOT change target definitions, split logic, saved split parquets, or
  `feature_contract.json`.
* Do NOT touch any data logic, feature engineering, or model code. This job
  is paths and names only.
* Make minimal changes. Do not refactor unrelated code.
* Inspect the real code before editing. Never infer behavior from a
  filename or notebook title.
* If a docstring and the code disagree, trust the code and report it.
* If unsure, stop and ask instead of guessing.
* Explain the plan in plain language before writing code.

## ID Convention (do not break this while editing)

IDs carry a dotted university suffix (e.g. `15.111`). The suffix is
meaningful identity, NOT a decimal. IDs are stored as pandas `string` via
`cleaning_utils.normalize_id_series`. Never join on float IDs.

## Data lifecycle (locked)

These rules govern how a dataset rebuild moves from candidate to accepted
"live" status. They apply always, independent of whichever job is
currently active.

* **L1. Live splits are the last accepted generation.** Live splits
  (`data/model_data/df_{train,valid,test}_final.parquet`) always equal the
  LAST ACCEPTED generation. Nothing writes to them except a promotion
  commit (L5).
* **L2. Every rebuild is a new immutable version.** Every dataset rebuild
  (feature engineering → splits → train-only stats → bucketing) writes to
  a new immutable folder `data/model_data/versions/<date>_<change-name>/`.
  Never modify an existing version folder; any fix is a new version.
* **L3. Experiments consume an explicit version.** Experiment training
  runs consume a version via explicit `--train`/`--valid`/`--test` paths.
  `model_split_path` defaults are reserved for the accepted live
  generation only.
* **L4. Every run records provenance.** Every run must record resolved
  input paths + file hashes (or mtimes) in `run_inputs.json` inside the
  run folder. Until tooling task T1 lands, the runner states the paths
  manually in the run REPORT.
* **L5. Promotion is one explicit commit, after acceptance is logged.**
  (a) copy the accepted version's files over the live splits; (b) write
  `data/model_data/CURRENT_VERSION.txt` with the source version; (c)
  re-base the parity reference (hashes + feature-count contract) to the
  new generation; (d) log the promotion in `Decisions_Log.md`. Rejected
  versions stay in `versions/`, marked rejected; live stays unchanged.
* **L6. Feature additions re-execute from stage 3.** Adding a feature
  re-executes stage 3 (feature engineering) onward into a new version.
  Stages 1–2 (raw, cleaning/merge) change only if the feature needs a new
  source column — then the rebuild starts from that stage.

---

*The full pipeline rules (targets, leakage control, merge validation, ML
metrics) are paused for this job and will return once path/name cleanup is
done. Keep them in git history / a separate doc — do not delete them.*
