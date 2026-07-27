# Decisions Log

This file is a new, git-tracked decision log for the Academic Advisor repo.
Earlier project decisions (D1–D7, the 2026-07-07 governance freeze, etc.)
were recorded in `obsidian_vault/Decisions_Log.md`, which lives outside
this git checkout and is gitignored. That history is not duplicated here;
this file starts a fresh, in-repo record going forward.

---

## 2026-07-27 — Multi-seed stability experiment: baseline_41 vs concurrent_44

**Scope.** Five paired seeds (42, 52, 62, 72, 82), each seed training both
`baseline_41` (41 features) and `concurrent_44` (44 features, adds
`concurrent_peer_difficulty_mean/max/missing`) on the same VALID-only
selection split. TEST stayed `closed_not_read` in all ten runs. Full
evidence: [`models/runs/MULTISEED_BASELINE41_VS_CONCURRENT44_REPORT.md`](models/runs/MULTISEED_BASELINE41_VS_CONCURRENT44_REPORT.md)
and its noise-band summary, [`models/runs/NOISE_BAND.md`](models/runs/NOISE_BAND.md).

**M1 (pass/fail classifier): stays on `baseline_41`. Status: INCONCLUSIVE.**
Mean VALID AUC delta +0.00048 (candidate better), but the train-valid AUC
gap worsened in 4/5 seeds (mean +0.0079) and fail-class AP was mixed
(improved in only 2/5 seeds). The AUC improvement is within the observed
seed-to-seed noise band and does not clear it on the metrics that matter
operationally (fail-class AP, generalization gap).

**M2 (grade regressor): concurrent features SUPPORTED in principle.**
Mean VALID MAE delta -0.0141 (SD 0.0372), direction stable in 4/5 seeds;
effect is small but consistent. Not wired into production and not
promoted by this task — see "Two-contract direction" below.

**RETRACTED: the cold-start justification from the earlier single-seed
experiment.** Cold-start AUC improved in only 3/5 seeds under the
five-seed run; the SD of paired deltas (0.0083) is the same order of
magnitude as the original single-seed effect (0.0082). Do not cite
cold-start improvement as evidence for the concurrent features again
without a wider or repeated-seed check.

**`concurrent_peer_difficulty_missing`: effectively dead.** Used by any
model in only 2 of 5 seeds (M1 seed 52 once; M2 seed 82 once), under
0.001% of total gain in both cases, zero splits everywhere else. Treated
as dead weight. Removal is deferred to the future `concurrent_43`
contract task — not done here (out of scope for this close-out).

**Residual assumption — seed-42 pair reused, not rerun.** The seed-42
baseline/candidate pair predates the `--seed` CLI flag and was reused from
the earlier single-seed experiment rather than rerun under the current
seeded code path. Their `feature_contract.json`/`metrics.json` do not
contain the `effective_seed_settings` field the other four seed pairs
have. Verification for this task read the four resolved LightGBM sub-seeds
directly out of the serialized `.lgbm` model files instead (see
`_effective_seed_settings`, `src/model_training.py`); they match the
report's table (`data_random_seed=175, feature_fraction_seed=30056,
bagging_seed=400, drop_seed=17869`) and are identical between the M1/M2
models and between the baseline/candidate arms. This is a verified
artifact-level equality, not a claim that the seed-42 pair was produced by
literally the same code path as seeds 52–82 — log it as an assumption
about equivalence, not a verified process equality.

**Seed-52 provenance repair.** The seed-52 baseline run's
`feature_contract.json` git block (`git.commit`, `git.branch`,
`git.working_tree_clean`, `git.dirty_paths`) was edited after the fact, at
2026-07-27T10:31 (its file mtime), roughly 3–4 minutes after the run's
`metrics.json` and both `.lgbm` models were written (10:27–10:28,
unchanged since). The repair made the baseline's git block byte-identical
to the candidate's git block from the same seed pair (commit
`5928aaa1485bf6fc9d930eb8f49e3146498c846d`, branch `main`,
`working_tree_clean: false`, same two dirty paths) — this current,
post-repair state is verified correct.

The **pre-repair values are not recoverable**: `models/` (including
`models/runs/`) is gitignored, so there is no git history for the file; no
backup copy exists on disk; and the run's stdout/stderr logs do not
capture the git-metadata step. The stated cause is that "the first
launcher failed to inject Git safe-directory state." Reading
`_git_state()` (`src/model_training.py:920-942`): it shells out to
`git rev-parse HEAD` / `git status --porcelain` with `check=False` and
silently converts any non-zero exit (e.g. a `dubious ownership` failure
from a missing `safe.directory` config) into `commit=None, branch=None,
working_tree_clean=None, dirty_paths=[]`. That is a plausible
reconstruction of what the pre-repair block likely looked like, consistent
with the stated failure mode — but it is an inference from reading the
code, not a recovered fact, and is recorded here as such. No metric,
model, or data artifact was touched by the repair; only this one JSON
file's `git` block changed.

**Two-contract direction (M1=41, M2=43) — intended direction only, not
wired.** Given the M1/M2 split above, running M1 on `baseline_41` and M2
on a future 43-feature contract (`concurrent_44` minus the dead
`concurrent_peer_difficulty_missing`) is a plausible direction. This is
**not implemented or decided** — no `concurrent_43` contract exists, no
promotion has happened, and no inference/recommendation wiring was
touched. The decision is deferred until after a regularization pass that
must run both arms (M1 and M2) under both contracts before any promotion
is considered.

**Integrity confirmations (verified against on-disk artifacts, not just
the report's prose):** all ten runs' `test_policy` = `closed_not_read`;
every M1/M2 `test` metric field is `null`; no run's JSON contains a
readable TEST path string; `df_test_final.parquet` for the dataset version
used (`2026-07-26_batched_fixes__registration_roster_concurrent`) has an
mtime (2026-07-26 12:34) that predates the entire experiment window
(2026-07-26 15:51 through 2026-07-27 10:39); train/valid SHA-256 hashes
are identical across all ten runs; row counts are 450465 train / 156097
valid in all ten. No dataset, live model artifact, `CURRENT_VERSION.txt`,
promotion marker, recommendation wiring, or inference wiring was changed
by this experiment or by this close-out task. No training was run as part
of this close-out; all numbers above were recomputed from existing
artifacts (including an independent reload of two trained models to
recompute the `level_1_difficulty` segment AUC, which is not stored in
`metrics.json`).

**Next action.** Keep M1 on `baseline_41`. Treat the M2 evidence for
`concurrent_44` as supported-but-small; no promotion or wiring change
follows from this task. Await explicit human review before acting on the
two-contract direction.
