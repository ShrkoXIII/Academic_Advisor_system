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

---

## 2026-07-27 — Define `concurrent_43`; re-base the position gate; five-seed validation against `concurrent_44`

**Scope.** Defined `concurrent_43` = `concurrent_44` minus
`concurrent_peer_difficulty_missing` (the feature the entry above calls
effectively dead), order of the remaining 43 features preserved exactly.
`baseline_41` and `concurrent_44` are untouched and remain selectable;
`DEFAULT_FEATURE_CONTRACT` stays `concurrent_44`. Trained `concurrent_43`
on the same immutable dataset version
(`data/model_data/versions/2026-07-26_batched_fixes__registration_roster_concurrent`)
at the same five seeds (42, 52, 62, 72, 82) already used for the
`baseline_41` vs `concurrent_44` experiment above; TEST stayed
`closed_not_read` in all five new runs (nonexistent `--test` path). The
five existing `concurrent_44` runs from that earlier experiment were reused
unchanged as the comparison arm — nothing was retrained except
`concurrent_43`. Full evidence:
[`models/runs/MULTISEED_CONCURRENT43_VS_CONCURRENT44_REPORT.md`](models/runs/MULTISEED_CONCURRENT43_VS_CONCURRENT44_REPORT.md),
verified pairwise against
[`models/runs/CONCURRENT43_VS_CONCURRENT44_VERIFICATION.json`](models/runs/CONCURRENT43_VS_CONCURRENT44_VERIFICATION.json).
Acceptance yardstick: [`models/runs/NOISE_BAND.md`](models/runs/NOISE_BAND.md)
(derived from the `baseline_41` vs `concurrent_44` paired deltas above).

**Position-gate re-base
(`scripts/build_concurrent_group_features.py:_assert_contract`).** The
dataset builder's `EXPECTED_LEGACY_MODEL_POSITION = 35` gate previously read
`MODEL_FEATURES` / `EXPECTED_FEATURE_COUNT`, deprecated globals in
`src/model_training.py` that are aliases for "whichever list a maintainer
last pointed them at" (in practice, always `CONCURRENT_44_FEATURES`, since
nothing had ever repointed them). Before: `legacy in MODEL_FEATURES and
MODEL_FEATURES.index(legacy) == 35`. After: the same check against the
explicit, named `CONCURRENT_44_FEATURES` list imported directly, so the gate
is pinned to `concurrent_44` specifically rather than to an ambiguous
global. Same gate values today (44 features, legacy indicator at index 35,
all seven gates pass); the dataset's column layout is unaffected — the
builder still writes all 8 concurrent columns (3 model + 5 audit) regardless
of which named model contract is selected for training. The re-base matters
going forward: now that `concurrent_43` is a real, selectable contract that
legitimately excludes this column, a gate still keyed off the deprecated
globals would risk breaking (or silently no-op) the moment anyone repointed
those globals — the re-based gate cannot break that way, because it no
longer reads them.

**Validation outcome — mixed by model, matching the M1/M2 split above.**

- **M2 (grade regressor): EQUIVALENT.** All three primary VALID deltas
  (MAE mean +0.0118, RMSE mean +0.0097, R2 mean -0.0010) fall inside the
  noise band, including the full per-seed range for RMSE and R2. `concurrent_43`
  and `concurrent_44` are indistinguishable for M2 on this evidence.
- **M1 (pass/fail classifier): INCONCLUSIVE.** VALID AUC, fail-class AP, and
  Brier all landed inside the noise band (mean deltas +0.00044, +0.00125,
  -0.00009 respectively — all improving or flat). The train-valid AUC gap
  improved on average (mean -0.0064) and in 3/5 seeds by more than the
  largest improvement recorded in the noise band (band min -0.005873); no
  seed worsened the gap beyond the band's harmful edge (+0.02672). This is
  an out-of-band result on the **beneficial** side, not evidence of harm —
  but per the band-is-the-bar rule, a metric outside the band in either
  direction blocks a clean EQUIVALENT verdict rather than being folded into
  one after the fact. Best-iteration also dropped materially for M1 (mean
  -42.4 rounds, -21% of the `concurrent_44` mean, in 3/5 seeds), directionally
  consistent with the smaller AUC gap; M2's best-iteration shift was noisy
  and not systematic (mixed direction, no band-relevant effect).

**No blanket "`concurrent_43` replaces `concurrent_44`" — split by model,
direction only, nothing wired.** M2's evidence supports treating
`concurrent_43` as the concurrent arm for M2 in future comparisons. M1's
evidence does not yet support that swap; `concurrent_44` remains the M1
concurrent arm pending more seeds or a repeated check specifically on the
train-valid AUC gap. This mirrors the M1=`baseline_41`/M2=candidate split
already recorded above, and is likewise **not implemented or wired** — no
promotion, no inference/recommendation change, no contract default change.
`baseline_41` and `concurrent_44` are untouched by this task; `concurrent_43`
is simply now available for the future regularization pass to select
explicitly.

**No production contract changed; nothing promoted.** `DEFAULT_FEATURE_CONTRACT`
is still `concurrent_44`. No dataset, live `MODEL_DATA_DIR`,
`CURRENT_VERSION.txt`, promotion marker, inference wiring, or recommendation
wiring was touched. TEST was not read by any of the five new runs.

**Next action (named, not started).** Regularization pass, two arms:
`baseline_41` vs `concurrent_43` — direction only; do not start it without
separate explicit approval.
