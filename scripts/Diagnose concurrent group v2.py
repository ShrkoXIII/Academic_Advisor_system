"""Read-only diagnostics for the Concurrent Course Group feature (v2).

Writes NOTHING, trains NOTHING. Run from the project root:

    python scripts/diagnose_concurrent_group_v2.py

Sections
--------
[0] Column contract gate on df_train_difficulty (GO / STOP)
[1] Semester group-size distribution + singleton share
[2] Duplicate course_id inside the same semester group
[3] Observed rows-per-group vs semester_reg_courses
    -> ALSO answers the withdrawal question: does the recorded registration
       count freeze at registration time (reg > observed where withdrawals
       happened) or track/derive from surviving rows (reg == observed)?
[4] Peer difficulty availability + weak-fallback share among peers
[5] Withdrawal / finish-status exclusion audit
    -> validates the decision: M1 target = (final_mark >= 50) on a
       population with ALL withdrawal rows removed upstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.paths import (
    FEATURE_ENGINEERED_PRIMARY_PATH,
    SELECTED_MODEL_POPULATION_PATH,
    assert_data_root,
    model_split_path,
)

# Semester grain — identical to feature_engineering.SEMESTER_KEY.
GROUP_KEY = ["university_id", "student_id", "degree_id", "part_id"]
COURSE_COL = "course_id"
PEER_DIFF_COL = "course_pass_rate_historical"     # aggregation input
PEER_MISSING_COL = "course_difficulty_missing"    # weak-fallback flag
REG_COUNT_COL = "semester_reg_courses"            # recorded registration count

HARD_REQUIRED = GROUP_KEY + [COURSE_COL, PEER_DIFF_COL]
SOFT_WANTED = [PEER_MISSING_COL, REG_COUNT_COL]


def _rule(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


def _schema_names(path) -> list[str] | None:
    try:
        return list(pq.ParquetFile(path).schema.names)
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"  (could not read schema of {path}: {exc})")
        return None


def finish_status_audit() -> None:
    """Section [5]: hunt for finish/withdrawal columns across key artifacts."""
    _rule("[5] WITHDRAWAL / FINISH-STATUS EXCLUSION AUDIT")
    print(
        "Decision under test: M1 target = (final_mark >= 50); ALL withdrawal\n"
        "rows must already be removed from the modeling population.\n"
    )
    targets = {
        "train difficulty split": model_split_path("train", "difficulty"),
        "valid difficulty split": model_split_path("valid", "difficulty"),
        "test  difficulty split": model_split_path("test", "difficulty"),
        "selected_model_population": SELECTED_MODEL_POPULATION_PATH,
        "feature_engineered_primary": FEATURE_ENGINEERED_PRIMARY_PATH,
    }
    any_hit = False
    for name, path in targets.items():
        if not path.exists():
            print(f"- {name}: MISSING file ({path}) — skipped")
            continue
        names = _schema_names(path)
        if names is None:
            continue
        hits = [c for c in names if "finish" in c.lower()]
        if not hits:
            print(f"- {name}: no finish-like column in schema")
            continue
        any_hit = True
        sub = pd.read_parquet(path, columns=hits)
        for col in hits:
            vc = sub[col].value_counts(dropna=False)
            print(f"- {name} :: {col} — {len(vc)} distinct value(s):")
            shown = vc if len(vc) <= 20 else vc.head(20)
            for val, cnt in shown.items():
                print(f"    {val!r:>24} : {cnt:,}")
            if len(vc) > 20:
                print(f"    ... {len(vc) - 20} more values truncated")
    if not any_hit:
        print(
            "\n>>> No finish-like column found in ANY audited artifact.\n"
            ">>> The status column was dropped upstream. The exclusion of\n"
            ">>> withdrawal rows must then be verified at the SOURCE: locate\n"
            ">>> the filter in the population-selection / cleaning stage and\n"
            ">>> confirm with row counts (before vs after)."
        )
    else:
        print(
            "\n>>> REVIEW REQUIRED: confirm that NONE of the values above is a\n"
            ">>> withdrawal status. If any withdrawal value remains in the\n"
            ">>> train/valid/test splits, the mark-based target is\n"
            ">>> contaminated and the exclusion filter must be fixed first."
        )

    # Guard: no finish-derived column may sit in the model feature contract.
    try:
        from src.model_training import MODEL_FEATURES, TRAINING_DATA_COLUMNS

        bad = sorted(
            c
            for c in set(MODEL_FEATURES) | set(TRAINING_DATA_COLUMNS)
            if "finish" in c.lower()
        )
        if bad:
            print(f"\n>>> STOP: finish-derived column(s) inside the feature contract: {bad}")
        else:
            print("\nfeature-contract guard: no finish-derived column in MODEL_FEATURES / TRAINING_DATA_COLUMNS — OK")
    except Exception as exc:  # noqa: BLE001
        print(f"\nfeature-contract guard skipped (import failed: {exc})")


def main() -> None:
    train_path = model_split_path("train", "difficulty")
    assert_data_root(train_path)

    _rule("CONCURRENT GROUP DIAGNOSTIC v2 (read-only)")
    print(f"input : {train_path}")

    # ---------------- [0] column contract gate ----------------
    _rule("[0] COLUMN CONTRACT (GO / STOP gate)")
    schema = _schema_names(train_path)
    if schema is None:
        finish_status_audit()
        return
    missing_hard = [c for c in HARD_REQUIRED if c not in schema]
    missing_soft = [c for c in SOFT_WANTED if c not in schema]
    print(f"hard-required present : {[c for c in HARD_REQUIRED if c in schema]}")
    print(f"hard-required missing : {missing_hard if missing_hard else 'none'}")
    print(f"soft-wanted   missing : {missing_soft if missing_soft else 'none'}")
    if missing_hard:
        print(
            "\n>>> STOP: grouping keys or the peer-difficulty source are absent\n"
            ">>> from the difficulty parquet. The concurrent feature cannot be\n"
            ">>> built. Plumbing task first: persist these columns through the\n"
            ">>> difficulty stage, then re-run this diagnostic."
        )
        finish_status_audit()
        return

    cols = HARD_REQUIRED + [c for c in SOFT_WANTED if c in schema]
    df = pd.read_parquet(train_path, columns=cols)
    n_rows = len(df)
    print(f"\nrows loaded : {n_rows:,}   columns : {cols}")

    for k in GROUP_KEY:
        df[k] = df[k].astype("string")

    grp = df.groupby(GROUP_KEY, dropna=False, sort=False)
    sizes = grp.size()
    n_groups = len(sizes)
    row_size = grp[COURSE_COL].transform("size")

    # ---------------- [1] group sizes ----------------
    _rule("[1] GROUP SIZE DISTRIBUTION (per semester group)")
    print(f"n_groups : {n_groups:,}")
    for size, count in sizes.value_counts().sort_index().items():
        print(f"  size {int(size):>2} : {count:>8,} groups  ({100.0 * count / n_groups:5.1f}%)")
    singleton_groups = int((sizes == 1).sum())
    singleton_rows = int((row_size == 1).sum())
    print(
        f"\nsingletons : {singleton_groups:,} groups "
        f"({100.0 * singleton_groups / n_groups:.1f}% of groups, "
        f"{100.0 * singleton_rows / n_rows:.1f}% of rows) "
        "-> these rows get concurrent_peer_difficulty_missing = 1"
    )

    # ---------------- [2] duplicates ----------------
    _rule("[2] DUPLICATE course_id WITHIN GROUP")
    dup_mask = df.duplicated(subset=GROUP_KEY + [COURSE_COL], keep=False)
    dup_rows = int(dup_mask.sum())
    dup_groups = df.loc[dup_mask, GROUP_KEY].drop_duplicates().shape[0]
    print(
        f"rows sharing a duplicated course inside their group : {dup_rows:,} "
        f"({100.0 * dup_rows / n_rows:.2f}% of rows) across {dup_groups:,} groups"
    )
    if dup_rows:
        print("  -> decide: same-semester retake vs data artifact; dedup before the peer mean?")

    # ---------------- [3] observed vs recorded registration ----------------
    _rule("[3] OBSERVED rows-per-group vs semester_reg_courses")
    if REG_COUNT_COL in df.columns:
        reg = pd.to_numeric(grp[REG_COUNT_COL].first(), errors="coerce")
        observed = sizes.astype(float)
        cmp = reg.notna()
        n_cmp = int(cmp.sum())
        eq = float((observed[cmp] == reg[cmp]).mean()) if n_cmp else float("nan")
        gt = float((reg[cmp] > observed[cmp]).mean()) if n_cmp else float("nan")
        lt = float((reg[cmp] < observed[cmp]).mean()) if n_cmp else float("nan")
        print(f"comparable groups            : {n_cmp:,}")
        print(f"reg == observed              : {100.0 * eq:5.1f}%")
        print(f"reg >  observed (rows gone)  : {100.0 * gt:5.1f}%")
        print(f"reg <  observed (unexpected) : {100.0 * lt:5.1f}%")
        gap = (reg[cmp] - observed[cmp])
        if len(gap):
            print(
                f"gap (reg - observed) : min {gap.min():.0f}, median {gap.median():.0f}, "
                f"p95 {gap.quantile(0.95):.0f}, max {gap.max():.0f}"
            )
        print(
            "\ninterpretation:\n"
            "  reg > observed in a real share of groups -> the recorded count is a\n"
            "  registration-time snapshot (withdrawn rows removed later): SAFE,\n"
            "  known at decision time.\n"
            "  reg == observed almost everywhere -> the recorded count tracks the\n"
            "  post-withdrawal state or is row-derived: mild leakage flag, and the\n"
            "  train/serve skew for the concurrent feature is negligible only if\n"
            "  the gap share is small."
        )
    else:
        print(f"{REG_COUNT_COL!r} absent — section skipped.")

    # ---------------- [4] peer difficulty availability ----------------
    _rule("[4] PEER DIFFICULTY AVAILABILITY (non-singleton groups)")
    multi = df.loc[row_size > 1].copy()
    if len(multi):  
        gm = multi.groupby(GROUP_KEY, dropna=False)
        multi["_size"] = gm[COURSE_COL].transform("size")
        multi["_valid"] = multi[PEER_DIFF_COL].notna()
        valid_in_group = gm["_valid"].transform("sum") if "_valid" in multi else None
        # transform on freshly assigned column requires re-grouping:
        valid_in_group = multi.groupby(GROUP_KEY, dropna=False)["_valid"].transform("sum")
        peers_valid = (valid_in_group - multi["_valid"].astype(int)) > 0
        print(f"rows in multi-course groups : {len(multi):,} ({100.0 * len(multi) / n_rows:.1f}% of rows)")
        print(f"  rows with >=1 valid-difficulty peer : {100.0 * peers_valid.mean():.1f}%")
        if PEER_MISSING_COL in multi.columns:
            multi["_weak"] = (
                pd.to_numeric(multi[PEER_MISSING_COL], errors="coerce").fillna(1).clip(0, 1)
            )
            weak_in_group = multi.groupby(GROUP_KEY, dropna=False)["_weak"].transform("sum")
            peer_weak_share = (weak_in_group - multi["_weak"]) / (multi["_size"] - 1)
            print(
                "  weak-fallback share among peers (per row):\n"
                f"    mean {100.0 * peer_weak_share.mean():.1f}%   "
                f"median {100.0 * peer_weak_share.median():.1f}%   "
                f"rows with NO weak peer {100.0 * (peer_weak_share == 0).mean():.1f}%   "
                f"rows where ALL peers weak {100.0 * (peer_weak_share == 1).mean():.1f}%"
            )
        print(
            "  -> drives the decision: include weak-fallback peers in the mean\n"
            "     (observed group as-is) vs down-weight/exclude them later."
        )
    else:
        print("no multi-course groups found (unexpected — inspect upstream).")

    # ---------------- [5] withdrawal audit ----------------
    finish_status_audit()

    _rule("DONE — paste this full output back before any feature code is written")


if __name__ == "__main__":
    main()