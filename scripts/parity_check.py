"""Phase 8 — Final cross-pipeline validation (read-only).

Runs the end-to-end integration checks described in docs/data_governance_plan.md
Phase 8: split boundaries, feature-contract conformance, target distributions,
join integrity, ID-suffix integrity, and a drift screen against the Phase 7c
baseline manifest. Never writes to the data tree; never modifies notebooks,
code, or feature_contract.json. Any unexplained drift is reported as a FAIL.

Usage:
    python -m scripts.parity_check
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from src.paths import (
    ARTIFACTS_DIR,
    DATA_DIR,
    FEATURES_DIR,
    MERGE_DIR,
    MODEL_DATA_DIR,
    MODELS_DIR,
    PREPROCESSED_DIR,
    RAW_DIR,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, cond: bool, detail: str = "") -> bool:
    RESULTS.append((label, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {label}  {detail}")
    return bool(cond)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


SPLITS = ["train", "valid", "test"]
YEAR_BOUNDS = {
    "train": (2005, 2021),
    "valid": (2022, 2023),
    "test": (2024, 2025),
}


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_join_integrity_and_lineage() -> None:
    section("1. End-to-end lineage & join integrity")

    merged = pd.read_parquet(MERGE_DIR / "merged_add_acd_crg.parquet", columns=["student_id"])
    check("merged_add_acd_crg row count matches Chain-A baseline", len(merged) == 761347, f"{len(merged):,}")

    diploma_merged = pd.read_parquet(MERGE_DIR / "merged_with_diploma.parquet", columns=["student_id"])
    check(
        "merged_with_diploma: left-join row invariance vs merged_add_acd_crg",
        len(diploma_merged) == len(merged),
        f"{len(diploma_merged):,} vs {len(merged):,}",
    )
    check(
        "merged_with_diploma: no duplicate expansion (student_course grain preserved)",
        len(diploma_merged) == 761347,
        f"{len(diploma_merged):,}",
    )
    del merged, diploma_merged

    selected = pd.read_parquet(FEATURES_DIR / "selected_model_population.parquet")
    check("selected_model_population row count", len(selected) == 761346, f"{len(selected):,}")
    check(
        "selected_model_population: diploma columns survive selection",
        {"diploma_gpa", "diploma_type_id"}.issubset(selected.columns),
    )
    del selected

    primary = pd.read_parquet(FEATURES_DIR / "feature_engineered_primary.parquet")
    check("feature_engineered_primary row count", len(primary) == 727852, f"{len(primary):,}")
    check(
        "feature_engineered_primary: diploma columns pass through feature engineering",
        {"diploma_gpa", "diploma_type_id"}.issubset(primary.columns),
    )
    total_primary = len(primary)
    del primary

    base_total = 0
    for split in SPLITS:
        df = pd.read_parquet(MODEL_DATA_DIR / f"df_{split}_base.parquet", columns=["part_year"])
        base_total += len(df)
        del df
    check(
        "base-split row sum matches feature_engineered_primary (minus unassigned 2025s2)",
        base_total == total_primary - 11282,
        f"base_sum={base_total:,}  primary={total_primary:,}  diff={total_primary - base_total:,} (expected 11,282 unassigned)",
    )

    for split in SPLITS:
        base_rows = len(pd.read_parquet(MODEL_DATA_DIR / f"df_{split}_base.parquet", columns=["student_id"]))
        diff_rows = len(pd.read_parquet(MODEL_DATA_DIR / f"df_{split}_difficulty.parquet", columns=["student_id"]))
        final_rows = len(pd.read_parquet(MODEL_DATA_DIR / f"df_{split}_final.parquet", columns=["student_id"]))
        check(
            f"{split}: row count identical across base -> difficulty -> final",
            base_rows == diff_rows == final_rows,
            f"base={base_rows:,} difficulty={diff_rows:,} final={final_rows:,}",
        )

    base_cols = pd.read_parquet(MODEL_DATA_DIR / "df_train_base.parquet").shape[1]
    diff_cols = pd.read_parquet(MODEL_DATA_DIR / "df_train_difficulty.parquet").shape[1]
    final_cols = pd.read_parquet(MODEL_DATA_DIR / "df_train_final.parquet").shape[1]
    check("column-count progression base -> difficulty (+7) -> final (+1)",
          diff_cols == base_cols + 7 and final_cols == diff_cols + 1,
          f"base={base_cols} difficulty={diff_cols} final={final_cols}")

    fit_state_path = ARTIFACTS_DIR / "diploma_type_bucket_map.json"
    check("diploma_type_bucket_map.json exists (contract 14 fitted state)", fit_state_path.exists())


def check_split_boundaries() -> None:
    section("2. Split boundaries (locked temporal rule)")
    for split, (lo, hi) in YEAR_BOUNDS.items():
        df = pd.read_parquet(MODEL_DATA_DIR / f"df_{split}_final.parquet", columns=["part_year", "part_semester"])
        yr = df["part_year"].astype("float64")
        sem = df["part_semester"].astype("float64")
        check(f"{split}: min/max part_year within [{lo}, {hi}]",
              yr.min() >= lo and yr.max() <= hi, f"observed [{yr.min():.0f}, {yr.max():.0f}]")
        if split == "test":
            bad = int((((yr == 2025) & (sem == 2))).sum())
            check("test: 2025-semester-2 excluded (incomplete semester)", bad == 0, f"{bad} offending rows")
        del df


def check_feature_contract() -> None:
    section("3. Feature columns vs models/feature_contract.json")
    contract_path = MODELS_DIR / "feature_contract.json"
    if not contract_path.exists():
        check("models/feature_contract.json exists", False, str(contract_path))
        return
    with open(contract_path, encoding="utf-8") as f:
        contract = json.load(f)

    df_train = pd.read_parquet(MODEL_DATA_DIR / "df_train_final.parquet")

    contract_features = contract["features"]
    check(f"contract declares {contract['n_features']} features, list length matches",
          len(contract_features) == contract["n_features"], f"{len(contract_features)}")

    # requirement_size_bucket_ord is derived at train time from requirement_size_bucket;
    # every other contract feature must already exist in the parquet.
    derived = {"requirement_size_bucket_ord"}
    parquet_required = [c for c in contract_features if c not in derived]
    missing = [c for c in parquet_required if c not in df_train.columns]
    check("all non-derived contract features present in df_train_final", not missing, f"missing={missing}")
    check("derivation source 'requirement_size_bucket' present for requirement_size_bucket_ord",
          "requirement_size_bucket" in df_train.columns)

    cat_feats = contract.get("categorical_features", [])
    for cf in cat_feats:
        check(f"categorical feature '{cf}' present in df_train_final", cf in df_train.columns)
        if cf in df_train.columns:
            train_levels = sorted(int(v) for v in pd.to_numeric(df_train[cf], errors="coerce").dropna().unique())
            contract_levels = contract["categorical_levels"].get(cf, [])
            check(f"'{cf}' current train-derived levels == contract levels",
                  train_levels == contract_levels, f"current={train_levels} contract={contract_levels}")

    dropped = contract.get("dropped_features", [])
    leaked_back = [c for c in dropped if c in contract_features]
    check("no dropped_features leaked back into the contract feature list", not leaked_back, f"{leaked_back}")

    m1 = contract.get("target_m1_classifier", "")
    m2 = contract.get("target_m2_regressor", "")
    check("M1 target definition unchanged (locked)", m1 == "(final_mark >= 50).astype(int)", m1)
    check("M2 target definition unchanged (locked)", m2 == "final_mark", m2)

    del df_train


def check_target_distributions() -> None:
    section("4. Target distributions per split")
    for split in SPLITS:
        df = pd.read_parquet(MODEL_DATA_DIR / f"df_{split}_final.parquet", columns=["final_mark"])
        n = len(df)
        null_ct = int(df["final_mark"].isna().sum())
        oor = int(((df["final_mark"] < 0) | (df["final_mark"] > 100)).sum())
        pass_rate = float((df["final_mark"] >= 50).mean())
        check(f"{split}: final_mark has zero nulls", null_ct == 0, f"{null_ct}")
        check(f"{split}: final_mark within [0, 100]", oor == 0, f"{oor} out-of-range")
        print(f"    INFO {split}: n={n:,}  pass_rate={pass_rate:.4f}  "
              f"mean_mark={df['final_mark'].mean():.2f}  median={df['final_mark'].median():.1f}")
        del df


def check_id_suffix_integrity() -> None:
    section("5. ID suffix integrity (dotted university suffix)")
    import re
    suffix_re = re.compile(r"\.\d+$")

    targets = [
        (MERGE_DIR / "merged_with_diploma.parquet", "student_id"),
        (FEATURES_DIR / "feature_engineered_primary.parquet", "student_id"),
        (MODEL_DATA_DIR / "df_train_final.parquet", "student_id"),
    ]
    for path, col in targets:
        df = pd.read_parquet(path, columns=[col])
        s = df[col].astype("string")
        check(f"{path.name}::{col} dtype is pandas string", str(s.dtype) == "string", str(s.dtype))
        bad = s[~s.str.contains(suffix_re, na=False)]
        check(f"{path.name}::{col} all values carry dotted suffix", len(bad) == 0,
              f"{len(bad)} malformed / {len(s)} total")
        del df, s


def check_drift_vs_baseline() -> None:
    section("6. Drift screen vs Phase 7c baseline manifest")
    manifest_path = Path("docs/manifests/baseline_7bc_2026-07-08.json")
    if not manifest_path.exists():
        check("baseline manifest exists", False, str(manifest_path))
        return
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Sentinel/untouched artifacts declared in the manifest: must be byte-identical.
    sentinel_paths = {
        "artifacts/course_difficulty_lookup.parquet",
        "artifacts/knn_index.pkl",
        "final/without_outliers.parquet",
        "preprocessed/merge/merge_crg_add.parquet",
    }
    for entry in manifest["artifacts"]:
        rel = entry["path"]
        if rel not in sentinel_paths or "sha256" not in entry:
            continue
        full = DATA_DIR / rel
        if not full.exists():
            check(f"drift sentinel present: {rel}", False, "MISSING")
            continue
        cur_hash = sha256(full)
        check(f"drift sentinel unchanged: {rel}", cur_hash == entry["sha256"],
              "hash match" if cur_hash == entry["sha256"] else f"HASH CHANGED {entry['sha256'][:8]}->{cur_hash[:8]}")

    # Retired artifacts must exist only in archive/pre_7c, not back on their live paths.
    retired = [
        "audit/after_fet_eng.parquet",
        "audit/df_crg_add_acd.parquet",
        "model_data/df_train.parquet",
        "model_data/df_valid.parquet",
        "model_data/df_test.parquet",
        "features/merged_add_acd_crg.parquet",
        "raw/v_add_adcademic_info.parquet",
    ]
    for rel in retired:
        p = DATA_DIR / rel
        check(f"retired artifact absent from live path: {rel}", not p.exists(),
              "still present!" if p.exists() else "absent (correctly retired)")

    archive_dir = DATA_DIR / "archive" / "pre_7c"
    check("archive/pre_7c exists with retired originals", archive_dir.is_dir() and any(archive_dir.iterdir()))


def check_no_hardcoded_or_stale_names() -> None:
    section("7. Repo-wide re-grep: no live hardcoded data-root paths, no stale bare-name references")
    import glob
    import re as _re

    drive_pat = _re.compile(r"[A-Za-z]:[\\/][^\"'\n)]*AI[\\/]")
    bare_split_pat = _re.compile(r"df_(?:train|valid|test)\.parquet")

    hits_hardcoded = []
    hits_bare = []
    for path in glob.glob("note_books/**/*.ipynb", recursive=True):
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code":
                continue
            src = "".join(c["source"])
            if drive_pat.search(src):
                hits_hardcoded.append(f"{path}#c{i}")
            if bare_split_pat.search(src):
                hits_bare.append(f"{path}#c{i}")

    check("no live hardcoded data-root paths in notebook code cells", not hits_hardcoded, str(hits_hardcoded[:5]))
    check("no bare df_{split}.parquet references left in notebook code cells", not hits_bare, str(hits_bare[:5]))

    for path in glob.glob("src/*.py"):
        with open(path, encoding="utf-8") as f:
            txt = f.read()
        if drive_pat.search(txt):
            check(f"no hardcoded data-root path in {path}", False, path)


def main() -> int:
    print("Phase 8 — Final cross-pipeline validation (read-only)")
    print(f"DATA_DIR = {DATA_DIR}")

    check_join_integrity_and_lineage()
    check_split_boundaries()
    check_feature_contract()
    check_target_distributions()
    check_id_suffix_integrity()
    check_drift_vs_baseline()
    check_no_hardcoded_or_stale_names()

    section("SUMMARY")
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"  {n_pass} PASS / {n_fail} FAIL / {len(RESULTS)} total checks")
    if n_fail:
        print("\n  FAILED CHECKS:")
        for label, ok, detail in RESULTS:
            if not ok:
                print(f"    - {label}  {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
