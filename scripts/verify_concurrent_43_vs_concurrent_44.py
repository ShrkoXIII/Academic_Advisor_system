"""Programmatic pairwise verification: concurrent_43 vs concurrent_44 runs.

For each seed, confirms every non-feature-contract setting recorded in the two
runs' feature_contract.json / metrics.json is identical, and that the
feature-set difference between the two contracts is exactly the one dropped
column. This never trains a model, modifies a dataset, or touches TEST.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = PROJECT_ROOT / "models" / "runs"

PAIRS = {
    42: {
        "concurrent_44": "2026-07-26_1554__concurrent-44-registration-roster-candidate",
        "concurrent_43": "2026-07-27_1327__seed42-concurrent-43-drop-dead-missing-flag",
    },
    52: {
        "concurrent_44": "2026-07-27_1028__seed52-concurrent-44-registration-roster-candidate",
        "concurrent_43": "2026-07-27_1328__seed52-concurrent-43-drop-dead-missing-flag",
    },
    62: {
        "concurrent_44": "2026-07-27_1033__seed62-concurrent-44-registration-roster-candidate",
        "concurrent_43": "2026-07-27_1329__seed62-concurrent-43-drop-dead-missing-flag",
    },
    72: {
        "concurrent_44": "2026-07-27_1036__seed72-concurrent-44-registration-roster-candidate",
        "concurrent_43": "2026-07-27_1330__seed72-concurrent-43-drop-dead-missing-flag",
    },
    82: {
        "concurrent_44": "2026-07-27_1039__seed82-concurrent-44-registration-roster-candidate",
        "concurrent_43": "2026-07-27_1331__seed82-concurrent-43-drop-dead-missing-flag",
    },
}

MULTISEED_REPORT_PATH = (
    PROJECT_ROOT / "models/runs/MULTISEED_BASELINE41_VS_CONCURRENT44_REPORT.json"
)

# The seed-42 concurrent_44 run (2026-07-26_1554__...) predates the --seed CLI
# / build_run_contract additions: effective_seed_settings, training_control,
# diploma_gpa_handling and data_rows are simply not recorded in its
# feature_contract.json or metrics.json (both are literally null there). This
# is a pre-existing historical gap, already documented in the multiseed
# baseline_41-vs-concurrent_44 report's contract_verification section (that
# report's seed-42 entry omits exactly these same four checks). These four
# fields are seed-independent and data-independent (locked code constants, or
# a deterministic function of the one shared train file), so for seed 42 they
# are instead cross-checked against another seed's concurrent_44 run and the
# multiseed report's independently recovered value.
HISTORICAL_GAP_SEED = 42
HISTORICAL_GAP_KEYS = {
    "effective_seed_settings",
    "training_control",
    "diploma_gpa_handling",
    "data_rows",
}
CROSS_CHECK_REFERENCE_SEED = 52  # any other seed's concurrent_44 run

# Keys in feature_contract.json that must match byte-for-byte between the two
# contracts (everything EXCEPT the feature list / contract identity itself,
# and except dtypes_after_model_preparation, which is checked separately
# because it is keyed by feature name and legitimately has one fewer key).
SHARED_CONTRACT_KEYS = [
    "categorical_features",
    "categorical_levels",
    "categorical_levels_learned_from",
    "unknown_category_code",
    "derived_feature_sources",
    "dropped_feature_guard",
    "target_m1_classifier",
    "target_m2_regressor",
    "reporting_threshold",
    "reporting_threshold_policy",
    "random_seed",
    "effective_seed_settings",
    "lightgbm_params",
    "training_control",
    "diploma_gpa_handling",
    "data_rows",
    "requires_concurrent_plan_context",
    "test_policy",
    "train_path",
    "valid_path",
    "dataset_version",
    "dataset_inputs",
]

SHARED_RUN_SETTINGS_KEYS = [
    "reporting_threshold",
    "test_policy",
    "random_seed",
    "effective_seed_settings",
    "num_threads",
    "training_control",
    "diploma_gpa_handling",
    "data_rows",
    "train_path",
    "valid_path",
    "dataset_version",
]

EXPECTED_FEATURE_DIFF = {"concurrent_peer_difficulty_missing"}


def load(run_id: str, filename: str) -> Dict[str, Any]:
    path = RUNS / run_id / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_pair(
    seed: int,
    run_44: str,
    run_43: str,
    reference_contract_44: Dict[str, Any],
    reference_run_settings_44: Dict[str, Any],
) -> Dict[str, Any]:
    contract_44 = load(run_44, "feature_contract.json")
    contract_43 = load(run_43, "feature_contract.json")
    metrics_44 = load(run_44, "metrics.json")
    metrics_43 = load(run_43, "metrics.json")

    checks: Dict[str, bool] = {}
    mismatches: Dict[str, Any] = {}
    notes: Dict[str, str] = {}

    is_historical_gap_seed = seed == HISTORICAL_GAP_SEED

    for key in SHARED_CONTRACT_KEYS:
        if is_historical_gap_seed and key in HISTORICAL_GAP_KEYS:
            # concurrent_44's own seed-42 record is null (historical gap);
            # cross-check concurrent_43's value against a reference seed's
            # concurrent_44 run instead, since this field is seed-independent.
            if key == "effective_seed_settings":
                continue  # verified separately against the recovered value
            ok = contract_43.get(key) == reference_contract_44.get(key)
            checks[f"contract.{key}_cross_checked_vs_seed{CROSS_CHECK_REFERENCE_SEED}"] = ok
            notes[f"contract.{key}"] = (
                f"seed {seed}'s own concurrent_44 run predates this field "
                f"(recorded null); cross-checked against seed "
                f"{CROSS_CHECK_REFERENCE_SEED}'s concurrent_44 run instead, "
                "since the field is seed-independent."
            )
            if not ok:
                mismatches[f"contract.{key}"] = {
                    f"concurrent_44 (seed {CROSS_CHECK_REFERENCE_SEED} reference)": reference_contract_44.get(key),
                    "concurrent_43": contract_43.get(key),
                }
            continue
        ok = contract_44.get(key) == contract_43.get(key)
        checks[f"contract.{key}"] = ok
        if not ok:
            mismatches[f"contract.{key}"] = {
                "concurrent_44": contract_44.get(key),
                "concurrent_43": contract_43.get(key),
            }

    # effective_seed_settings for seed 42: verified against the value
    # independently recovered in the multiseed report (contract_verification).
    if is_historical_gap_seed:
        multiseed_report = json.loads(MULTISEED_REPORT_PATH.read_text(encoding="utf-8"))
        recovered = multiseed_report["contract_verification"]["42"]["effective_seed_settings"]
        ok = contract_43.get("effective_seed_settings") == recovered
        checks["contract.effective_seed_settings_vs_multiseed_report_recovery"] = ok
        if not ok:
            mismatches["contract.effective_seed_settings"] = {
                "multiseed_report_recovered": recovered,
                "concurrent_43": contract_43.get("effective_seed_settings"),
            }

    # dtypes_after_model_preparation: keyed by feature name, so concurrent_44
    # legitimately has exactly one more key (the dropped legacy indicator).
    # Every SHARED feature's dtype must still match exactly.
    dtypes_44 = contract_44.get("dtypes_after_model_preparation", {})
    dtypes_43 = contract_43.get("dtypes_after_model_preparation", {})
    shared_dtype_features = set(dtypes_44) & set(dtypes_43)
    dtypes_match = all(dtypes_44[f] == dtypes_43[f] for f in shared_dtype_features)
    checks["dtypes_identical_on_shared_features"] = dtypes_match
    checks["dtypes_key_diff_is_exactly_legacy_indicator"] = (
        set(dtypes_44) - set(dtypes_43) == EXPECTED_FEATURE_DIFF
        and set(dtypes_43) - set(dtypes_44) == set()
    )
    if not dtypes_match:
        mismatches["dtypes_after_model_preparation"] = {
            f: {"concurrent_44": dtypes_44[f], "concurrent_43": dtypes_43[f]}
            for f in shared_dtype_features
            if dtypes_44[f] != dtypes_43[f]
        }

    run_settings_44 = metrics_44.get("run_settings", {})
    run_settings_43 = metrics_43.get("run_settings", {})
    for key in SHARED_RUN_SETTINGS_KEYS:
        if is_historical_gap_seed and key in HISTORICAL_GAP_KEYS:
            if key == "effective_seed_settings":
                continue
            ok = run_settings_43.get(key) == reference_run_settings_44.get(key)
            checks[f"run_settings.{key}_cross_checked_vs_seed{CROSS_CHECK_REFERENCE_SEED}"] = ok
            if not ok:
                mismatches[f"run_settings.{key}"] = {
                    f"concurrent_44 (seed {CROSS_CHECK_REFERENCE_SEED} reference)": reference_run_settings_44.get(key),
                    "concurrent_43": run_settings_43.get(key),
                }
            continue
        ok = run_settings_44.get(key) == run_settings_43.get(key)
        checks[f"run_settings.{key}"] = ok
        if not ok:
            mismatches[f"run_settings.{key}"] = {
                "concurrent_44": run_settings_44.get(key),
                "concurrent_43": run_settings_43.get(key),
            }

    # contract identity itself must differ, feature count must differ by 1.
    checks["contract_name_differs"] = (
        contract_44["contract_name"] != contract_43["contract_name"]
    )
    checks["feature_count_differs_by_one"] = (
        contract_44["feature_count"] - contract_43["feature_count"] == 1
    )

    # feature-set difference is exactly the dropped legacy indicator.
    diff_44_minus_43 = set(contract_44["ordered_features"]) - set(
        contract_43["ordered_features"]
    )
    diff_43_minus_44 = set(contract_43["ordered_features"]) - set(
        contract_44["ordered_features"]
    )
    checks["feature_diff_is_exactly_legacy_indicator"] = (
        diff_44_minus_43 == EXPECTED_FEATURE_DIFF and diff_43_minus_44 == set()
    )
    if not checks["feature_diff_is_exactly_legacy_indicator"]:
        mismatches["feature_diff"] = {
            "concurrent_44_minus_concurrent_43": sorted(diff_44_minus_43),
            "concurrent_43_minus_concurrent_44": sorted(diff_43_minus_44),
        }

    # concurrent_43's 43 features must equal concurrent_44's 44 features with
    # the legacy indicator removed, order preserved (order-preserving identity).
    checks["order_preserving_identity"] = (
        [f for f in contract_44["ordered_features"] if f != "concurrent_peer_difficulty_missing"]
        == contract_43["ordered_features"]
    )

    # TEST must be closed for both.
    checks["test_closed_44"] = contract_44["test_policy"] == "closed_not_read"
    checks["test_closed_43"] = contract_43["test_policy"] == "closed_not_read"
    checks["test_null_44"] = (
        metrics_44["m1_pass_classifier"]["test"] is None
        and metrics_44["m2_grade_regressor"]["test"] is None
    )
    checks["test_null_43"] = (
        metrics_43["m1_pass_classifier"]["test"] is None
        and metrics_43["m2_grade_regressor"]["test"] is None
    )

    all_ok = all(checks.values())
    return {
        "seed": seed,
        "run_44": run_44,
        "run_43": run_43,
        "valid": all_ok,
        "checks": checks,
        "notes": notes,
        "mismatches": mismatches,
    }


def main() -> int:
    reference_contract_44 = load(
        PAIRS[CROSS_CHECK_REFERENCE_SEED]["concurrent_44"], "feature_contract.json"
    )
    reference_metrics_44 = load(
        PAIRS[CROSS_CHECK_REFERENCE_SEED]["concurrent_44"], "metrics.json"
    )
    reference_run_settings_44 = reference_metrics_44.get("run_settings", {})

    results = {}
    overall_ok = True
    for seed, pair in PAIRS.items():
        result = verify_pair(
            seed,
            pair["concurrent_44"],
            pair["concurrent_43"],
            reference_contract_44,
            reference_run_settings_44,
        )
        results[seed] = result
        overall_ok = overall_ok and result["valid"]
        status = "PASS" if result["valid"] else "FAIL"
        print(f"seed {seed}: {status}")
        if not result["valid"]:
            print(json.dumps(result["mismatches"], indent=2))

    out_path = RUNS / "CONCURRENT43_VS_CONCURRENT44_VERIFICATION.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("OVERALL:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
