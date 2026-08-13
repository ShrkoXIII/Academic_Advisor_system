"""Named model-input feature contracts and the constants they pin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple


# ---------------------------------------------------------------------------
# Feature contracts — explicit named registries
# ---------------------------------------------------------------------------
# These are explicit allowlists. They are NOT derived from fe.FEATURE_COLUMNS,
# because that list contains diagnostic / string / leakage columns and is not a
# model allowlist.
#
# Three contracts exist so the concurrent-feature effect can be measured in
# isolation:
#
#   baseline_41   the established list including GPA trend, WITHOUT the three
#                 concurrent model features
#   concurrent_44 baseline_41 plus exactly those three
#   concurrent_43 concurrent_44 minus concurrent_peer_difficulty_missing, the
#                 one of the three found effectively dead (Decisions_Log.md:
#                 used by any model in only 2 of 5 seeds, under 0.001% of
#                 total gain, zero splits everywhere else)
#
# ORDER NOTE (deliberate, see the module tests): the three concurrent features
# sit at zero-based positions 33/34/35 of the accepted 44-feature contract, not
# at the end. That position is pinned by the dataset builder
# (scripts/build_concurrent_group_features.py EXPECTED_LEGACY_MODEL_POSITION=35)
# and by tests. baseline_41 is therefore the 44-list with those three removed,
# preserving every other feature's relative order. The relationship is a
# set-difference identity, not a list concatenation. concurrent_43 preserves
# the order of the remaining 43 features exactly as they sit in concurrent_44.

CONCURRENT_MODEL_FEATURES: List[str] = [
    "concurrent_peer_difficulty_mean",
    "concurrent_peer_difficulty_max",
    "concurrent_peer_difficulty_missing",
]

CONCURRENT_44_FEATURES: List[str] = [
    # --- Student GPA history ---
    "prev_gpa_points_clean",                  # zero + flags design (NOT model_prev_gpa)
    "start_agpa_points",                      # cumulative AGPA at semester start
    "last_valid_gpa_before_current_semester", # last real GPA, skips interruptions
    "gpa_trend_delta",                        # last valid GPA minus second-last valid GPA
    "gpa_trend_missing",                      # trend undefined until two prior valid GPAs
    # --- Student workload / fail history ---
    "start_total_in_credits",
    "start_total_in_courses",
    "total_fail_credits_capped",
    "fail_credit_ratio_capped",
    "is_extreme_fail_history",
    "reg_total_semesters",
    # --- Academic standing ---
    "start_level_ord",
    "start_semester",                         # 1/2/3 term started (NOT start_year)
    # --- Timeline / interruptions (PAST-ONLY; is_interruption_semester dropped) ---
    "prior_interruption_count",
    "consecutive_interruption_count",
    "prev_semester_was_interruption",
    "no_previous_progress",
    "is_first_active_semester",
    "is_first_row_in_timeline",
    # --- Previous-GPA data-quality flags ---
    "prev_gpa_points_missing",
    "prev_gpa_points_zero",
    "prev_gpa_invalid_zero_case",
    # --- Current semester context ---
    "semester_reg_credits",
    "semester_reg_courses",
    "part_semester",                          # 1/2/3 term (NOT part_year)
    # --- Course properties ---
    "course_credits",
    "attempt_number",
    "is_high_credit_course",
    # --- Course difficulty (empirical, train-only) ---
    "course_pass_rate_historical",
    "course_avg_mark_historical",
    "course_retake_rate_historical",
    "course_history_count",                   # support count -> reliability of difficulty stats
    "course_difficulty_missing",
    # --- Concurrent course group (same-semester peer difficulty, LOO) ---
    "concurrent_peer_difficulty_mean",        # mean (1 - pass_rate) over valid peers
    "concurrent_peer_difficulty_max",         # hardest valid peer
    "concurrent_peer_difficulty_missing",     # 1 when the peer set is empty (singleton)
    # --- Degree / requirement context ---
    "requirement_type_id",                    # CATEGORICAL (handled specially)
    "requirement_type_missing",
    "degree_requirement_credits_count",
    "degree_requirement_credits_count_missing",
    "course_share_of_requirement",
    "requirement_size_bucket_ord",            # ordinal; raw string never enters X
    # --- Pre-admission diploma signals ---
    "diploma_gpa",
    "diploma_type_bucket",                    # CATEGORICAL; raw diploma_type_id stays audit-only
]

# baseline_41 preserves the order of every non-concurrent feature exactly.
BASELINE_41_FEATURES: List[str] = [
    feature
    for feature in CONCURRENT_44_FEATURES
    if feature not in set(CONCURRENT_MODEL_FEATURES)
]

# DEPRECATED alias kept so existing importers keep working unchanged
# (src/inference.py, scripts/build_*.py, notebooks). New code should resolve a
# named contract via resolve_feature_contract() instead of reading this global.
MODEL_FEATURES: List[str] = CONCURRENT_44_FEATURES

# --- binding contract invariants (checked at import) ---
assert len(BASELINE_41_FEATURES) == 41, len(BASELINE_41_FEATURES)
assert len(CONCURRENT_44_FEATURES) == 44, len(CONCURRENT_44_FEATURES)
assert len(set(BASELINE_41_FEATURES)) == 41, "duplicate in BASELINE_41_FEATURES"
assert len(set(CONCURRENT_44_FEATURES)) == 44, "duplicate in CONCURRENT_44_FEATURES"
assert set(CONCURRENT_44_FEATURES) - set(BASELINE_41_FEATURES) == set(
    CONCURRENT_MODEL_FEATURES
), "concurrent_44 minus baseline_41 must be exactly the three concurrent features"
assert set(BASELINE_41_FEATURES) - set(CONCURRENT_44_FEATURES) == set(), (
    "baseline_41 must be a subset of concurrent_44"
)
# Order-preserving form of "44 = 41 + the three": dropping the concurrent
# features from the 44-list reproduces the 41-list exactly, in order.
assert [
    f for f in CONCURRENT_44_FEATURES if f not in set(CONCURRENT_MODEL_FEATURES)
] == BASELINE_41_FEATURES
for _gpa_feature in ("gpa_trend_delta", "gpa_trend_missing"):
    assert _gpa_feature in BASELINE_41_FEATURES, _gpa_feature
    assert _gpa_feature in CONCURRENT_44_FEATURES, _gpa_feature

# concurrent_43 = concurrent_44 minus the dead legacy indicator, order preserved.
CONCURRENT_43_FEATURES: List[str] = [
    feature
    for feature in CONCURRENT_44_FEATURES
    if feature != "concurrent_peer_difficulty_missing"
]

assert len(CONCURRENT_43_FEATURES) == 43, len(CONCURRENT_43_FEATURES)
assert len(set(CONCURRENT_43_FEATURES)) == 43, "duplicate in CONCURRENT_43_FEATURES"
assert "concurrent_peer_difficulty_missing" not in CONCURRENT_43_FEATURES
assert set(CONCURRENT_44_FEATURES) - set(CONCURRENT_43_FEATURES) == {
    "concurrent_peer_difficulty_missing"
}, "concurrent_44 minus concurrent_43 must be exactly the dead legacy indicator"
assert set(CONCURRENT_43_FEATURES) - set(CONCURRENT_44_FEATURES) == set(), (
    "concurrent_43 must be a subset of concurrent_44"
)
for _gpa_feature in ("gpa_trend_delta", "gpa_trend_missing"):
    assert _gpa_feature in CONCURRENT_43_FEATURES, _gpa_feature
for _remaining_concurrent_feature in (
    "concurrent_peer_difficulty_mean",
    "concurrent_peer_difficulty_max",
):
    assert _remaining_concurrent_feature in CONCURRENT_43_FEATURES, _remaining_concurrent_feature

# Columns deliberately EXCLUDED (kept here as a guard list, asserted absent from X).
DROPPED_FEATURES: List[str] = [
    "is_interruption_semester",          # LEAKAGE: current-semester pass_credits/gpa_points
    "model_prev_gpa",                    # depends on structural_zero_as_nan; use clean + flags
    "prev_gpa_actual_zero_performance",  # dead stub, hardcoded 0
    "start_level_missing",               # audit-only; constant in current train population
    "difficulty_fallback_level",         # audit-only; train is intentionally all Level 1
    "part_year",                         # absolute calendar year -> drift
    "start_year",                        # absolute calendar year -> drift
    "difficulty_group_support_count",    # audit only
]

CATEGORICAL_FEATURES: List[str] = ["requirement_type_id", "diploma_type_bucket"]
UNKNOWN_CATEGORY = -1   # NaN + categories unseen in train map here

REQUIREMENT_BUCKET_ORD = {
    "none_or_unknown": 0,
    "small": 1,
    "medium": 2,
    "large": 3,
    "very_large": 4,
}

# Patterns that indicate a leftover composite/lookup key from difficulty building.
_LEFTOVER_KEY_PATTERNS = ["l3_key", "l4_key", "_key", "tmp", "temp", "idx"]

TARGET_GRADE = "final_mark"       # M2 regressor target
EXPECTED_FEATURE_COUNT = 44       # DEPRECATED alias: concurrent_44's count
DERIVED_FEATURE_SOURCES = {
    "requirement_size_bucket_ord": "requirement_size_bucket",
}
SEGMENT_ONLY_COLUMNS = ["difficulty_fallback_level"]

# LOCKED reporting threshold. Precision / recall / F1 / confusion matrix are all
# reported at this probability cut. It is NOT a training or early-stopping
# parameter, and it is NOT optimized per run — it is fixed so runs stay
# comparable. Runs made in the earlier 0.5 / 0.85 era are therefore NOT
# P/R-comparable with runs made at this threshold.
REPORTING_THRESHOLD = 0.80

TARGET_M1_DEFINITION = "(final_mark >= 50).astype(int)"
TARGET_M2_DEFINITION = "final_mark"


@dataclass(frozen=True)
class FeatureContract:
    """One immutable, explicitly named model-input contract."""

    name: str
    version: str
    features: Tuple[str, ...]
    categorical_features: Tuple[str, ...]
    derived_feature_sources: Mapping[str, str]
    reporting_threshold: float
    requires_concurrent_plan_context: bool
    description: str

    @property
    def expected_feature_count(self) -> int:
        return len(self.features)

    @property
    def source_features(self) -> List[str]:
        """Contract features that are read from the parquet as-is."""
        return [c for c in self.features if c not in self.derived_feature_sources]

    @property
    def numeric_features(self) -> List[str]:
        return [c for c in self.features if c not in self.categorical_features]

    @property
    def training_data_columns(self) -> List[str]:
        """Only the columns this contract needs: features, derivations, target, segments.

        The final parquet files carry many audit/string columns that training
        never consumes; loading them added hundreds of MiB per process. The
        baseline_41 contract simply does not list the concurrent columns, so
        they stay on disk.
        """
        return list(dict.fromkeys(
            self.source_features
            + list(self.derived_feature_sources.values())
            + [TARGET_GRADE]
            + SEGMENT_ONLY_COLUMNS
        ))


BASELINE_41_CONTRACT = FeatureContract(
    name="baseline_41",
    version="v1",
    features=tuple(BASELINE_41_FEATURES),
    categorical_features=tuple(CATEGORICAL_FEATURES),
    derived_feature_sources=DERIVED_FEATURE_SOURCES,
    reporting_threshold=REPORTING_THRESHOLD,
    requires_concurrent_plan_context=False,
    description=(
        "Established feature list including GPA trend, excluding the three "
        "concurrent peer-difficulty features. Controlled-experiment baseline."
    ),
)

CONCURRENT_44_CONTRACT = FeatureContract(
    name="concurrent_44",
    version="v1",
    features=tuple(CONCURRENT_44_FEATURES),
    categorical_features=tuple(CATEGORICAL_FEATURES),
    derived_feature_sources=DERIVED_FEATURE_SOURCES,
    reporting_threshold=REPORTING_THRESHOLD,
    requires_concurrent_plan_context=True,
    description=(
        "baseline_41 plus concurrent_peer_difficulty_{mean,max,missing}. "
        "Valid at serve time only once a full plan exists and score_plan has "
        "recomputed the same-semester peer context."
    ),
)

CONCURRENT_43_CONTRACT = FeatureContract(
    name="concurrent_43",
    version="v1",
    features=tuple(CONCURRENT_43_FEATURES),
    categorical_features=tuple(CATEGORICAL_FEATURES),
    derived_feature_sources=DERIVED_FEATURE_SOURCES,
    reporting_threshold=REPORTING_THRESHOLD,
    requires_concurrent_plan_context=True,
    description=(
        "concurrent_44 minus concurrent_peer_difficulty_missing (dead: used by "
        "any model in only 2 of 5 seeds, under 0.001% of total gain, zero "
        "splits everywhere else; see Decisions_Log.md). Still requires plan "
        "context at serve time because concurrent_peer_difficulty_mean/max "
        "remain in the contract."
    ),
)

# Known production limitation, recorded with every run so it cannot be lost.
# NOT wired into recommendation by this experiment.
SERVING_LIMITATION_NOTE = (
    "score() pre-plan candidate ranking must use baseline_41; concurrent_44 is "
    "semantically valid only after a complete plan is formed and score_plan "
    "recomputes the same-semester concurrent context."
)

FEATURE_CONTRACTS: Dict[str, FeatureContract] = {
    BASELINE_41_CONTRACT.name: BASELINE_41_CONTRACT,
    CONCURRENT_44_CONTRACT.name: CONCURRENT_44_CONTRACT,
    CONCURRENT_43_CONTRACT.name: CONCURRENT_43_CONTRACT,
}

DEFAULT_FEATURE_CONTRACT = CONCURRENT_44_CONTRACT.name


def resolve_feature_contract(
    contract: "str | FeatureContract | None" = None,
) -> FeatureContract:
    """Resolve a contract by name. Never infers one from available columns."""
    if isinstance(contract, FeatureContract):
        return contract
    if contract is None:
        return FEATURE_CONTRACTS[DEFAULT_FEATURE_CONTRACT]
    try:
        return FEATURE_CONTRACTS[contract]
    except KeyError:
        raise ValueError(
            f"Unknown feature contract {contract!r}. "
            f"Choose one of: {sorted(FEATURE_CONTRACTS)}"
        ) from None


# DEPRECATED alias: the concurrent_44 column set. Kept for existing importers.
TRAINING_DATA_COLUMNS = CONCURRENT_44_CONTRACT.training_data_columns
