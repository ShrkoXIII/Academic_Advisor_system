"""Phase 2 (REVISED) read-only proposal tables: split/merge, degree lineage,
course link, difficulty prototype.

Revision context
-----------------
The prior version of this task shipped a normalization "FIX" line and a hard
67/67 gate, both of which turned out to be wrong: the FIX line was a no-op
(digit extraction already captured a glued trailing digit; inserting a space
before it changes nothing), and the pair it claimed to fix
(``بنيان الحواسيب`` / ``بنيان الحواسيب1``) is not a rename at all -- catalog
evidence shows the old course was **split** into two new courses
(``1183.111``, ``1192.111``), later repeated as a second generation
(``1451.111``, ``1454.111``). The FIX line has been removed; the gate now
requires exactly 66/67 with that specific pair as the sole, expected
non-match; and a new Table 0 detects splits/merges catalog-wide so they are
excluded from ordinary predecessor matching in Table 2.

Guarantees
----------
* Reads only: the frozen TRAIN/VALID parquets of dataset version
  ``2026-07-26_batched_fixes__registration_roster_concurrent``, the read-only
  cleaned catalog (``clean_v_acd_degree_course.parquet``), and
  ``COURSE_IDENTITY_67_HUMAN_REVIEW.csv``. TEST is never referenced by path,
  read, globbed or stat-ed anywhere in this module.
* ``final_mark`` is loaded from TRAIN (an allowed outcome column there -- it is
  needed for raw TRAIN pass-rate/avg-mark statistics) but is NEVER included in
  the VALID column list. Asserted at runtime immediately after the VALID read.
* No model is loaded, trained, fit, or scored, and no dataset/feature frame
  under ``data/model_data/`` is written or modified. ``src/course_difficulty.py``
  is imported READ-ONLY for one purpose only: the Table-3 validation step
  reuses ``fit_difficulty_state``/``apply_difficulty_state`` (the exact method
  Phase 0 used and validated at 0/156,097 mismatches) to obtain the
  authoritative Level-1/Level-2 support-count selection for VALID rows drawn
  from a state fit on the *complete* TRAIN split. This is a documented
  deviation from building Table 3's own from-scratch temporal walk (which is
  restricted to a sampled course scope, per the task's tractability
  instruction, and therefore cannot alone cover all 156,097 VALID rows for
  the mismatch check without silently risking divergence from production's
  tie-break rules). Table 3 itself is an independent recomputation.
* Writes exactly four CSVs under ``models/runs/`` and prints the JSON payload
  the markdown report is written from by hand.

Interpretive choices made where the Phase 2 spec is under-determined (stated
here, and again in the report, per the project's convention of declaring
deviations rather than silently resolving them):

1. "Name key" (Table 1's overlap computation, Table 2's matching) is the
   single ``norm_name(course_name_sl)`` string defined by the spec -- there is
   no narrow/wide split in the Phase 2 spec (that was a Phase 1-only
   distinction). Table 1 degree overlap is computed over the *set* of name
   keys of each degree's catalog course listing (deduplicated by course_id).
2. ``degree_name_similarity`` reuses ``name_similarity`` from
   ``scripts/course_identity_investigation.py`` (max of sequence-ratio and
   token-Jaccard over the normalized strings; the same function and the same
   0.60 threshold the repository's existing ``degree_lineage()`` already
   uses for "plausible"), applied to ``degree_family(new)`` vs
   ``degree_family(old)`` -- i.e. after the spec's own 2023/suffix strip.
3. Table 1 candidate ranking (top 3): primary key ``overlap_pct_of_new``
   descending, tie-broken by ``jaccard`` descending, then ``old_degree_id``
   ascending, for determinism.
4. Table 2 Step 4 edge case not covered by the spec's three bullet points: a
   *specific*-scope new course that nonetheless resolves >=2 eligible
   predecessors (possible when two differently-scoped old courses share a
   normalized name). Documented choice: classified ``consolidated_into``
   like the shared case, since the many-to-one relationship is the same;
   noted in that row's ``notes``.
5. Table 2's per-degree same-family fallback (spec: "If Table 1 gives no
   candidate for the relevant degree, fall back to same-family search")
   triggers when the new degree's Table-1 rank-1 candidate has
   ``shared_course_key_count == 0`` (i.e. the "candidate" carries no actual
   evidence). The fallback searches *all* old degrees with
   ``degree_family(old) == degree_family(new)`` -- not just Table 1's top 3
   -- since the spec asks for a search, not a re-ranking of already-computed
   candidates.
6. Table 0 split/merge clustering: when a ``name_stem`` has exactly one
   no-suffix course on one generation side and >=2 suffixed courses on the
   other, the suffixed courses are grouped into connected components by
   shared ``degree_id`` (pairwise edge if two courses share >=1 degree_id).
   Each component of size >=2 is a separate split/merge row (a "generation");
   a lone, unconnected suffixed course is not a candidate by itself, since
   the spec's rule requires >=2 *and* mutual degree sharing.
   ``generations_detected`` counts all qualifying components sharing that
   stem+direction, repeated on every row belonging to that stem+direction.
7. Table 3's ``as_of_part_id`` grid is the 55 distinct TRAIN ``part_id``
   values (the same global semester grid ``build_temporal_train`` walks in
   ``src/course_difficulty.py``), not a per-course subset -- this matches the
   only precedented definition of "TRAIN semester" in this codebase.
   Level-1 rows are restricted to (course_id, degree_id) pairs that actually
   occur together in TRAIN (a pair that never co-occurs has no meaningful
   "as of" history to walk).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"
VERSION_DIR = ROOT / "data" / "model_data" / "versions" / VERSION
TRAIN_PATH = VERSION_DIR / "df_train_final.parquet"
VALID_PATH = VERSION_DIR / "df_valid_final.parquet"

CATALOG_PATH = ROOT / "data" / "preprocessed" / "V_ACD_DEGREE_COURSE" / "clean_v_acd_degree_course.parquet"
PAIRS_PATH = ROOT / "models" / "runs" / "COURSE_IDENTITY_67_HUMAN_REVIEW.csv"

OUT_SPLIT_CSV = ROOT / "models" / "runs" / "course_split_candidates.csv"
OUT_LINEAGE_CSV = ROOT / "models" / "runs" / "degree_lineage_proposed.csv"
OUT_LINK_CSV = ROOT / "models" / "runs" / "course_link_proposed.csv"
OUT_STATS_CSV = ROOT / "models" / "runs" / "course_difficulty_stats_prototype.csv"

UNIVERSITY_ID = "111"
MIN_TRAIN_SUPPORT = 20  # predecessor eligibility threshold (spec: train_support >= 20)
COURSE_GENERATION_BOUNDARY = 1150  # numeric course_id proxy boundary (Phase 1 Q7 precedent)
RANDOM_SEED = 42
SAMPLE_OTHER_COURSES = 200

EXPECTED_GATE_MATCHED = 66
EXPECTED_GATE_NONMATCH_NEW = "1183.111"
EXPECTED_GATE_NONMATCH_OLD = "510.111"

TRAIN_COLUMNS = [
    "course_id",
    "degree_id",
    "faculty_id",
    "requirement_type_id",
    "course_credits",
    "part_id",
    "final_mark",
    "attempt_number",
    "university_id",
    "degree_course_key",
]

# final_mark is intentionally ABSENT -- VALID outcomes are never loaded.
VALID_COLUMNS = [
    "course_id",
    "degree_id",
    "faculty_id",
    "requirement_type_id",
    "course_credits",
    "part_id",
    "course_difficulty_missing",
    "course_history_count",
    "degree_course_key",
]


# ---------------------------------------------------------------------------
# Normalization spec (verbatim from the task; do not modify)
# ---------------------------------------------------------------------------
AR2LAT = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).translate(AR2LAT)
    s = re.sub(r"[ً-ْٰـ]", "", s)  # harakat + tatweel
    for a, b in [
        ("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"),
        ("ة", "ه"), ("ؤ", "و"), ("ئ", "ي"),
    ]:
        s = s.replace(a, b)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    digits = re.findall(r"\d+", s)
    base = re.sub(r"\s+", " ", re.sub(r"\d+", " ", s)).strip()
    tokens = [re.sub(r"^ال", "", t) if len(t) > 3 else t for t in base.split()]
    return " ".join(tokens) + ("#" + digits[-1] if digits else "")


def name_stem(s: str) -> str:
    """The key with any level suffix removed. Used for split/merge detection only."""
    return norm_name(s).split("#")[0]


def degree_family(degree_name: str) -> str:
    x = re.sub(r"\s*20\d{2}\s*$", "", str(degree_name)).strip()
    return norm_name(x.split("/")[0].strip())


def modal(series: pd.Series) -> Any:
    counts = series.value_counts()
    top = counts.max()
    candidates = sorted(counts[counts == top].index.tolist())
    return candidates[0]


def load_course_identity_investigation():
    path = ROOT / "scripts" / "course_identity_investigation.py"
    spec = importlib.util.spec_from_file_location("course_identity_investigation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_catalog() -> pd.DataFrame:
    cat = pd.read_parquet(CATALOG_PATH)
    for col in ("course_id", "degree_id"):
        cat[col] = cat[col].astype("string")
    cat["degree_numeric"] = cat["degree_id"].str.split(".").str[0].astype("int64")
    cat["course_numeric"] = cat["course_id"].str.split(".").str[0].astype("int64")
    cat["name_key"] = cat["course_name_sl"].map(norm_name)
    cat["name_stem"] = cat["name_key"].str.split("#").str[0]
    cat["has_level_suffix"] = cat["name_key"].str.contains("#")
    return cat


def load_train() -> pd.DataFrame:
    train = pd.read_parquet(TRAIN_PATH, columns=TRAIN_COLUMNS)
    for col in ("course_id", "degree_id", "faculty_id", "degree_course_key"):
        train[col] = train[col].astype("string")
    train["part_id"] = pd.to_numeric(train["part_id"], errors="raise").astype("int64")
    return train


def load_valid() -> pd.DataFrame:
    valid = pd.read_parquet(VALID_PATH, columns=VALID_COLUMNS)
    if "final_mark" in valid.columns:
        raise SystemExit("STOP: a VALID outcome column was loaded.")
    for col in ("course_id", "degree_id", "faculty_id", "degree_course_key"):
        valid[col] = valid[col].astype("string")
    valid["part_id"] = pd.to_numeric(valid["part_id"], errors="raise").astype("int64")
    return valid


# ---------------------------------------------------------------------------
# Normalization validation gate
# ---------------------------------------------------------------------------
def validate_normalization_gate(pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for r in pairs.itertuples(index=False):
        rows.append(
            {
                "new_course_id": r.new_course_id,
                "old_course_id": r.old_course_id,
                "new_course_name": r.new_course_name,
                "old_course_name": r.old_course_name,
                "new_key": norm_name(r.new_course_name),
                "old_key": norm_name(r.old_course_name),
            }
        )
    frame = pd.DataFrame(rows)
    frame["match"] = frame["new_key"] == frame["old_key"]
    matched = int(frame["match"].sum())
    mismatches = frame.loc[~frame["match"]]

    expected_nonmatch = mismatches.loc[
        (mismatches["new_course_id"] == EXPECTED_GATE_NONMATCH_NEW)
        & (mismatches["old_course_id"] == EXPECTED_GATE_NONMATCH_OLD)
    ]
    gate_pass = (
        matched == EXPECTED_GATE_MATCHED
        and len(mismatches) == 1
        and len(expected_nonmatch) == 1
    )
    result = {
        "pairs_total": int(len(frame)),
        "matched": matched,
        "expected_matched": EXPECTED_GATE_MATCHED,
        "mismatches": mismatches.to_dict(orient="records"),
        "expected_nonmatch_found": len(expected_nonmatch) == 1,
        "gate_pass": bool(gate_pass),
    }
    return frame, result


# ---------------------------------------------------------------------------
# Table 0 -- split / merge candidates
# ---------------------------------------------------------------------------
def _cluster_by_shared_degree(course_ids: list[str], degree_map: dict[str, set[str]]) -> list[list[str]]:
    parent = {c: c for c in course_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(course_ids)):
        for j in range(i + 1, len(course_ids)):
            a, b = course_ids[i], course_ids[j]
            if degree_map.get(a, set()) & degree_map.get(b, set()):
                union(a, b)

    groups: dict[str, list[str]] = {}
    for c in course_ids:
        groups.setdefault(find(c), []).append(c)
    return list(groups.values())


def build_split_merge_candidates(catalog: pd.DataFrame, train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    dedup = catalog.drop_duplicates("course_id").copy()
    dedup["generation"] = np.where(dedup["course_numeric"] >= COURSE_GENERATION_BOUNDARY, "new", "old")

    catalog_names = dict(zip(dedup["course_id"], dedup["course_name_sl"]))
    catalog_credits = dict(zip(dedup["course_id"], dedup["course_credits"].astype("float64")))
    degree_map: dict[str, set[str]] = catalog.groupby("course_id")["degree_id"].apply(lambda s: set(s.tolist())).to_dict()

    train_support = train.groupby("course_id").size().to_dict()
    valid_rows = valid_row_counts_all = valid.groupby("course_id").size().to_dict() if "course_id" in valid.columns else {}

    rows: list[dict[str, Any]] = []

    def emit(direction: str, stem: str, single_ids: list[str], group_ids: list[str], n_generations: int) -> None:
        if direction == "split":
            old_ids, new_ids = single_ids, group_ids
        else:
            new_ids, old_ids = single_ids, group_ids

        old_names = [catalog_names.get(c, "") for c in old_ids]
        new_names = [catalog_names.get(c, "") for c in new_ids]
        old_total_credits = float(sum(catalog_credits.get(c, 0.0) for c in old_ids))
        new_total_credits = float(sum(catalog_credits.get(c, 0.0) for c in new_ids))
        old_support = int(sum(train_support.get(c, 0) for c in old_ids))
        new_valid = int(sum(valid_row_counts_all.get(c, 0) for c in new_ids))

        degree_sets = [degree_map.get(c, set()) for c in group_ids]
        shared_degrees = set.intersection(*degree_sets) if degree_sets else set()
        if not shared_degrees and degree_sets:
            shared_degrees = set.union(*degree_sets)

        rows.append(
            {
                "university_id": UNIVERSITY_ID,
                "name_stem": stem,
                "direction": direction,
                "old_course_ids": "|".join(sorted(old_ids)),
                "old_course_names": "|".join(old_names),
                "old_total_credits": old_total_credits,
                "old_train_support": old_support,
                "new_course_ids": "|".join(sorted(new_ids)),
                "new_course_names": "|".join(new_names),
                "new_total_credits": new_total_credits,
                "new_valid_rows": new_valid,
                "shared_degree_ids": "|".join(sorted(shared_degrees)),
                "credit_change": new_total_credits - old_total_credits,
                "generations_detected": n_generations,
                "review_decision": "",
                "reviewer_name": "",
                "review_date": "",
                "review_notes": "",
            }
        )

    for stem, g in dedup.groupby("name_stem"):
        old_nosuf = g.loc[(g["generation"] == "old") & (~g["has_level_suffix"]), "course_id"].tolist()
        new_suf = g.loc[(g["generation"] == "new") & (g["has_level_suffix"]), "course_id"].tolist()
        if len(old_nosuf) == 1 and len(new_suf) >= 2:
            clusters = [c for c in _cluster_by_shared_degree(new_suf, degree_map) if len(c) >= 2]
            for c in clusters:
                emit("split", stem, old_nosuf, sorted(c), len(clusters))

        new_nosuf = g.loc[(g["generation"] == "new") & (~g["has_level_suffix"]), "course_id"].tolist()
        old_suf = g.loc[(g["generation"] == "old") & (g["has_level_suffix"]), "course_id"].tolist()
        if len(new_nosuf) == 1 and len(old_suf) >= 2:
            clusters = [c for c in _cluster_by_shared_degree(old_suf, degree_map) if len(c) >= 2]
            for c in clusters:
                emit("merge", stem, new_nosuf, sorted(c), len(clusters))

    return pd.DataFrame(rows).sort_values(["direction", "name_stem", "new_course_ids"], kind="stable")


# ---------------------------------------------------------------------------
# Table 1 -- degree lineage
# ---------------------------------------------------------------------------
def build_degree_lineage(catalog: pd.DataFrame, cii_module) -> tuple[pd.DataFrame, list[str], list[str], dict[str, str]]:
    dedup = catalog.drop_duplicates(["degree_id", "course_id"])
    degree_names = catalog.drop_duplicates("degree_id").set_index("degree_id")["degree_name_sl"].to_dict()
    degree_course_keys = dedup.groupby("degree_id")["name_key"].apply(lambda s: set(s.dropna()))
    degree_course_counts = dedup.groupby("degree_id")["course_id"].nunique()

    all_degrees = sorted(degree_names.keys())
    new_degrees = [d for d in all_degrees if int(d.split(".")[0]) >= 40]
    old_degrees = [d for d in all_degrees if int(d.split(".")[0]) < 40]

    rows: list[dict[str, Any]] = []
    for nd in new_degrees:
        new_name = degree_names[nd]
        new_keys = degree_course_keys.get(nd, set())
        new_count = int(degree_course_counts.get(nd, 0))
        has_2023 = bool(re.search(r"2023", new_name))
        new_family = degree_family(new_name)

        candidates = []
        for od in old_degrees:
            old_name = degree_names[od]
            old_keys = degree_course_keys.get(od, set())
            old_count = int(degree_course_counts.get(od, 0))
            shared = new_keys & old_keys
            union = new_keys | old_keys
            overlap_new = (len(shared) / len(new_keys)) if new_keys else 0.0
            overlap_old = (len(shared) / len(old_keys)) if old_keys else 0.0
            jaccard = (len(shared) / len(union)) if union else 0.0
            old_family = degree_family(old_name)
            sim = float(cii_module.name_similarity(new_family, old_family))
            candidates.append(
                {
                    "old_degree_id": od,
                    "old_degree_name": old_name,
                    "old_degree_course_count": old_count,
                    "shared_course_key_count": len(shared),
                    "overlap_pct_of_new": overlap_new,
                    "overlap_pct_of_old": overlap_old,
                    "jaccard": jaccard,
                    "degree_name_similarity": sim,
                    "same_family_after_strip": bool(new_family == old_family),
                    "courses_added": len(new_keys - old_keys),
                    "courses_removed": len(old_keys - new_keys),
                }
            )

        candidates.sort(key=lambda c: (-c["overlap_pct_of_new"], -c["jaccard"], c["old_degree_id"]))
        top3 = candidates[:3]
        for rank, c in enumerate(top3, start=1):
            if rank == 1:
                if c["overlap_pct_of_new"] >= 0.75 and (has_2023 or c["degree_name_similarity"] >= 0.60):
                    sugg = "STRONG"
                elif c["overlap_pct_of_new"] >= 0.50:
                    sugg = "PLAUSIBLE"
                elif c["overlap_pct_of_new"] >= 0.25:
                    sugg = "WEAK"
                else:
                    sugg = "NONE"
            else:
                sugg = "NONE"
            rows.append(
                {
                    "university_id": UNIVERSITY_ID,
                    "new_degree_id": nd,
                    "new_degree_name": new_name,
                    "new_degree_course_count": new_count,
                    "candidate_rank": rank,
                    "old_degree_id": c["old_degree_id"],
                    "old_degree_name": c["old_degree_name"],
                    "old_degree_course_count": c["old_degree_course_count"],
                    "shared_course_key_count": c["shared_course_key_count"],
                    "overlap_pct_of_new": c["overlap_pct_of_new"],
                    "overlap_pct_of_old": c["overlap_pct_of_old"],
                    "jaccard": c["jaccard"],
                    "degree_name_similarity": c["degree_name_similarity"],
                    "has_2023_suffix": has_2023,
                    "same_family_after_strip": c["same_family_after_strip"],
                    "courses_added": c["courses_added"],
                    "courses_removed": c["courses_removed"],
                    "auto_suggestion": sugg,
                    "review_decision": "",
                    "reviewer_name": "",
                    "review_date": "",
                    "review_notes": "",
                }
            )
    return pd.DataFrame(rows), new_degrees, old_degrees, degree_names


# ---------------------------------------------------------------------------
# Table 2 -- course link
# ---------------------------------------------------------------------------
def build_course_link(
    catalog: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    lineage_df: pd.DataFrame,
    split_merge_df: pd.DataFrame,
    old_degrees: list[str],
    degree_names: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # --- 182 never-in-TRAIN VALID course IDs (same definition as Phase 1) ---
    never = valid.loc[(valid["course_difficulty_missing"] == 1) & (valid["course_history_count"] == 0)].copy()
    new_course_ids = sorted(never["course_id"].unique().tolist())

    catalog_names = dict(zip(catalog.drop_duplicates("course_id")["course_id"], catalog.drop_duplicates("course_id")["course_name_sl"]))
    catalog_credits = dict(zip(catalog.drop_duplicates("course_id")["course_id"], catalog.drop_duplicates("course_id")["course_credits"].astype("float64")))
    catalog_name_key = dict(zip(catalog.drop_duplicates("course_id")["course_id"], catalog.drop_duplicates("course_id")["name_key"]))
    catalog_name_stem = dict(zip(catalog.drop_duplicates("course_id")["course_id"], catalog.drop_duplicates("course_id")["name_stem"]))

    course_degree_ids = catalog.groupby("course_id")["degree_id"].apply(lambda s: sorted(set(s.tolist())))
    course_families = catalog.groupby("course_id")["degree_id"].apply(
        lambda s: {degree_family(degree_names[d]) for d in s if d in degree_names}
    )

    modal_req = never.groupby("course_id")["requirement_type_id"].apply(modal)
    valid_row_counts = never.groupby("course_id").size()
    effective_from = never.groupby("course_id")["part_id"].min()

    # --- TRAIN raw stats per course_id (Level-2 style, unsmoothed) ---
    train = train.copy()
    train["mark_present"] = train["final_mark"].notna()
    train["pass_value"] = (train["final_mark"] >= 50) & train["mark_present"]
    train_agg = (
        train.groupby("course_id")
        .agg(
            train_support=("course_id", "size"),
            sum_pass=("pass_value", "sum"),
            sum_mark_present=("mark_present", "sum"),
            sum_mark=("final_mark", "sum"),
        )
        .reset_index()
    )
    train_agg["train_pass_rate"] = np.where(
        train_agg["sum_mark_present"] > 0, train_agg["sum_pass"] / train_agg["sum_mark_present"], np.nan
    )
    train_agg["train_avg_mark"] = np.where(
        train_agg["sum_mark_present"] > 0, train_agg["sum_mark"] / train_agg["sum_mark_present"], np.nan
    )
    train_support_map = dict(zip(train_agg["course_id"], train_agg["train_support"]))
    train_pass_map = dict(zip(train_agg["course_id"], train_agg["train_pass_rate"]))
    train_avg_map = dict(zip(train_agg["course_id"], train_agg["train_avg_mark"]))

    # --- old-generation catalog rows, name_key -> course_id set, per degree ---
    old_catalog = catalog.loc[catalog["degree_id"].isin(old_degrees)]
    old_key_to_course_by_degree: dict[str, dict[Any, set[str]]] = {}
    for od in old_degrees:
        sub = old_catalog.loc[old_catalog["degree_id"] == od]
        old_key_to_course_by_degree[od] = sub.groupby("name_key")["course_id"].apply(set).to_dict()

    old_key_to_course_global: dict[Any, set[str]] = old_catalog.groupby("name_key")["course_id"].apply(set).to_dict()

    # --- rank-1 lineage candidate per new degree (from Table 1) ---
    rank1 = lineage_df.loc[lineage_df["candidate_rank"] == 1].set_index("new_degree_id")

    def lineage_scope_for_degree(nd: str) -> tuple[set[str], str]:
        # Table 1 only carries a lineage row for degrees with numeric degree_id
        # >= 40 (the task's literal Table 1 rule). Several genuinely
        # VALID-only "new generation" degrees fall under that threshold
        # (e.g. 26.111-31.111, 33.111-37.111, 39.111 -- confirmed against
        # Phase 0 Q5's 25-degree VALID-only list) and therefore never get a
        # Table 1 row at all. The spec's own words ("If Table 1 gives no
        # candidate for the relevant degree, fall back to same-family
        # search") cover this case exactly as much as the zero-overlap case
        # -- "gives no candidate" is true whether the row is absent or
        # present-but-empty -- so both are routed through the same fallback
        # here. Reported in the markdown as a finding, not silently patched
        # in Table 1 itself.
        if nd in rank1.index:
            row = rank1.loc[nd]
            if int(row["shared_course_key_count"]) > 0:
                return {row["old_degree_id"]}, "rank1_lineage"
        fam = degree_family(degree_names.get(nd, ""))
        same_fam_olds = {od for od in old_degrees if degree_family(degree_names[od]) == fam}
        if same_fam_olds:
            return same_fam_olds, "same_family_fallback"
        return set(), "no_lineage_row"

    # ------------------------------------------------------------------
    # Table 0 exclusion maps: new_course_id -> [(old_course_id, direction)]
    # ------------------------------------------------------------------
    split_map: dict[str, list[str]] = {}
    merge_map: dict[str, list[str]] = {}
    for r in split_merge_df.itertuples(index=False):
        new_ids = [c for c in r.new_course_ids.split("|") if c]
        old_ids = [c for c in r.old_course_ids.split("|") if c]
        if r.direction == "split":
            for nc in new_ids:
                split_map.setdefault(nc, []).extend(old_ids)
        else:  # merge
            for nc in new_ids:
                merge_map.setdefault(nc, []).extend(old_ids)

    # ------------------------------------------------------------------
    # Per new-course predecessor search
    # ------------------------------------------------------------------
    link_rows: list[dict[str, Any]] = []
    for nc in new_course_ids:
        name_key = catalog_name_key.get(nc)
        name_stem_val = catalog_name_stem.get(nc)
        degree_ids = course_degree_ids.get(nc, [])
        family_count = len(course_families.get(nc, set()))

        base_common = {
            "university_id": UNIVERSITY_ID,
            "new_course_id": nc,
            "new_course_name": catalog_names.get(nc, ""),
            "new_course_name_key": name_key,
            "new_course_name_stem": name_stem_val,
            "new_course_credits": catalog_credits.get(nc),
            "new_course_requirement_type_id": int(modal_req.get(nc)) if nc in modal_req.index else None,
            "new_course_degree_ids": "|".join(degree_ids),
            "new_course_family_count": family_count,
            "new_course_valid_rows": int(valid_row_counts.get(nc, 0)),
            "effective_from_part_id": int(effective_from.get(nc)) if nc in effective_from.index else None,
            "approval_status": "pending",
            "approved_by": "",
            "approved_at": "",
        }

        # --- Step 1: split / merge exclusion ---
        if nc in split_map or nc in merge_map:
            is_split = nc in split_map
            old_ids = split_map.get(nc) if is_split else merge_map.get(nc)
            relationship = "split_from" if is_split else "merged_from"
            n = len(old_ids)
            for rank, cid in enumerate(sorted(old_ids), start=1):
                row = dict(base_common)
                row.update(
                    {
                        "new_course_scope": "split_or_merge",
                        "old_course_id": cid,
                        "old_course_name": catalog_names.get(cid, ""),
                        "old_course_name_key": catalog_name_key.get(cid, ""),
                        "old_course_degree_ids": "|".join(course_degree_ids.get(cid, [])),
                        "old_course_train_support": int(train_support_map.get(cid, 0)),
                        "old_course_train_pass_rate": train_pass_map.get(cid),
                        "old_course_train_avg_mark": train_avg_map.get(cid),
                        "predecessor_rank": rank,
                        "predecessor_count_for_new_course": n,
                        "relationship_type": relationship,
                        "weight_hint": None,
                        "match_method": "split_detection",
                        "lineage_scope_used": "",
                        "pass_rate_spread_across_predecessors": None,
                        "notes": f"Table 0 {'split' if is_split else 'merge'} candidate: name_stem={name_stem_val}",
                    }
                )
                link_rows.append(row)
            continue

        # --- Step 2/3/4: ordinary shared/specific search ---
        scope = "shared" if family_count >= 5 else "specific"

        candidate_course_ids: set[str] = set()
        scope_used: set[str] = set()
        fallback_notes: list[str] = []
        if scope == "shared":
            candidate_course_ids = set(old_key_to_course_global.get(name_key, set()))
            match_method_attempt = "name_key_global"
            scope_label = "ALL_OLD_DEGREES"
        else:
            for d in degree_ids:
                s, method = lineage_scope_for_degree(d)
                scope_used |= s
                if method == "same_family_fallback":
                    fallback_notes.append(f"{d}->same_family_fallback")
                elif method == "no_lineage_row":
                    fallback_notes.append(f"{d}->no_lineage_row")
            for od in scope_used:
                candidate_course_ids |= old_key_to_course_by_degree.get(od, {}).get(name_key, set())
            if scope_used:
                match_method_attempt = "name_key_family_fallback" if any(
                    n.endswith("same_family_fallback") for n in fallback_notes
                ) else "name_key_scoped"
            else:
                match_method_attempt = "none"
            scope_label = "|".join(sorted(scope_used)) if scope_used else ""

        candidate_course_ids.discard(nc)  # a course cannot be its own predecessor

        eligible = [
            (cid, int(train_support_map.get(cid, 0)))
            for cid in candidate_course_ids
            if train_support_map.get(cid, 0) >= MIN_TRAIN_SUPPORT
        ]
        eligible.sort(key=lambda t: (-t[1], t[0]))

        n = len(eligible)
        if n == 0:
            relationship = "none"
        elif n == 1:
            relationship = "successor"
        else:
            relationship = "consolidated_into"  # covers shared, and specific-scope edge case (choice 4)

        total_support = sum(sup for _, sup in eligible)
        pass_rates = [train_pass_map.get(cid) for cid, _ in eligible]
        pass_rates = [p for p in pass_rates if pd.notna(p)]
        spread = (max(pass_rates) - min(pass_rates)) if len(pass_rates) >= 2 else (0.0 if len(pass_rates) == 1 else None)

        base = dict(base_common)
        base.update(
            {
                "new_course_scope": scope,
                "predecessor_count_for_new_course": n,
                "relationship_type": relationship,
                "match_method": match_method_attempt if candidate_course_ids or match_method_attempt != "none" else "none",
                "lineage_scope_used": scope_label,
                "pass_rate_spread_across_predecessors": spread,
            }
        )

        if n == 0:
            row = dict(base)
            row.update(
                {
                    "old_course_id": "",
                    "old_course_name": "",
                    "old_course_name_key": "",
                    "old_course_degree_ids": "",
                    "old_course_train_support": "",
                    "old_course_train_pass_rate": "",
                    "old_course_train_avg_mark": "",
                    "predecessor_rank": "",
                    "weight_hint": "",
                    "notes": ("; ".join(fallback_notes) if fallback_notes else "") + (
                        " no candidate found" if not candidate_course_ids else " candidates found but none reach train_support>=20"
                    ),
                }
            )
            link_rows.append(row)
        else:
            for rank, (cid, sup) in enumerate(eligible, start=1):
                row = dict(base)
                notes_parts = list(fallback_notes)
                if relationship == "consolidated_into" and scope == "specific":
                    notes_parts.append("interpretive choice 4: specific-scope course with >=2 predecessors treated as consolidated")
                row.update(
                    {
                        "old_course_id": cid,
                        "old_course_name": catalog_names.get(cid, ""),
                        "old_course_name_key": catalog_name_key.get(cid, ""),
                        "old_course_degree_ids": "|".join(course_degree_ids.get(cid, [])),
                        "old_course_train_support": sup,
                        "old_course_train_pass_rate": train_pass_map.get(cid),
                        "old_course_train_avg_mark": train_avg_map.get(cid),
                        "predecessor_rank": rank,
                        "weight_hint": (sup / total_support) if relationship == "consolidated_into" else 1.0,
                        "notes": "; ".join(notes_parts),
                    }
                )
                link_rows.append(row)

    link_df = pd.DataFrame(link_rows)

    column_order = [
        "university_id", "new_course_id", "new_course_name", "new_course_name_key", "new_course_name_stem",
        "new_course_credits", "new_course_requirement_type_id", "new_course_degree_ids", "new_course_family_count",
        "new_course_scope", "new_course_valid_rows", "old_course_id", "old_course_name", "old_course_name_key",
        "old_course_degree_ids", "old_course_train_support", "old_course_train_pass_rate", "old_course_train_avg_mark",
        "predecessor_rank", "predecessor_count_for_new_course", "relationship_type", "weight_hint", "match_method",
        "lineage_scope_used", "effective_from_part_id", "pass_rate_spread_across_predecessors", "approval_status",
        "approved_by", "approved_at", "notes",
    ]
    link_df = link_df[column_order]

    # --- weight-sum assertion (per consolidated_into group; split/merge weights are null by design) ---
    consolidated = link_df.loc[link_df["relationship_type"] == "consolidated_into"]
    if not consolidated.empty:
        sums = consolidated.groupby("new_course_id")["weight_hint"].sum()
        bad = sums.loc[(sums - 1.0).abs() > 1e-6]
        if len(bad):
            raise SystemExit(
                f"STOP: weight_hint does not sum to 1.0 within 1e-6 for {len(bad)} consolidated_into group(s): "
                f"{bad.to_dict()}"
            )

    meta = {
        "new_course_ids": new_course_ids,
        "never_frame_rows": int(len(never)),
        "train_support_map": train_support_map,
    }
    return link_df, meta


# ---------------------------------------------------------------------------
# Table 3 -- difficulty stats prototype
# ---------------------------------------------------------------------------
def build_stats_prototype(train: pd.DataFrame, link_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], set[str], list[str]]:
    train_course_ids = sorted(train["course_id"].unique().tolist())
    predecessor_courses = set(link_df.loc[link_df["old_course_id"] != "", "old_course_id"].unique().tolist())
    predecessor_courses &= set(train_course_ids)

    other_pool = sorted(set(train_course_ids) - predecessor_courses)
    n_sample = min(SAMPLE_OTHER_COURSES, len(other_pool))
    sample_other = (
        pd.Series(other_pool).sample(n=n_sample, random_state=RANDOM_SEED).tolist() if n_sample else []
    )
    scope_courses = sorted(predecessor_courses | set(sample_other))

    semesters = sorted(train["part_id"].unique().tolist())

    work = train.loc[train["course_id"].isin(scope_courses)].copy()
    work["mark_present"] = work["final_mark"].notna()
    work["pass_value"] = ((work["final_mark"] >= 50) & work["mark_present"]).astype("int64")
    work["mark_value"] = work["final_mark"].fillna(0.0).astype("float64")
    work["retake_present"] = work["attempt_number"].notna()
    work["retake_value"] = ((work["attempt_number"] > 1) & work["retake_present"]).astype("int64")

    def cumulative_before(group_cols: list[str]) -> pd.DataFrame:
        agg = (
            work.groupby(group_cols + ["part_id"])
            .agg(
                support=("mark_present", "sum"),
                sum_pass=("pass_value", "sum"),
                sum_mark=("mark_value", "sum"),
                retake_support=("retake_present", "sum"),
                sum_retake=("retake_value", "sum"),
            )
            .reset_index()
        )
        stat_frames = {}
        for stat in ("support", "sum_pass", "sum_mark", "retake_support", "sum_retake"):
            piv = agg.pivot_table(index=group_cols, columns="part_id", values=stat, fill_value=0)
            piv = piv.reindex(columns=semesters, fill_value=0)
            piv.columns.name = "as_of_part_id"
            cum = piv.cumsum(axis=1)
            before = cum.shift(1, axis=1, fill_value=0)
            stat_frames[stat] = before.stack()
        combined = pd.concat(stat_frames.values(), axis=1)
        combined.columns = list(stat_frames.keys())
        combined = combined.reset_index()
        return combined

    lvl1_groups = work[["course_id", "degree_id"]].drop_duplicates()
    lvl1 = cumulative_before(["course_id", "degree_id"])
    lvl1 = lvl1.merge(lvl1_groups, on=["course_id", "degree_id"], how="inner")
    lvl1["source_level"] = 1

    lvl2 = cumulative_before(["course_id"])
    lvl2["degree_id"] = "ALL"
    lvl2["source_level"] = 2

    stats = pd.concat([lvl1, lvl2], ignore_index=True, sort=False)
    stats["pass_rate"] = np.where(stats["support"] > 0, stats["sum_pass"] / stats["support"], np.nan)
    stats["avg_mark"] = np.where(stats["support"] > 0, stats["sum_mark"] / stats["support"], np.nan)
    stats["retake_rate"] = np.where(
        stats["retake_support"] > 0, stats["sum_retake"] / stats["retake_support"], np.nan
    )
    stats["university_id"] = UNIVERSITY_ID
    stats["support_count"] = stats["support"].astype("int64")
    stats["link_used"] = None
    stats["link_weight"] = None

    stats = stats[
        [
            "university_id",
            "course_id",
            "degree_id",
            "as_of_part_id",
            "support_count",
            "pass_rate",
            "avg_mark",
            "retake_rate",
            "source_level",
            "link_used",
            "link_weight",
        ]
    ].sort_values(["source_level", "course_id", "degree_id", "as_of_part_id"], kind="stable")

    return stats, scope_courses, predecessor_courses, sample_other


def validate_course_history_count(train_full: pd.DataFrame, valid_full: pd.DataFrame) -> dict[str, Any]:
    """Reuse src.course_difficulty (read-only) to reproduce Phase 0's mismatch check."""
    from src.course_difficulty import fit_difficulty_state, apply_difficulty_state

    state = fit_difficulty_state(train_full)
    enriched = apply_difficulty_state(valid_full, state, include_source=False)
    recomputed = enriched["course_history_count"].to_numpy()
    frozen = valid_full["course_history_count"].to_numpy()
    mismatches = int((recomputed != frozen).sum())
    return {"rows_checked": int(len(valid_full)), "mismatches": mismatches}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    pairs_raw = pd.read_csv(PAIRS_PATH, dtype="string", keep_default_na=False)
    pairs = pairs_raw[["new_course_id", "new_course_name", "old_course_id", "old_course_name", "new_valid_row_count"]].copy()

    q1_frame, gate_result = validate_normalization_gate(pairs)
    if not gate_result["gate_pass"]:
        print(json.dumps({"STOP": "normalization gate failed", "gate_result": gate_result}, ensure_ascii=False, indent=2, default=str))
        return 1

    catalog = load_catalog()
    cii_module = load_course_identity_investigation()
    train = load_train()
    valid = load_valid()

    # --- Table 0 ---
    split_merge_df = build_split_merge_candidates(catalog, train, valid)
    OUT_SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    split_merge_df.to_csv(OUT_SPLIT_CSV, index=False, encoding="utf-8-sig")

    bunyan = split_merge_df.loc[
        (split_merge_df["direction"] == "split")
        & (split_merge_df["old_course_ids"] == "510.111")
        & (split_merge_df["new_course_ids"] == "1183.111|1192.111")
    ]
    if bunyan.empty or float(bunyan["credit_change"].iloc[0]) != 3.0:
        print(
            json.dumps(
                {
                    "STOP": "بنيان الحواسيب split not detected as specified",
                    "found_rows_for_stem": split_merge_df.loc[
                        split_merge_df["name_stem"].str.contains("حواسيب", na=False)
                    ].to_dict(orient="records"),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 1

    # --- Table 1 ---
    lineage_df, new_degrees, old_degrees, degree_names = build_degree_lineage(catalog, cii_module)
    OUT_LINEAGE_CSV.parent.mkdir(parents=True, exist_ok=True)
    lineage_df.to_csv(OUT_LINEAGE_CSV, index=False, encoding="utf-8-sig")

    # --- Table 2 ---
    link_df, link_meta = build_course_link(catalog, train, valid, lineage_df, split_merge_df, old_degrees, degree_names)
    link_df.to_csv(OUT_LINK_CSV, index=False, encoding="utf-8-sig")

    # --- Table 2 validations ---
    val_67 = []
    for r in pairs.itertuples(index=False):
        sub = link_df.loc[(link_df["new_course_id"] == r.new_course_id) & (link_df["old_course_id"] == r.old_course_id)]
        val_67.append(
            {
                "new_course_id": r.new_course_id,
                "old_course_id": r.old_course_id,
                "present": bool(len(sub)),
                "relationship_type": (sub["relationship_type"].iloc[0] if len(sub) else None),
            }
        )
    expected_split_pair = next(
        v for v in val_67 if v["new_course_id"] == EXPECTED_GATE_NONMATCH_NEW and v["old_course_id"] == EXPECTED_GATE_NONMATCH_OLD
    )
    other_missing_67 = [
        v for v in val_67
        if not v["present"] and not (v["new_course_id"] == EXPECTED_GATE_NONMATCH_NEW and v["old_course_id"] == EXPECTED_GATE_NONMATCH_OLD)
    ]

    row_1423 = link_df.loc[link_df["new_course_id"] == "1423.111"]
    expected_five = ["151.111", "221.111", "391.111", "830.111", "893.111"]
    train_support_map = link_meta["train_support_map"]
    five_check = {
        cid: {
            "train_support": int(train_support_map.get(cid, 0)),
            "qualifies": bool(train_support_map.get(cid, 0) >= MIN_TRAIN_SUPPORT),
            "present_in_1423_predecessors": bool((row_1423["old_course_id"] == cid).any()),
            "weight_hint": (
                float(row_1423.loc[row_1423["old_course_id"] == cid, "weight_hint"].iloc[0])
                if (row_1423["old_course_id"] == cid).any()
                else None
            ),
        }
        for cid in expected_five
    }

    every_course_present = set(link_meta["new_course_ids"]) <= set(link_df["new_course_id"].unique().tolist())

    consolidated = link_df.loc[link_df["relationship_type"] == "consolidated_into"]
    weight_sums = consolidated.groupby("new_course_id")["weight_hint"].sum() if len(consolidated) else pd.Series(dtype="float64")
    weight_ok = bool((weight_sums - 1.0).abs().max() <= 1e-6) if len(weight_sums) else True

    table2_validation = {
        "expected_split_pair_present_as_split_from": (
            expected_split_pair["present"] and expected_split_pair["relationship_type"] == "split_from"
        ),
        "other_66_all_present": len(other_missing_67) == 0,
        "other_missing": other_missing_67,
        "row_1423_relationship": (row_1423["relationship_type"].iloc[0] if len(row_1423) else None),
        "row_1423_scope": (row_1423["new_course_scope"].iloc[0] if len(row_1423) else None),
        "five_predecessors_check": five_check,
        "every_new_course_present": every_course_present,
        "weight_sums_ok": weight_ok,
        "n_consolidated_groups": int(len(weight_sums)),
    }

    # --- Table 3 ---
    stats_df, scope_courses, predecessor_courses, sample_other = build_stats_prototype(train, link_df)
    stats_df.to_csv(OUT_STATS_CSV, index=False, encoding="utf-8-sig")

    hist_check = validate_course_history_count(train, valid)

    # ------------------------------------------------------------------
    # Census / summary numbers for the report
    # ------------------------------------------------------------------
    rank1 = lineage_df.loc[lineage_df["candidate_rank"] == 1]
    auto_counts = rank1["auto_suggestion"].value_counts().to_dict()
    non_strong = rank1.loc[rank1["auto_suggestion"] != "STRONG"][
        [
            "new_degree_id", "new_degree_name", "old_degree_id", "old_degree_name",
            "overlap_pct_of_new", "overlap_pct_of_old", "jaccard",
            "degree_name_similarity", "has_2023_suffix", "same_family_after_strip", "auto_suggestion",
        ]
    ].to_dict(orient="records")

    scope_counts = link_df.drop_duplicates("new_course_id")["new_course_scope"].value_counts().to_dict()
    rel_counts = link_df.drop_duplicates("new_course_id")["relationship_type"].value_counts().to_dict()
    per_course = link_df.drop_duplicates("new_course_id").set_index("new_course_id")
    never_valid_rows = pd.read_parquet(VALID_PATH, columns=["course_id", "course_difficulty_missing", "course_history_count"])
    never_valid_rows = never_valid_rows.loc[
        (never_valid_rows["course_difficulty_missing"] == 1) & (never_valid_rows["course_history_count"] == 0)
    ]
    rows_per_course = never_valid_rows.groupby("course_id").size()
    rel_rows_covered = {}
    for rel in per_course["relationship_type"].unique():
        cids = per_course.loc[per_course["relationship_type"] == rel].index
        rel_rows_covered[rel] = int(rows_per_course.reindex(cids).fillna(0).sum())

    payload = {
        "normalization_gate": gate_result,
        "table0_summary": {
            "split_candidates": int((split_merge_df["direction"] == "split").sum()),
            "merge_candidates": int((split_merge_df["direction"] == "merge").sum()),
            "rows": split_merge_df.to_dict(orient="records"),
        },
        "table1_summary": {
            "new_degrees": len(new_degrees),
            "old_degrees": len(old_degrees),
            "auto_suggestion_counts_rank1": auto_counts,
            "non_strong_rank1_cases": non_strong,
        },
        "table2_summary": {
            "new_courses_total": len(link_meta["new_course_ids"]),
            "scope_counts": scope_counts,
            "relationship_counts": rel_counts,
            "valid_rows_covered_by_relationship": rel_rows_covered,
            "never_in_train_rows_total": int(link_meta["never_frame_rows"]),
            "validation": table2_validation,
        },
        "table3_summary": {
            "row_count": int(len(stats_df)),
            "scope_course_count": len(scope_courses),
            "predecessor_course_count": len(predecessor_courses),
            "sampled_other_course_count": len(sample_other),
            "course_history_count_validation": hist_check,
        },
    }
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print("TEST reads: 0; models loaded/trained/rescored: 0; datasets under data/model_data written: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
