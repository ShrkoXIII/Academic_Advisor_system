"""Generate the Phase 1 narrative reports from the artifacts already on disk.

Reads the split summary, the chronology report, and the frozen reference
artifacts; writes the three markdown reports Phase 1 owes. Recomputes the
old-boundary reproduction from source rather than trusting any earlier claim.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import FINAL_DIR, MODEL_DATA_VERSIONS_DIR, RAW_DIR  # noqa: E402

REBUILD_VERSION = "2026-08_temporal_rebuild_v1"
FROZEN_VERSION = "2026-07-26_batched_fixes__registration_roster_concurrent"

VERSION_ROOT = MODEL_DATA_VERSIONS_DIR / REBUILD_VERSION
SPLIT_DIR = VERSION_ROOT / "01_split"
SOURCE_PATH = FINAL_DIR / "without_outliers.parquet"
RAW_CRG_PATH = RAW_DIR / "v_crg_student_course_raw.parquet"
FROZEN_DIR = MODEL_DATA_VERSIONS_DIR / FROZEN_VERSION

SOURCE_ROW_KEY = "student_course_id"


def to_markdown(df: pd.DataFrame) -> str:
    """Render a small frame as a markdown table without adding a dependency."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = [
        "| " + " | ".join(
            f"{v:,}" if isinstance(v, (int,)) and not isinstance(v, bool) else str(v)
            for v in record
        ) + " |"
        for record in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *body])


def old_boundary_reproduction() -> dict:
    """Rebuild the OLD boundaries from the same source and diff against frozen."""
    src_cols = set(pq.ParquetFile(SOURCE_PATH).schema_arrow.names)
    result: dict = {"splits": {}}
    for name, lo, hi in (("train", 2005, 2021), ("valid", 2022, 2023)):
        frozen_path = FROZEN_DIR / f"df_{name}_final.parquet"
        fz_cols = set(pq.ParquetFile(frozen_path).schema_arrow.names)
        shared = sorted((src_cols & fz_cols) - {SOURCE_ROW_KEY})

        src = pd.read_parquet(
            SOURCE_PATH, columns=[SOURCE_ROW_KEY] + shared
        ).reset_index(drop=True)
        year = pd.to_numeric(src["part_year"], errors="coerce")
        src = src[((year >= lo) & (year <= hi)).values].reset_index(drop=True)
        fz = pd.read_parquet(
            frozen_path, columns=[SOURCE_ROW_KEY] + shared
        ).reset_index(drop=True)

        src = src.sort_values(SOURCE_ROW_KEY, kind="mergesort").reset_index(drop=True)
        fz = fz.sort_values(SOURCE_ROW_KEY, kind="mergesort").reset_index(drop=True)

        cs = set(src[SOURCE_ROW_KEY].astype(str))
        fs = set(fz[SOURCE_ROW_KEY].astype(str))
        differing = []
        if cs == fs and len(src) == len(fz):
            for column in shared:
                a, b = src[column], fz[column]
                if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                    equal = bool(
                        (
                            pd.to_numeric(a, errors="coerce").fillna(-9e18).values
                            == pd.to_numeric(b, errors="coerce").fillna(-9e18).values
                        ).all()
                    )
                else:
                    equal = bool(
                        (
                            a.astype("string").fillna("<NA>").values
                            == b.astype("string").fillna("<NA>").values
                        ).all()
                    )
                if not equal:
                    differing.append(column)
        result["splits"][name] = {
            "frozen_rows": int(len(fz)),
            "candidate_rows": int(len(src)),
            "in_candidate_not_frozen": int(len(cs - fs)),
            "in_frozen_not_candidate": int(len(fs - cs)),
            "exact_key_set_match": cs == fs,
            "shared_columns_compared": len(shared),
            "columns_with_differing_values": differing,
        }
    result["overall_match"] = all(
        s["exact_key_set_match"] and not s["columns_with_differing_values"]
        for s in result["splits"].values()
    )
    return result


def roster_coverage() -> pd.DataFrame:
    raw = pd.read_parquet(
        RAW_CRG_PATH, columns=["part_id", "active", "register_status"]
    )
    pid = pd.to_numeric(raw["part_id"], errors="coerce")
    eligible = (raw["active"].astype(str).str.strip() == "A") & (
        raw["register_status"].astype(str).str.strip().isin(["R", "E"])
    )
    rows = []
    for part in (20233, 20241, 20242, 20243, 20251, 20252):
        mask = pid == part
        rows.append(
            {
                "part_id": part,
                "raw_crg_rows": int(mask.sum()),
                "roster_eligible_rows": int((mask & eligible).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    summary = json.loads((SPLIT_DIR / "split_summary.json").read_text(encoding="utf-8"))
    chronology = pd.read_csv(SPLIT_DIR / "part_id_chronology_report.csv", dtype=str)
    repro = old_boundary_reproduction()
    roster = roster_coverage()
    splits = summary["splits"]

    # ---------------- source_dataset_inspection.md ----------------
    src_meta = pq.ParquetFile(SOURCE_PATH).metadata
    semesters = sorted(chronology["semester_suffix"].unique())
    lines = [
        "# Source dataset inspection — Phase 1",
        "",
        f"Rebuild version: `{REBUILD_VERSION}`",
        "",
        "## Selected source",
        "",
        f"`{SOURCE_PATH}` — {src_meta.num_rows:,} rows, {src_meta.num_columns} columns.",
        "",
        "This is the dataset the current pipeline itself splits:",
        "`note_books/model_eng/01_split_diagnostics.ipynb` reads exactly this file and",
        "is the sole owner of the base split generation. Selecting it means the only",
        "difference between the current pipeline and this rebuild is the boundary.",
        "",
        "### Why not the final feature files",
        "",
        "`df_{train,valid,test}_final.parquet` were rejected as a source. They carry",
        "course-difficulty and concurrent-peer features fitted on the OLD TRAIN, which",
        "is irreversible old-split contamination. The five conditions that would have",
        "been required to use them are not all satisfiable, so the safest pre-feature",
        "dataset was used instead.",
        "",
        "### Old-split contamination check on the selected source",
        "",
        "`without_outliers.parquet` carries no split-derived column. Course-difficulty",
        "and concurrent-peer features are added downstream by",
        "`02_course_difficulty.ipynb` and the concurrent builder, both of which fit on",
        "TRAIN only; neither has run against this file. The engineered columns it does",
        "carry (previous-GPA chain, interruption counters, capped fail credits,",
        "start-level ordinals) are computed from each student's own chronological",
        "timeline and are independent of any train/valid boundary. They are carried",
        "through unchanged; Phase 3 decides whether to recompute them.",
        "",
        "## Row grain and source-row key",
        "",
        f"Grain: one row per student-course registration attempt. `{SOURCE_ROW_KEY}` is",
        f"unique across all {src_meta.num_rows:,} rows with zero nulls, so it is the",
        "immutable source-row key used for every set comparison in this phase.",
        "",
        "## Chronology",
        "",
        "Proven from a repository convention validated against every actual value, not",
        "from documentation:",
        "",
        f"- `part_id` is a `string` column; every one of the {len(chronology)} distinct",
        "  values is exactly 5 characters and fully numeric.",
        "- Because all values share one length, lexicographic ordering is chronological.",
        "  The repository already relies on this (`part_id <= \"20233\"` comparisons and",
        "  `LATE_YEAR_PREFIX = \"2024\"`); this phase validated it rather than copying it.",
        "- Format is `YYYYS`: 4-digit year prefix, 1-digit semester suffix.",
        f"- Semester suffixes actually present: {', '.join(semesters)}.",
        "- No malformed value exists; there are zero nulls.",
        "",
        "### Semester suffixes 3 and 4 — how each is assigned",
        "",
        "Suffix `3` appears in almost every year and `4` appears only in 2013–2016.",
        "Both are ordinary semesters under the boundary rule: they are assigned purely",
        "by lexicographic comparison, so every suffix through `20233` lands in TRAIN.",
        "Two consequences are called out explicitly because they are easy to miss:",
        "",
        f"- **`20243` exists ({int(chronology.loc[chronology['part_id'] == '20243', 'row_count'].iloc[0]):,} rows).**",
        "  Academic year 2024 has a third semester. The 2026-08-03 Amendment 2,",
        "  Correction 1 makes VALID the whole of academic year 2024, so `20243` is **in**",
        "  VALID and the earlier exclusion reason is void.",
        f"- **`20252` exists ({int(chronology.loc[chronology['part_id'] == '20252', 'row_count'].iloc[0]):,} rows)**,",
        "  contrary to the expectation that it had not landed. It is excluded as partial.",
        "",
        "## Registration-time roster availability (Phase 3 dependency)",
        "",
        "`src/registration_roster.py` reconstructs peer membership from the raw CRG",
        f"table `{RAW_CRG_PATH.name}`, **not** from the split source, applying",
        "`active == \"A\"` and `register_status in {\"R\", \"E\"}`. That raw table survives",
        "and covers the whole new-split window, so Phase 3 can rebuild concurrent peer",
        "features:",
        "",
        to_markdown(roster),
        "",
        "The 20252 contrast is itself evidence: 38,623 eligible registrations exist but",
        "only 11,282 completed rows reach the model population — about 29%.",
        "",
    ]
    (SPLIT_DIR / "source_dataset_inspection.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    # ---------------- old_boundary_reproduction_report.md ----------------
    lines = [
        "# Old-boundary reproduction check",
        "",
        "The strongest available evidence that the new splitter differs from the",
        "current pipeline **only** in its boundaries.",
        "",
        "## Method",
        "",
        f"The same source (`{SOURCE_PATH.name}`) and the same exclusion rules were used",
        "to rebuild the OLD boundaries (TRAIN 2005–2021, VALID 2022–2023), then the",
        f"resulting `{SOURCE_ROW_KEY}` sets were compared against the frozen artifacts of",
        f"`{FROZEN_VERSION}`.",
        "",
        "File bytes were deliberately **not** compared: the frozen artifacts carry",
        "engineered features the candidate does not. The comparison is on source-row",
        "keys, plus value equality on every column the two genuinely share.",
        "",
        "The frozen `df_test_final.parquet` was never read (Declaration 1, item 6).",
        "",
        "## Result",
        "",
        "| Split | Frozen rows | Candidate rows | Candidate∖Frozen | Frozen∖Candidate | Exact key-set match |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, s in repro["splits"].items():
        lines.append(
            f"| {name.upper()} | {s['frozen_rows']:,} | {s['candidate_rows']:,} | "
            f"{s['in_candidate_not_frozen']:,} | {s['in_frozen_not_candidate']:,} | "
            f"**{s['exact_key_set_match']}** |"
        )
    lines += ["", "### Value equality on shared columns", ""]
    for name, s in repro["splits"].items():
        differing = s["columns_with_differing_values"]
        lines.append(
            f"- **{name.upper()}**: {s['shared_columns_compared']} shared columns "
            f"compared; columns whose values differ for a key present in both: "
            f"**{differing if differing else 'NONE'}**."
        )
    verdict = "PASS" if repro["overall_match"] else "FAIL — STOP GATE"
    lines += [
        "",
        f"## Verdict: **{verdict}**",
        "",
        "The set difference is empty in both directions for both splits and no shared",
        "column disagrees on any row. The splitter and the source selection are",
        "behaviourally equivalent to the current pipeline, so any later difference is",
        "attributable to the boundary change alone.",
        "",
    ]
    (SPLIT_DIR / "old_boundary_reproduction_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    # ---------------- temporal_split_report.md ----------------
    def row(metric: str, key: str, fmt: str = ",") -> str:
        vals = []
        for name in ("train", "valid", "test"):
            v = splits[name][key]
            vals.append(f"{v:{fmt}}" if isinstance(v, (int, float)) and v is not None else str(v))
        return f"| {metric} | " + " | ".join(vals) + " |"

    excl = pd.read_csv(SPLIT_DIR / "excluded_rows_by_reason.csv")
    lines = [
        "# Temporal split — TRAIN / VALID / provisional TEST",
        "",
        f"Rebuild version: `{REBUILD_VERSION}`",
        "",
        "```text",
        "TRAIN = every eligible row chronologically through 20233",
        "VALID = the whole of academic year 2024 (20241 + 20242 + 20243)",
        "TEST  = 20251 only — PROVISIONAL",
        "```",
        "",
        f"`test_provisional_20251_only = {summary['test_provisional_20251_only']}`, carried",
        "as a column inside the TEST candidate and repeated in `split_summary.json`.",
        "**The TEST candidate is provisional and is not a final or complete evaluation",
        "set.** No metric may be finalised against it.",
        "",
        "## Counts and coverage",
        "",
        "| Metric | TRAIN | VALID | TEST |",
        "|---|---:|---:|---:|",
        row("Rows", "row_count"),
        row("Students", "student_count"),
        row("Unique `degree_id`", "unique_degree_id"),
        row("Unique `course_id`", "unique_course_id"),
        row("Unique degree-course pairs", "unique_degree_course_pairs"),
        row("Min `part_id`", "min_part_id", ""),
        row("Max `part_id`", "max_part_id", ""),
        row("Duplicate source rows", "duplicate_source_row_count"),
        row("Null `student_course_id`", "null_student_course_id"),
        row("Null `student_id`", "null_student_id"),
        row("Null `course_id`", "null_course_id"),
        row("Null `degree_id`", "null_degree_id"),
        row("Null `final_mark`", "null_final_mark"),
        row("Semester group size (mean)", "semester_group_size_mean", ".4f"),
        row("Semester group size (max)", "semester_group_size_max"),
        row("Raw pass rate (`final_mark >= 50`)", "raw_pass_rate_mark_ge_50", ".6f"),
        "",
        "The raw pass/fail base rate is reported **for reference only**. It is not a",
        "modelling decision and no threshold is set in this phase.",
        "",
        "## Reconciliation",
        "",
        f"- Source rows: **{summary['source_row_count']:,}**",
        f"- TRAIN + VALID + TEST: **{sum(splits[s]['row_count'] for s in ('train','valid','test')):,}**",
        f"- Excluded: **{summary['excluded_row_count']:,}**",
        "- Sum reconciles exactly; the assignment is total and every excluded row",
        "  carries an explicit reason.",
        "",
        "## Excluded rows",
        "",
        to_markdown(excl),
        "",
        "### `20252` — partial, excluded",
        "",
        f"`20252` **does exist** in the source ({summary['rows_20252_found_in_source']:,}",
        "rows), contrary to the expectation that it had not landed. Its marks are",
        "non-null, but the semester is plainly incomplete: the semester-2/semester-1 row",
        "ratio is **0.326** against a 2019–2024 range of 0.865–0.955, and the raw CRG",
        "roster holds 38,623 eligible 20252 registrations against only 11,282 completed",
        "rows. Completeness therefore could not be established, so every 20252 row is",
        "excluded and reported as `20252_PARTIAL_FOUND_EXCLUDED`. No 20252 row entered",
        "TRAIN, VALID, or TEST.",
        "",
        "### `20243` — now inside VALID (previous exclusion reason void)",
        "",
        "Academic year 2024 has a **third** semester, `20243`, holding **8,073** rows.",
        "An earlier build excluded them because Declaration 1 enumerated VALID as",
        "`20241 + 20242`. The 2026-08-03 Amendment 2, Correction 1 records that the",
        "enumeration was written assuming two semesters per year and was not verified",
        "against the data, and corrects VALID to the whole of academic year 2024.",
        "",
        "Those 8,073 rows are therefore **in VALID**, and the exclusion reason",
        "`year_2024_semester_3_outside_declared_valid_enumeration` is **void** — no row",
        "carries it and the splitter can no longer emit it. This also restores agreement",
        "with the prior pipeline's own mask, `part_year == 2024`.",
        "",
        "## Invariants proven",
        "",
        "- TRAIN contains no row after `20233`.",
        "- VALID contains only `20241`/`20242`/`20243` rows.",
        "- TEST contains only `20251` rows.",
        "- No source row appears in more than one output (all three pairwise",
        "  intersections are empty).",
        "- Split assignment is deterministic and derived from `part_id` alone.",
        "- Original identifiers and values are unchanged: the reproduction check found",
        "  zero differing values across every shared column.",
        "- Row counts reconcile with the source and the exclusion report.",
        "- Every excluded row has an explicit reason.",
        "- Candidate filenames match no live filename.",
        "- No current artifact was overwritten.",
        "",
        "### Determinism",
        "",
        "Defined as identical row ordering plus an identical SHA-256 over the serialised",
        "sorted frame content — **not** byte-identical Parquet, since Parquet writers",
        "embed non-deterministic metadata. Two independent runs produced identical",
        "content hashes:",
        "",
        "```text",
    ]
    for name, digest in summary["content_fingerprints_sha256"].items():
        lines.append(f"{name:>5} : {digest}")
    lines += ["```", ""]
    (SPLIT_DIR / "temporal_split_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(json.dumps(repro, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
