---
title: "Pipeline map — how to open and read it"
tags:
  - academic-advisor
  - pipeline
date: 2026-08-06
status: snapshot
---

# Pipeline map — how to open and read it

Five stage documents that trace the checkable route from Oracle extraction to `models/runs/leaderboard.csv`, one stage per file, each with its own mermaid diagram.

## Opening this in Obsidian

**The vault root is the repository root** — `D:\AI\Real projects\Academic_Advisor`, not `docs/` and not `docs/map/`. Open Obsidian, choose **Open folder as vault**, and pick the repository root. Opening `docs/` or `docs/map/` as the vault will break the cross-links to the manifests, which live outside this folder.

The mermaid diagrams render natively in Obsidian's reading view and in live preview. Nothing needs to be installed.

## Read in numbered order

| File | Covers |
|---|---|
| `01-extract-and-clean.md` | Oracle → `data/raw/*` → the five table cleaners → `data/preprocessed/<table>/clean_*` |
| `02-merge-and-features.md` | clean tables → both merges → population select → feature engineering → the frozen `without_outliers.parquet` handoff |
| `03-rebuild-2026-08.md` | `without_outliers` → preflight → split → lineage gate → diploma map → GPA/B2 difficulty → concurrent → `05_dataset` |
| `04-training-and-runs.md` | `05_dataset` → `src/model_training.py` → `src/experiment_tracking.py` → `models/runs/*` and `leaderboard.csv` |
| `05-off-route.md` | what is **not** on the route: report-only tools, closed historical branches, files with no confirmed caller, and the broken maintained-notebook route |

Documents 01 through 04 are the route, in order. Document 05 is the complement — read it before concluding that a file you found is live, because a filename is not evidence of reachability.

## Graph view settings

The graph view is only useful here with two adjustments:

1. **Filters → turn off "Unresolved links".** Without this, every file path mentioned in prose would render as a phantom node and the graph collapses into a hub with dozens of spokes and no edges between them. These documents deliberately use code formatting rather than wikilinks for file references, so leaving this on shows nothing useful.
2. **Filters → path filter `path:docs/map`.** Scopes the graph to the five stage documents and this README, so you see the stage chain rather than the whole vault.

With both applied, the graph is a simple chain: `01 → 02 → 03 → 04 → 05`, which is the point.

## Which document answers which question

- *What runs, in what order, and what does it write?* — the five stage documents here.
- *What does one specific file in `src/` or `scripts/` do, and is it live?* — `docs/manifests/codebase_map_2026-08.md`, one row per file with its class, evidence, reads, writes, and last commit. Per-file detail is **not** duplicated here.
- *Which training runs exist, grouped by what question they answer?* — `docs/manifests/models_runs_index_2026-08.md`.
- *What is the history of this map?* — `docs/manifests/project_pipeline_routes_2026-08.md`, the single-page predecessor these five documents replace.

## Provenance

Every claim in these documents is anchored to something checkable: a cell index in a notebook's JSON, a line number in a script, a field in a report artifact on disk, or a git commit. Where the census found no evidence either way, the document says so and names what was checked. Filenames, docstrings, and comments were not treated as evidence.

Nothing in this folder was produced by running the pipeline. No model was trained, no rebuild executed, and no TEST artifact read.
