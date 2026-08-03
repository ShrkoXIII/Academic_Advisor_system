"""Guard against two tracked paths that differ only by letter case.

Windows and macOS resolve such pairs to the SAME file, so committing both means
whichever is written last silently overwrites the other. That is not
hypothetical here: `c6a9656` destroyed the course-identity investigation's
candidate CSV by adding `models/runs/COURSE_IDENTITY_CANDIDATES.csv` alongside
the existing `models/runs/course_identity_candidates.csv`.
"""

from collections import defaultdict
from pathlib import Path
import subprocess
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrackedPathCaseCollisionTests(unittest.TestCase):
    def test_no_tracked_paths_differ_only_by_case(self) -> None:
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.split("\0")

        by_lowercase: dict[str, list[str]] = defaultdict(list)
        for path in tracked:
            if path:
                by_lowercase[path.lower()].append(path)

        collisions = {
            lowered: sorted(paths)
            for lowered, paths in by_lowercase.items()
            if len(paths) > 1
        }
        self.assertEqual(
            collisions,
            {},
            "tracked paths differ only by case and collide on case-insensitive "
            f"filesystems: {collisions}",
        )


if __name__ == "__main__":
    unittest.main()
