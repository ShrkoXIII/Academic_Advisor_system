from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CanonicalDtype = Literal["string", "Int64"]
IdSemantic = Literal[
    "entity_id",
    "semester_code",
    "categorical_code",
]


@dataclass(frozen=True)
class IdColumnRule:
    dtype: CanonicalDtype
    semantic: IdSemantic
    nullable: bool = True


# ============================================================
# Global canonical ID registry
# ============================================================

ID_COLUMN_RULES: dict[str, IdColumnRule] = {
    # -------------------------
    # Entity / join IDs
    # -------------------------
    "student_id": IdColumnRule(
        dtype="string",
        semantic="entity_id",
    ),
    "course_id": IdColumnRule(
        dtype="string",
        semantic="entity_id",
    ),
    "degree_id": IdColumnRule(
        dtype="string",
        semantic="entity_id",
    ),
    "degree_course_id": IdColumnRule(
        dtype="string",
        semantic="entity_id",
    ),
    "student_course_id": IdColumnRule(
        dtype="string",
        semantic="entity_id",
    ),
    "student_status_id": IdColumnRule(
        dtype="string",
        semantic="entity_id",
    ),
    "faculty_id": IdColumnRule(
        dtype="string",
        semantic="entity_id",
    ),
    "grade_id": IdColumnRule(
        dtype="string",
        semantic="entity_id",
    ),
    "grade_version_id": IdColumnRule(
        dtype="string",
        semantic="entity_id",
    ),
    "req_degree_id": IdColumnRule(
        dtype="string",
        semantic="entity_id",
    ),
    "start_level_id": IdColumnRule(
        dtype="string",
        semantic="entity_id",
    ),
    # -------------------------
    # Semester / part codes
    # -------------------------
    "part_id": IdColumnRule(
        dtype="Int64",
        semantic="semester_code",
    ),
    "start_part_id": IdColumnRule(
        dtype="Int64",
        semantic="semester_code",
    ),
    "finish_part_id": IdColumnRule(
        dtype="Int64",
        semantic="semester_code",
    ),

    # -------------------------
    # Integer categorical codes
    # -------------------------
    "course_type_id": IdColumnRule(
        dtype="Int64",
        semantic="categorical_code",
    ),
    "diploma_state_id": IdColumnRule(
        dtype="Int64",
        semantic="categorical_code",
    ),
    "diploma_type_id": IdColumnRule(
        dtype="Int64",
        semantic="categorical_code",
    ),
    "permanent_status_id": IdColumnRule(
        dtype="Int64",
        semantic="categorical_code",
    ),
    "requirement_type_id": IdColumnRule(
        dtype="Int64",
        semantic="categorical_code",
    ),
}


# ============================================================
# Canonical table names — import these wherever normalize_ids
# is called; never pass a hand-typed string literal.
# ============================================================

TABLE_ACD_DEGREE_COURSE = "V_ACD_DEGREE_COURSE"
TABLE_ACS_GRADE = "V_ACS_GRADE"
TABLE_ADD_ACADEMIC_INFO = "V_ADD_ACADEMIC_INFO"
TABLE_ADD_STUDENT_DEGREE_STATUS = "V_ADD_STUDENT_DEGREE_STATUS"
TABLE_CRG_STUDENT_COURSE = "V_CRG_STUDENT_COURSE"


# ============================================================
# ID columns present in each table at cleaning-cast time
# (after the notebook's drop-columns step). Raw-only IDs that
# never survive to the cast (e.g. faculty_id in ACD,
# diploma_state_id in ACADEMIC_INFO) are not listed here;
# their rules stay in ID_COLUMN_RULES for documentation.
# ============================================================

TABLE_ID_COLUMNS: dict[str, tuple[str, ...]] = {
    TABLE_ACD_DEGREE_COURSE: (
        "course_id",
        "degree_course_id",
        "degree_id",
        "requirement_type_id",
    ),

    TABLE_ADD_ACADEMIC_INFO: (
        "diploma_type_id",
        "student_id",
    ),

    TABLE_ADD_STUDENT_DEGREE_STATUS: (
        "degree_id",
        "finish_part_id",
        "grade_version_id",
        "part_id",
        "permanent_status_id",
        "start_level_id",
        "start_part_id",
        "student_id",
        "student_status_id",
    ),

    TABLE_ACS_GRADE: (
        "grade_id",
        "grade_version_id",
    ),

    TABLE_CRG_STUDENT_COURSE: (
        "course_id",
        "degree_id",
        "faculty_id",
        "grade_id",
        "part_id",
        "student_course_id",
        "student_id",
    ),
}