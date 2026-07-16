from __future__ import annotations

from pathlib import Path

import pytest

from scripts.auth import (
    ROLL_EMAIL_RE,
    AuthError,
    Principal,
    allowed_student_ids,
    can_access_student,
    extract_roll,
    load_teacher_sections,
    parse_sections,
    principal_from_identity,
)

HEADER = "email,sections\n"

TEACHER_SECTIONS = {
    "arista@islorg.com": ["English.03", "English.04"],
    "nisha@islorg.com": ["Economics.02"],
}


def write_csv(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def resolve(email: str, lms_role: str) -> Principal | None:
    return principal_from_identity(
        email,
        lms_role,
        teacher_sections=dict(TEACHER_SECTIONS),
    )


# --- roll extraction ---


def test_roll_email_re_extracts_roll_from_localpart() -> None:
    match = ROLL_EMAIL_RE.search("bhagyashree_2302@islorg.com")
    assert match is not None
    assert match.group(1) == "2302"


@pytest.mark.parametrize(
    "email,expected",
    [
        ("bhagyashree_2302@islorg.com", "2302"),
        ("disha_rajesh_2505@islorg.com", "2505"),
        ("arista@islorg.com", None),
        ("nisha.sharma@islorg.com", None),
        ("student_23@islorg.com", None),
        ("student_23025@islorg.com", None),
        ("2302@islorg.com", None),
        ("", None),
    ],
)
def test_extract_roll(email: str, expected: str | None) -> None:
    assert extract_roll(email) == expected


# --- principal_from_identity ---


def test_admin_lms_role_is_denied() -> None:
    assert resolve("ratnanjali@islorg.com", "admin") is None


def test_mapped_teacher_gets_their_sections() -> None:
    principal = resolve("arista@islorg.com", "teacher")
    assert principal is not None
    assert principal.role == "teacher"
    assert principal.sections == ["English.03", "English.04"]
    assert principal.student_id is None


def test_unknown_teacher_is_denied_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        assert resolve("newteacher@islorg.com", "teacher") is None
    assert "newteacher@islorg.com" in caplog.text
    assert "no section mapping" in caplog.text


def test_teacher_principal_does_not_alias_the_sections_mapping() -> None:
    sections = {"arista@islorg.com": ["English.03"]}
    principal = principal_from_identity(
        "arista@islorg.com", "teacher", teacher_sections=sections
    )
    assert principal is not None
    principal.sections.append("Economics.02")
    assert sections["arista@islorg.com"] == ["English.03"]


def test_student_email_resolves_roll_to_student_id() -> None:
    principal = resolve("bhagyashree_2302@islorg.com", "student")
    assert principal is not None
    assert principal.role == "student"
    assert principal.student_id == "2302"
    assert principal.sections == []


def test_roll_less_student_email_is_denied_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        assert resolve("arista@islorg.com", "student") is None
    assert "no 4-digit roll" in caplog.text


def test_observer_is_denied() -> None:
    assert resolve("someone_2302@islorg.com", "observer") is None


@pytest.mark.parametrize("lms_role", ["", "Observer", "STUDENT", "Teacher", "superuser"])
def test_unknown_or_miscased_lms_role_is_denied(lms_role: str) -> None:
    assert resolve("bhagyashree_2302@islorg.com", lms_role) is None


def test_empty_email_is_denied() -> None:
    assert resolve("", "student") is None
    assert resolve("   ", "student") is None


def test_emails_are_case_and_whitespace_insensitive() -> None:
    student = resolve("  Bhagyashree_2302@ISLORG.com  ", "student")
    assert student is not None
    assert student.username == "bhagyashree_2302@islorg.com"
    assert student.student_id == "2302"

    teacher = resolve("ARISTA@islorg.com", "teacher")
    assert teacher is not None
    assert teacher.sections == ["English.03", "English.04"]


def test_no_teachers_configured_still_resolves_students() -> None:
    principal = principal_from_identity(
        "bhagyashree_2302@islorg.com",
        "student",
        teacher_sections={},
    )
    assert principal is not None
    assert principal.student_id == "2302"


# --- load_teacher_sections ---


def test_load_teacher_sections_success(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "t.csv",
        HEADER + "arista@islorg.com,English.03;English.04\nnisha@islorg.com,Economics.02\n",
    )
    assert load_teacher_sections(path) == {
        "arista@islorg.com": ["English.03", "English.04"],
        "nisha@islorg.com": ["Economics.02"],
    }


def test_load_teacher_sections_tolerant_headers(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "t.csv", " Email , Sections \narista@islorg.com,English.03\n")
    assert load_teacher_sections(path) == {"arista@islorg.com": ["English.03"]}


def test_load_teacher_sections_lowercases_emails(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "t.csv", HEADER + "ARISTA@ISLORG.com,English.03\n")
    assert load_teacher_sections(path) == {"arista@islorg.com": ["English.03"]}


def test_load_teacher_sections_trims_section_labels(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "t.csv", HEADER + "arista@islorg.com, English.03 ; English.04 \n")
    assert load_teacher_sections(path) == {"arista@islorg.com": ["English.03", "English.04"]}


def test_load_teacher_sections_keeps_labels_with_spaces(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "t.csv", HEADER + "arista@islorg.com,Math.01 A\n")
    assert load_teacher_sections(path) == {"arista@islorg.com": ["Math.01 A"]}


def test_load_teacher_sections_skips_comment_lines(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "t.csv",
        HEADER + "#,a docs comment line\narista@islorg.com,English.03\n",
    )
    assert load_teacher_sections(path) == {"arista@islorg.com": ["English.03"]}


def test_load_teacher_sections_skips_blank_lines(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "t.csv", HEADER + "arista@islorg.com,English.03\n\n")
    assert load_teacher_sections(path) == {"arista@islorg.com": ["English.03"]}


def test_load_teacher_sections_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AuthError, match="not found"):
        load_teacher_sections(tmp_path / "nope.csv")


def test_load_teacher_sections_missing_column(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "t.csv", "email\narista@islorg.com\n")
    with pytest.raises(AuthError, match="columns"):
        load_teacher_sections(path)


def test_load_teacher_sections_duplicate_email(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "t.csv",
        HEADER + "arista@islorg.com,English.03\nARISTA@islorg.com,English.04\n",
    )
    with pytest.raises(AuthError, match="Duplicate email"):
        load_teacher_sections(path)


def test_load_teacher_sections_empty_sections(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "t.csv", HEADER + "arista@islorg.com,\n")
    with pytest.raises(AuthError, match="has no sections"):
        load_teacher_sections(path)


def test_load_teacher_sections_separator_only_sections(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "t.csv", HEADER + "arista@islorg.com,; ;\n")
    with pytest.raises(AuthError, match="has no sections"):
        load_teacher_sections(path)


def test_load_teacher_sections_missing_email(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "t.csv", HEADER + ",English.03\n")
    with pytest.raises(AuthError, match="missing an email"):
        load_teacher_sections(path)


def test_load_teacher_sections_example_file_is_a_valid_empty_template() -> None:
    assert load_teacher_sections(Path("data/teacher_sections.example.csv")) == {}


def test_loaded_teacher_sections_feed_principal_from_identity(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "t.csv", HEADER + "arista@islorg.com,English.03;English.04\n")
    principal = principal_from_identity(
        "arista@islorg.com",
        "teacher",
        teacher_sections=load_teacher_sections(path),
    )
    assert principal is not None
    assert allowed_student_ids(principal, pairs_fixture()) == {"2402", "2403", "2405"}


# --- parse_sections ---


def test_parse_sections_splits_and_trims() -> None:
    assert parse_sections(" English.03 ; English.04 ") == ["English.03", "English.04"]


def test_parse_sections_of_empty_scope() -> None:
    assert parse_sections("") == []
    assert parse_sections(" ; ") == []


# --- authorization ---


def pairs_fixture() -> list[tuple[str, str, str]]:
    return [
        ("2401", "Aarav Shah", "Economics.02_AY2025-26_supply_2026-05-04"),
        ("2402", "Bhavna Rao", "English.03_AY2025-26_poetry_2026-05-05"),
        ("2403", "Chirag Jain", "English.04_AY2025-26_prose_2026-05-06"),
        ("2404", "Divya Menon", "Math.01 A_AY2025-26_algebra_2026-05-07"),
        ("2405", "Esha Patel", "Economics.02_AY2025-26_supply_2026-05-04"),
        ("2405", "Esha Patel", "English.04_AY2025-26_prose_2026-05-06"),
    ]


def teacher(sections: list[str]) -> Principal:
    return Principal(username="t@islorg.com", role="teacher", sections=sections)


def test_allowed_student_ids_student_is_self_only() -> None:
    principal = Principal(username="b_2402@islorg.com", role="student", student_id="2402")
    assert allowed_student_ids(principal, pairs_fixture()) == {"2402"}


def test_allowed_student_ids_student_without_scope_sees_nothing() -> None:
    principal = Principal(username="x@islorg.com", role="student")
    assert allowed_student_ids(principal, pairs_fixture()) == set()


def test_allowed_student_ids_english_teacher_sees_only_english_sections() -> None:
    allowed = allowed_student_ids(teacher(["English.03", "English.04"]), pairs_fixture())
    assert allowed == {"2402", "2403", "2405"}


def test_allowed_student_ids_economics_teacher_sees_only_economics_section() -> None:
    allowed = allowed_student_ids(teacher(["Economics.02"]), pairs_fixture())
    assert allowed == {"2401", "2405"}


def test_dual_subject_student_is_visible_to_both_teachers() -> None:
    english = allowed_student_ids(teacher(["English.04"]), pairs_fixture())
    economics = allowed_student_ids(teacher(["Economics.02"]), pairs_fixture())
    assert "2405" in english and "2405" in economics
    assert english & economics == {"2405"}


def test_allowed_student_ids_teacher_section_with_space_label() -> None:
    assert allowed_student_ids(teacher(["Math.01 A"]), pairs_fixture()) == {"2404"}


def test_allowed_student_ids_unknown_section_is_empty() -> None:
    assert allowed_student_ids(teacher(["History.09"]), pairs_fixture()) == set()


def test_allowed_student_ids_teacher_with_no_pairs_is_empty() -> None:
    assert allowed_student_ids(teacher(["English.03"]), []) == set()


def test_can_access_student_student_self_only() -> None:
    principal = Principal(username="b_2402@islorg.com", role="student", student_id="2402")
    assert can_access_student(principal, "2402", pairs_fixture()) is True
    assert can_access_student(principal, "2403", pairs_fixture()) is False


def test_can_access_student_teacher_is_section_scoped() -> None:
    principal = teacher(["English.03"])
    assert can_access_student(principal, "2402", pairs_fixture()) is True
    assert can_access_student(principal, "2401", pairs_fixture()) is False


def test_identity_to_authorization_end_to_end() -> None:
    student = resolve("bhagyashree_2302@islorg.com", "student")
    arista = resolve("arista@islorg.com", "teacher")
    assert student is not None and arista is not None

    pairs = pairs_fixture()
    assert can_access_student(student, "2302", pairs) is True
    assert can_access_student(student, "2402", pairs) is False
    assert can_access_student(arista, "2402", pairs) is True
    assert can_access_student(arista, "2401", pairs) is False
