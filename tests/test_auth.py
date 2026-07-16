from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from scripts.auth import (
    PBKDF2_ITERATIONS,
    AuthError,
    AuthService,
    CredentialRecord,
    Principal,
    _DUMMY_PASSWORD_HASH,
    _parse_password_hash,
    add_user,
    allowed_student_ids,
    build_parser,
    can_access_student,
    hash_password,
    load_credentials,
    main,
    verify_password,
)

HEADER = "username,role,scope,password_hash\n"


def write_csv(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def row(username: str, role: str, scope: str, password: str) -> str:
    return f"{username},{role},{scope},{hash_password(password, iterations=1)}\n"


def prompter(*answers: str) -> Callable[[str], str]:
    values: Iterator[str] = iter(answers)
    return lambda _: next(values)


# --- hashing ---


def test_hash_password_roundtrip() -> None:
    encoded = hash_password("s3cret", iterations=1)
    assert verify_password("s3cret", encoded) is True


def test_hash_password_wrong_password_fails() -> None:
    assert verify_password("nope", hash_password("s3cret", iterations=1)) is False


def test_hash_password_encodes_scheme_and_iterations() -> None:
    scheme, iterations, salt_hex, hash_hex = hash_password("s3cret", iterations=7).split("$")
    assert scheme == "pbkdf2_sha256"
    assert iterations == "7"
    assert len(bytes.fromhex(salt_hex)) == 16
    assert len(bytes.fromhex(hash_hex)) == 32


def test_hash_password_uses_distinct_salt_per_user() -> None:
    first = hash_password("same-password", iterations=1)
    second = hash_password("same-password", iterations=1)
    assert first.split("$")[2] != second.split("$")[2]
    assert first != second


def test_hash_password_default_iterations_is_the_constant() -> None:
    assert hash_password("s3cret", salt=b"\x00" * 16).split("$")[1] == str(PBKDF2_ITERATIONS)
    assert PBKDF2_ITERATIONS == 600_000


def test_verify_rejects_tampered_hash() -> None:
    scheme, iterations, salt_hex, hash_hex = hash_password("s3cret", iterations=1).split("$")
    flipped = f"{int(hash_hex[:2], 16) ^ 0xFF:02x}{hash_hex[2:]}"
    assert verify_password("s3cret", f"{scheme}${iterations}${salt_hex}${flipped}") is False


def test_verify_rejects_tampered_salt() -> None:
    scheme, iterations, salt_hex, hash_hex = hash_password("s3cret", iterations=1).split("$")
    flipped = f"{int(salt_hex[:2], 16) ^ 0xFF:02x}{salt_hex[2:]}"
    assert verify_password("s3cret", f"{scheme}${iterations}${flipped}${hash_hex}") is False


def test_verify_rejects_tampered_iterations() -> None:
    scheme, _, salt_hex, hash_hex = hash_password("s3cret", iterations=1).split("$")
    assert verify_password("s3cret", f"{scheme}$2${salt_hex}${hash_hex}") is False


@pytest.mark.parametrize(
    "encoded",
    [
        "",
        "plaintext",
        "pbkdf2_sha256$1$deadbeef",
        "bcrypt$1$dead$beef",
        "pbkdf2_sha256$notanint$dead$beef",
        "pbkdf2_sha256$1$nothex$beef",
        "pbkdf2_sha256$0$dead$beef",
    ],
)
def test_verify_rejects_malformed_encoding(encoded: str) -> None:
    assert verify_password("s3cret", encoded) is False


def test_dummy_hash_iterations_match_the_constant() -> None:
    parsed = _parse_password_hash(_DUMMY_PASSWORD_HASH)
    assert parsed is not None
    assert parsed[0] == PBKDF2_ITERATIONS


# --- load_credentials ---


def test_load_credentials_success(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "c.csv",
        HEADER + row("owner", "admin", "", "a") + row("2302", "student", "2302", "b"),
    )
    creds = load_credentials(path)
    assert set(creds) == {"owner", "2302"}
    assert creds["2302"].role == "student"
    assert creds["2302"].scope == "2302"


def test_load_credentials_tolerant_headers(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "c.csv",
        " Username , Role , Scope , Password_Hash \n" + row("owner", "admin", "", "a"),
    )
    assert set(load_credentials(path)) == {"owner"}


def test_load_credentials_skips_blank_lines(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "c.csv",
        HEADER + row("owner", "admin", "", "a") + "\n" + row("2302", "student", "2302", "b"),
    )
    assert set(load_credentials(path)) == {"owner", "2302"}


def test_load_credentials_skips_comment_lines(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "c.csv",
        HEADER + "#,a docs comment line,ignored,ignored\n" + row("owner", "admin", "", "a"),
    )
    assert set(load_credentials(path)) == {"owner"}


def test_load_credentials_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AuthError, match="not found"):
        load_credentials(tmp_path / "nope.csv")


def test_load_credentials_missing_columns(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", "username,role\nowner,admin\n")
    with pytest.raises(AuthError, match="columns"):
        load_credentials(path)


def test_load_credentials_old_plaintext_format_names_the_cli(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", "student_id,password\n2302,alpha\n")
    with pytest.raises(AuthError, match="old plaintext format") as exc:
        load_credentials(path)
    assert "python -m scripts.auth add-user" in str(exc.value)


def test_load_credentials_bad_role(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", HEADER + row("x", "principal", "", "a"))
    with pytest.raises(AuthError, match="invalid role 'principal'"):
        load_credentials(path)


def test_load_credentials_duplicate_username(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "c.csv",
        HEADER + row("owner", "admin", "", "a") + row("owner", "admin", "", "b"),
    )
    with pytest.raises(AuthError, match="Duplicate username"):
        load_credentials(path)


def test_load_credentials_empty_username(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", HEADER + f",admin,,{hash_password('a', iterations=1)}\n")
    with pytest.raises(AuthError, match="missing a username"):
        load_credentials(path)


def test_load_credentials_empty_password_hash(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", HEADER + "owner,admin,,\n")
    with pytest.raises(AuthError, match="empty password_hash"):
        load_credentials(path)


def test_load_credentials_rejects_plaintext_in_hash_column(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", HEADER + "owner,admin,,hunter2\n")
    with pytest.raises(AuthError, match="malformed password_hash"):
        load_credentials(path)


def test_load_credentials_student_without_scope(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", HEADER + row("2302", "student", "", "a"))
    with pytest.raises(AuthError, match="student '2302' needs a scope"):
        load_credentials(path)


def test_load_credentials_teacher_without_scope(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", HEADER + row("nisha", "teacher", "", "a"))
    with pytest.raises(AuthError, match="teacher 'nisha' needs a scope"):
        load_credentials(path)


def test_load_credentials_admin_with_scope(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", HEADER + row("owner", "admin", "English.03", "a"))
    with pytest.raises(AuthError, match="admin 'owner' must have an empty scope"):
        load_credentials(path)


def test_load_credentials_example_file_is_valid() -> None:
    creds = load_credentials(Path("data/credentials.example.csv"))
    assert {r.role for r in creds.values()} == {"admin", "teacher", "student"}


# --- authenticate ---


def make_service() -> AuthService:
    return AuthService(
        {
            "owner": CredentialRecord(
                username="owner", role="admin", scope="", password_hash=hash_password("alpha", iterations=1)
            ),
            "nisha": CredentialRecord(
                username="nisha",
                role="teacher",
                scope="English.03;English.04",
                password_hash=hash_password("beta", iterations=1),
            ),
            "2302": CredentialRecord(
                username="2302", role="student", scope="2302", password_hash=hash_password("gamma", iterations=1)
            ),
        }
    )


def test_authenticate_admin_principal() -> None:
    principal = make_service().authenticate("owner", "alpha")
    assert principal == Principal(username="owner", role="admin", student_id=None, sections=[])


def test_authenticate_teacher_principal_parses_sections() -> None:
    principal = make_service().authenticate("nisha", "beta")
    assert principal is not None
    assert principal.role == "teacher"
    assert principal.sections == ["English.03", "English.04"]
    assert principal.student_id is None


def test_authenticate_student_principal_carries_student_id() -> None:
    principal = make_service().authenticate("2302", "gamma")
    assert principal is not None
    assert principal.role == "student"
    assert principal.student_id == "2302"
    assert principal.sections == []


def test_authenticate_wrong_password() -> None:
    assert make_service().authenticate("2302", "wrong") is None


def test_authenticate_unknown_username() -> None:
    assert make_service().authenticate("9999", "gamma") is None


def test_authenticate_unknown_username_runs_a_real_pbkdf2(monkeypatch: pytest.MonkeyPatch) -> None:
    service = make_service()
    calls: list[tuple[Any, ...]] = []
    real = hashlib.pbkdf2_hmac

    def spy(*args: Any, **kwargs: Any) -> bytes:
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(hashlib, "pbkdf2_hmac", spy)
    assert service.authenticate("9999", "gamma") is None

    assert len(calls) == 1
    scheme, password, salt, iterations = calls[0]
    assert scheme == "sha256"
    assert password == b"gamma"
    assert salt == bytes.fromhex(_DUMMY_PASSWORD_HASH.split("$")[2])
    assert iterations == PBKDF2_ITERATIONS


def test_authenticate_never_logs_the_password(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("DEBUG"):
        make_service().authenticate("2302", "gamma")
        make_service().authenticate("9999", "hunter2")
    assert "gamma" not in caplog.text
    assert "hunter2" not in caplog.text


def test_authenticate_non_ascii_password() -> None:
    service = AuthService(
        {
            "2302": CredentialRecord(
                username="2302",
                role="student",
                scope="2302",
                password_hash=hash_password("pä55wörd-गुप्त", iterations=1),
            )
        }
    )
    assert service.authenticate("2302", "pä55wörd-गुप्त") is not None
    assert service.authenticate("2302", "pä55wörd") is None


def test_from_csv(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", HEADER + row("2302", "student", "2302", "alpha"))
    principal = AuthService.from_csv(path).authenticate("2302", "alpha")
    assert principal is not None
    assert principal.student_id == "2302"


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
    return Principal(username="t", role="teacher", sections=sections)


def test_allowed_student_ids_admin_is_unrestricted() -> None:
    assert allowed_student_ids(Principal(username="o", role="admin"), pairs_fixture()) is None


def test_allowed_student_ids_student_is_self_only() -> None:
    principal = Principal(username="2402", role="student", student_id="2402")
    assert allowed_student_ids(principal, pairs_fixture()) == {"2402"}


def test_allowed_student_ids_student_without_scope_sees_nothing() -> None:
    assert allowed_student_ids(Principal(username="x", role="student"), pairs_fixture()) == set()


def test_allowed_student_ids_english_teacher_sees_only_english_sections() -> None:
    allowed = allowed_student_ids(teacher(["English.03", "English.04"]), pairs_fixture())
    assert allowed == {"2402", "2403", "2405"}


def test_allowed_student_ids_economics_teacher_sees_only_economics_section() -> None:
    allowed = allowed_student_ids(teacher(["Economics.02"]), pairs_fixture())
    assert allowed == {"2401", "2405"}


def test_dual_subject_student_is_visible_to_both_teachers() -> None:
    english = allowed_student_ids(teacher(["English.04"]), pairs_fixture())
    economics = allowed_student_ids(teacher(["Economics.02"]), pairs_fixture())
    assert english is not None and economics is not None
    assert "2405" in english and "2405" in economics
    assert english & economics == {"2405"}


def test_allowed_student_ids_teacher_section_with_space_label() -> None:
    assert allowed_student_ids(teacher(["Math.01 A"]), pairs_fixture()) == {"2404"}


def test_allowed_student_ids_unknown_section_is_empty() -> None:
    assert allowed_student_ids(teacher(["History.09"]), pairs_fixture()) == set()


def test_allowed_student_ids_teacher_with_no_pairs_is_empty() -> None:
    assert allowed_student_ids(teacher(["English.03"]), []) == set()


def test_can_access_student_admin_can_access_anyone() -> None:
    principal = Principal(username="o", role="admin")
    assert can_access_student(principal, "2401", pairs_fixture()) is True
    assert can_access_student(principal, "unknown-id", pairs_fixture()) is True


def test_can_access_student_student_self_only() -> None:
    principal = Principal(username="2402", role="student", student_id="2402")
    assert can_access_student(principal, "2402", pairs_fixture()) is True
    assert can_access_student(principal, "2403", pairs_fixture()) is False


def test_can_access_student_teacher_is_section_scoped() -> None:
    principal = teacher(["English.03"])
    assert can_access_student(principal, "2402", pairs_fixture()) is True
    assert can_access_student(principal, "2401", pairs_fixture()) is False


# --- CLI ---


def test_add_user_creates_a_verifiable_row(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    exit_code = main(
        ["add-user", "--credentials", str(path), "--username", "2302", "--role", "student", "--scope", "2302"],
        prompt_fn=prompter("s3cret", "s3cret"),
        output_fn=lambda _: None,
    )
    assert exit_code == 0
    principal = AuthService.from_csv(path).authenticate("2302", "s3cret")
    assert principal is not None
    assert principal.student_id == "2302"


def test_add_user_stores_no_plaintext(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    main(
        ["add-user", "--credentials", str(path), "--username", "owner", "--role", "admin"],
        prompt_fn=prompter("s3cret", "s3cret"),
        output_fn=lambda _: None,
    )
    body = path.read_text(encoding="utf-8")
    assert "s3cret" not in body
    assert "pbkdf2_sha256$600000$" in body


def test_add_user_idempotently_updates_an_existing_username(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    argv = ["add-user", "--credentials", str(path), "--username", "owner", "--role", "admin"]
    main(argv, prompt_fn=prompter("first-pw", "first-pw"), output_fn=lambda _: None)
    main(argv, prompt_fn=prompter("second-pw", "second-pw"), output_fn=lambda _: None)

    creds = load_credentials(path)
    assert list(creds) == ["owner"]
    service = AuthService(creds)
    assert service.authenticate("owner", "second-pw") is not None
    assert service.authenticate("owner", "first-pw") is None


def test_add_user_keeps_other_rows(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    write_csv(path, HEADER + row("nisha", "teacher", "English.03", "beta"))
    main(
        ["add-user", "--credentials", str(path), "--username", "owner", "--role", "admin"],
        prompt_fn=prompter("s3cret", "s3cret"),
        output_fn=lambda _: None,
    )
    assert set(load_credentials(path)) == {"nisha", "owner"}


def test_add_user_teacher_scope_parses_into_sections(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    main(
        [
            "add-user",
            "--credentials",
            str(path),
            "--username",
            "nisha",
            "--role",
            "teacher",
            "--scope",
            "English.03;English.04",
        ],
        prompt_fn=prompter("s3cret", "s3cret"),
        output_fn=lambda _: None,
    )
    principal = AuthService.from_csv(path).authenticate("nisha", "s3cret")
    assert principal is not None
    assert principal.sections == ["English.03", "English.04"]


def test_add_user_rejects_mismatched_confirmation(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    messages: list[str] = []
    exit_code = main(
        ["add-user", "--credentials", str(path), "--username", "owner", "--role", "admin"],
        prompt_fn=prompter("s3cret", "typo"),
        output_fn=messages.append,
    )
    assert exit_code == 1
    assert "do not match" in messages[0]
    assert not path.exists()


def test_add_user_rejects_empty_password(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    messages: list[str] = []
    exit_code = main(
        ["add-user", "--credentials", str(path), "--username", "owner", "--role", "admin"],
        prompt_fn=prompter("", ""),
        output_fn=messages.append,
    )
    assert exit_code == 1
    assert "must not be empty" in messages[0]


def test_add_user_rejects_invalid_role_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    with pytest.raises(SystemExit):
        main(
            ["add-user", "--credentials", str(path), "--username", "x", "--role", "principal"],
            prompt_fn=prompter("s3cret", "s3cret"),
            output_fn=lambda _: None,
        )
    assert not path.exists()


def test_add_user_rejects_admin_with_scope(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    messages: list[str] = []
    exit_code = main(
        ["add-user", "--credentials", str(path), "--username", "owner", "--role", "admin", "--scope", "English.03"],
        prompt_fn=prompter("s3cret", "s3cret"),
        output_fn=messages.append,
    )
    assert exit_code == 1
    assert "empty scope" in messages[0]


def test_cli_has_no_password_flag() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["add-user", "--credentials", "c.csv", "--username", "u", "--role", "admin", "--password", "s3cret"]
        )


def test_add_user_helper_hashes_the_password(tmp_path: Path) -> None:
    record = add_user(tmp_path / "c.csv", "owner", "admin", "", "s3cret")
    assert record.password_hash.startswith("pbkdf2_sha256$")
    assert verify_password("s3cret", record.password_hash) is True


def test_verify_prints_role_and_scope_on_success(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", HEADER + row("nisha", "teacher", "English.03;English.04", "beta"))
    messages: list[str] = []
    exit_code = main(
        ["verify", "--credentials", str(path), "--username", "nisha"],
        prompt_fn=prompter("beta"),
        output_fn=messages.append,
    )
    assert exit_code == 0
    assert "role=teacher" in messages[0]
    assert "scope=English.03;English.04" in messages[0]


def test_verify_fails_on_wrong_password(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "c.csv", HEADER + row("owner", "admin", "", "alpha"))
    messages: list[str] = []
    exit_code = main(
        ["verify", "--credentials", str(path), "--username", "owner"],
        prompt_fn=prompter("wrong"),
        output_fn=messages.append,
    )
    assert exit_code == 1
    assert "failed" in messages[0]


def test_verify_reports_a_missing_file(tmp_path: Path) -> None:
    messages: list[str] = []
    exit_code = main(
        ["verify", "--credentials", str(tmp_path / "nope.csv"), "--username", "owner"],
        prompt_fn=prompter("alpha"),
        output_fn=messages.append,
    )
    assert exit_code == 1
    assert "not found" in messages[0]


# --- bulk provisioning ---


def write_roster(path: Path, rows: list[tuple[str, str]]) -> Path:
    body = "STUDENT NAME,STUDENT ID\n" + "".join(f"{name},{roll}\n" for name, roll in rows)
    path.write_text(body, encoding="utf-8")
    return path


def test_generate_password_length_and_alphabet() -> None:
    from scripts.auth import _PASSWORD_ALPHABET, generate_password

    password = generate_password()
    assert len(password) == 10
    assert all(ch in _PASSWORD_ALPHABET for ch in password)
    assert not set("01ilo") & set(password)


def test_bulk_provision_creates_verifiable_student_accounts(tmp_path: Path) -> None:
    from scripts.auth import bulk_provision

    roster = write_roster(tmp_path / "c1.csv", [("anshi", "2301"), ("Ranu Suthar", "2306")])
    creds = tmp_path / "credentials.csv"
    handout = tmp_path / "handout.csv"

    created, skipped = bulk_provision(creds, [roster], handout, iterations=1)

    assert (created, skipped) == (2, 0)
    service = AuthService.from_csv(creds)
    lines = handout.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "username,student_name,password"
    for line in lines[1:]:
        username, student_name, password = line.split(",")
        principal = service.authenticate(username, password)
        assert principal is not None
        assert principal.role == "student"
        assert principal.student_id == username
    assert "anshi" in lines[1]


def test_bulk_provision_skips_existing_users_by_default(tmp_path: Path) -> None:
    from scripts.auth import bulk_provision

    roster = write_roster(tmp_path / "c1.csv", [("anshi", "2301")])
    creds = write_csv(tmp_path / "credentials.csv", HEADER + row("2301", "student", "2301", "keepme"))
    before = load_credentials(creds)["2301"].password_hash

    created, skipped = bulk_provision(creds, [roster], tmp_path / "handout.csv", iterations=1)

    assert (created, skipped) == (0, 1)
    assert load_credentials(creds)["2301"].password_hash == before


def test_bulk_provision_reset_existing_regenerates(tmp_path: Path) -> None:
    from scripts.auth import bulk_provision

    roster = write_roster(tmp_path / "c1.csv", [("anshi", "2301")])
    creds = write_csv(tmp_path / "credentials.csv", HEADER + row("2301", "student", "2301", "old"))
    before = load_credentials(creds)["2301"].password_hash

    created, skipped = bulk_provision(
        creds, [roster], tmp_path / "handout.csv", reset_existing=True, iterations=1
    )

    assert (created, skipped) == (1, 0)
    assert load_credentials(creds)["2301"].password_hash != before


def test_bulk_provision_duplicate_roll_across_rosters_provisioned_once(tmp_path: Path) -> None:
    from scripts.auth import bulk_provision

    first = write_roster(tmp_path / "a.csv", [("anshi", "2301")])
    second = write_roster(tmp_path / "b.csv", [("anshi again", "2301")])
    handout = tmp_path / "handout.csv"

    created, skipped = bulk_provision(
        tmp_path / "credentials.csv", [first, second], handout, iterations=1
    )

    assert (created, skipped) == (1, 0)
    assert len(handout.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_bulk_provision_preserves_teacher_and_admin_rows(tmp_path: Path) -> None:
    from scripts.auth import bulk_provision

    roster = write_roster(tmp_path / "c1.csv", [("anshi", "2301")])
    creds = write_csv(
        tmp_path / "credentials.csv",
        HEADER + row("owner", "admin", "", "alpha") + row("arista", "teacher", "English.03", "beta"),
    )

    bulk_provision(creds, [roster], tmp_path / "handout.csv", iterations=1)

    loaded = load_credentials(creds)
    assert set(loaded) == {"owner", "arista", "2301"}
    assert loaded["arista"].role == "teacher"


def test_bulk_provision_empty_roster_fails_loud(tmp_path: Path) -> None:
    from scripts.auth import bulk_provision

    roster = write_roster(tmp_path / "c1.csv", [])
    with pytest.raises(AuthError, match="No roster students"):
        bulk_provision(tmp_path / "credentials.csv", [roster], tmp_path / "handout.csv", iterations=1)


def test_cli_bulk_provision_reports_and_warns(tmp_path: Path) -> None:
    roster = write_roster(tmp_path / "c1.csv", [("anshi", "2301")])
    handout = tmp_path / "handout.csv"
    messages: list[str] = []

    exit_code = main(
        [
            "bulk-provision",
            "--credentials", str(tmp_path / "credentials.csv"),
            "--roster", str(roster),
            "--handout", str(handout),
        ],
        prompt_fn=prompter(),
        output_fn=messages.append,
    )

    assert exit_code == 0
    assert "1 student account(s)" in messages[0]
    assert "DELETE" in messages[0]
    assert handout.exists()
