from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.drive_sync import (
    DriveSyncArgs,
    DriveSyncService,
    GoogleDriveClient,
    build_run_config,
    resolve_roster_path,
    validate_inputs,
)
from scripts.models.pipeline import ClassSessionReport, DriveFile
from scripts.run_pipeline import RunArgs


# --- GoogleDriveClient.list_zip_files ---


def _fake_service(pages: list[dict]) -> MagicMock:
    service = MagicMock()
    service.files.return_value.list.return_value.execute.side_effect = pages
    return service


def test_list_zip_files_filters_to_zip() -> None:
    service = _fake_service(
        [
            {
                "files": [
                    {"id": "1", "name": "Economics.02.zip"},
                    {"id": "2", "name": "notes.pdf"},
                    {"id": "3", "name": "Math.01.ZIP"},
                ]
            }
        ]
    )
    client = GoogleDriveClient(service)

    files = client.list_zip_files("folder-1")

    assert [f.id for f in files] == ["1", "3"]
    assert all(f.name.lower().endswith(".zip") for f in files)


def test_list_zip_files_follows_pagination() -> None:
    service = _fake_service(
        [
            {"files": [{"id": "1", "name": "a.zip"}], "nextPageToken": "tok"},
            {"files": [{"id": "2", "name": "b.zip"}]},
        ]
    )
    client = GoogleDriveClient(service)

    files = client.list_zip_files("folder-1")

    assert [f.id for f in files] == ["1", "2"]


# --- DriveSyncService.sync ---


def make_config(tmp_path: Path) -> RunArgs:
    return RunArgs(
        input_path=tmp_path,
        output_dir=tmp_path / "out",
        teacher=["Nisha"],
        db_url="postgresql://localhost/adira",
    )


def make_service(
    tmp_path: Path,
    drive: MagicMock,
    store: MagicMock,
) -> DriveSyncService:
    return DriveSyncService(drive, store, make_config(tmp_path), "folder-1")


def session_report(class_name: str, success: bool, error: str | None = None) -> ClassSessionReport:
    return ClassSessionReport(
        class_name=class_name,
        zip_file=f"{class_name}.zip",
        output_dir=f"/out/{class_name}",
        step_results={},
        success=success,
        error=error,
    )


def test_sync_skips_already_processed(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_zip_files.return_value = [DriveFile(id="1", name="Econ.zip")]
    store = MagicMock()
    store.is_processed.return_value = True

    with patch("scripts.drive_sync.process_single_class") as mock_proc:
        report = make_service(tmp_path, drive, store).sync()

    mock_proc.assert_not_called()
    drive.download.assert_not_called()
    store.mark_processed.assert_not_called()
    assert report.skipped == 1
    assert report.processed == 0
    assert report.results[0].status == "skipped_duplicate"


def test_sync_processes_and_records_new_file(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_zip_files.return_value = [DriveFile(id="42", name="Economics.02.zip")]
    store = MagicMock()
    store.is_processed.return_value = False

    with patch(
        "scripts.drive_sync.process_single_class",
        return_value=session_report("Economics.02", success=True),
    ) as mock_proc:
        report = make_service(tmp_path, drive, store).sync()

    drive.download.assert_called_once()
    # The download target lives under a temp dir and is named after the Drive file.
    file_id, dest = drive.download.call_args.args
    assert file_id == "42"
    assert dest.name == "Economics.02.zip"
    mock_proc.assert_called_once()
    store.mark_processed.assert_called_once_with("42", "Economics.02")
    assert report.processed == 1
    assert report.results[0].status == "processed"


def test_sync_does_not_record_failed_pipeline(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_zip_files.return_value = [DriveFile(id="9", name="bomb.zip")]
    store = MagicMock()
    store.is_processed.return_value = False

    with patch(
        "scripts.drive_sync.process_single_class",
        return_value=session_report("bomb", success=False, error="zip bomb rejected"),
    ):
        report = make_service(tmp_path, drive, store).sync()

    store.mark_processed.assert_not_called()
    assert report.failed == 1
    assert report.results[0].status == "failed"
    assert report.results[0].error == "zip bomb rejected"


def test_sync_isolates_exception_and_continues(tmp_path: Path) -> None:
    drive = MagicMock()
    drive.list_zip_files.return_value = [
        DriveFile(id="bad", name="collide.zip"),
        DriveFile(id="good", name="Math.01.zip"),
    ]
    store = MagicMock()
    store.is_processed.return_value = False

    def proc(zip_path: Path, config: RunArgs) -> ClassSessionReport:
        if zip_path.name == "collide.zip":
            raise ValueError("two M4As share roll 2504")
        return session_report("Math.01", success=True)

    with patch("scripts.drive_sync.process_single_class", side_effect=proc):
        report = make_service(tmp_path, drive, store).sync()

    # Bad file failed and was not recorded; the batch still processed the good one.
    assert report.failed == 1
    assert report.processed == 1
    store.mark_processed.assert_called_once_with("good", "Math.01")
    statuses = {r.name: r.status for r in report.results}
    assert statuses == {"collide.zip": "failed", "Math.01.zip": "processed"}


# --- validate_inputs ---


def _args(tmp_path: Path, **overrides: object) -> DriveSyncArgs:
    sa = tmp_path / "sa.json"
    sa.write_text("{}", encoding="utf-8")
    defaults: dict[str, object] = {
        "service_account_json": sa,
        "folder_id": "folder-1",
        "output_dir": tmp_path / "out",
        "teacher": ["Nisha"],
        "db_url": "postgresql://localhost/adira",
    }
    defaults.update(overrides)
    return DriveSyncArgs(**defaults)  # type: ignore[arg-type]


def test_validate_inputs_ok(tmp_path: Path) -> None:
    validate_inputs(_args(tmp_path))  # should not raise


def test_validate_inputs_missing_service_account(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        validate_inputs(_args(tmp_path, service_account_json=tmp_path / "absent.json"))


def test_validate_inputs_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="folder id"):
        validate_inputs(_args(tmp_path, folder_id=""))


def test_validate_inputs_missing_db_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Database URL"):
        validate_inputs(_args(tmp_path, db_url=""))


def test_validate_inputs_missing_teacher(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="teacher"):
        validate_inputs(_args(tmp_path, teacher=[]))


# --- build_run_config ---


def test_build_run_config_carries_pipeline_settings(tmp_path: Path) -> None:
    roster = tmp_path / "roster.csv"
    config = build_run_config(_args(tmp_path, roster_path=roster, model="medium"))
    assert config.output_dir == tmp_path / "out"
    assert config.teacher == ["Nisha"]
    assert config.roster_path == roster
    assert config.model == "medium"
    assert config.db_url == "postgresql://localhost/adira"


# --- resolve_roster_path ---


def test_resolve_roster_path_prefers_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROSTER_CSV", str(tmp_path / "env.csv"))
    flag = tmp_path / "flag.csv"
    assert resolve_roster_path(flag) == flag


def test_resolve_roster_path_uses_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ROSTER_CSV", str(tmp_path / "env.csv"))
    assert resolve_roster_path(None) == tmp_path / "env.csv"


def test_resolve_roster_path_none_when_no_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROSTER_CSV", raising=False)
    monkeypatch.setattr("scripts.drive_sync._DEFAULT_ROSTER_PATH", Path("does/not/exist.csv"))
    assert resolve_roster_path(None) is None


def test_resolve_roster_path_falls_back_to_existing_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ROSTER_CSV", raising=False)
    default = tmp_path / "roster.csv"
    default.write_text("Name,RollNo,Email\n", encoding="utf-8")
    monkeypatch.setattr("scripts.drive_sync._DEFAULT_ROSTER_PATH", default)
    assert resolve_roster_path(None) == default
