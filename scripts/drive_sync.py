from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol, Sequence

from dotenv import load_dotenv
from pydantic import BaseModel

from scripts.models.pipeline import DriveFile, DriveFileResult, DriveSyncReport
from scripts.run_pipeline import RunArgs, process_single_class
from scripts.utils.db_url import resolve_db_url
from scripts.utils.processed_files import (
    ProcessedFilesStore,
    connect_processed_files_store,
)

logger = logging.getLogger(__name__)

# Read-only: the sync only ever lists and downloads; it never writes to Drive.
_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class DriveClient(Protocol):
    """The slice of a Drive client the sync depends on (mockable in tests)."""

    def list_zip_files(self, folder_id: str) -> list[DriveFile]: ...

    def download(self, file_id: str, dest: Path) -> None: ...


class GoogleDriveClient:
    """Thin wrapper over the Google Drive v3 API resource.

    Google's client libraries ship no type stubs, so the ``google*`` imports are
    lazy (kept out of import time / test collection) and the resource is held as
    ``Any``; the public method surface here is fully typed.
    """

    def __init__(self, service: Any) -> None:
        self._service = service

    def list_zip_files(self, folder_id: str) -> list[DriveFile]:
        query = f"'{folder_id}' in parents and trashed = false"
        files: list[DriveFile] = []
        page_token: str | None = None
        while True:
            response = (
                self._service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name)",
                    pageToken=page_token,
                )
                .execute()
            )
            for item in response.get("files", []):
                name = item.get("name", "")
                if name.lower().endswith(".zip"):
                    files.append(DriveFile(id=item["id"], name=name))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return files

    def download(self, file_id: str, dest: Path) -> None:
        from googleapiclient.http import MediaIoBaseDownload

        request = self._service.files().get_media(fileId=file_id)
        with dest.open("wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()


def build_drive_client(service_account_json: Path) -> GoogleDriveClient:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds = service_account.Credentials.from_service_account_file(
        str(service_account_json), scopes=_DRIVE_SCOPES
    )
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    return GoogleDriveClient(service)


class DriveSyncService:
    """Poll a Drive folder for Zoom ``.zip`` exports and feed new ones to the pipeline.

    Idempotent via the ``processed_files`` ledger: a zip whose Drive file id is
    already recorded is skipped. Each new zip is downloaded to a temp dir, processed
    by :func:`scripts.run_pipeline.process_single_class`, and recorded **only on
    success**. A failed file -- including the two guard rejections the pipeline now
    raises, the zip-bomb guard (oversized / too-many-entry archives) and the
    colliding-roll guard (two M4As sharing a 4-digit roll) -- is left UNrecorded so a
    fixed re-upload can be retried, and one bad zip never aborts the batch.
    """

    def __init__(
        self,
        drive_client: DriveClient,
        store: ProcessedFilesStore,
        config: RunArgs,
        folder_id: str,
    ) -> None:
        self._drive = drive_client
        self._store = store
        self._config = config
        self._folder_id = folder_id

    def sync(self) -> DriveSyncReport:
        files = self._drive.list_zip_files(self._folder_id)
        report = DriveSyncReport(folder_id=self._folder_id, total_listed=len(files))
        for drive_file in files:
            result = self._process_one(drive_file)
            report.results.append(result)
            if result.status == "processed":
                report.processed += 1
            elif result.status == "skipped_duplicate":
                report.skipped += 1
            else:
                report.failed += 1
        return report

    def _process_one(self, drive_file: DriveFile) -> DriveFileResult:
        class_name = Path(drive_file.name).stem
        if self._store.is_processed(drive_file.id):
            logger.info("Skipping already-processed %s", drive_file.name)
            return DriveFileResult(
                drive_file_id=drive_file.id,
                name=drive_file.name,
                class_name=class_name,
                status="skipped_duplicate",
            )
        try:
            with tempfile.TemporaryDirectory(prefix="drive_sync_") as tmp:
                local_zip = Path(tmp) / drive_file.name
                self._drive.download(drive_file.id, local_zip)
                session = process_single_class(local_zip, self._config)
        except Exception as exc:
            # Per-file isolation: one bad zip must not abort the batch, and an
            # errored file is deliberately left UNrecorded so it can be retried.
            logger.error("Error processing %s: %s", drive_file.name, exc)
            logger.debug("Drive file %s traceback", drive_file.name, exc_info=True)
            return DriveFileResult(
                drive_file_id=drive_file.id,
                name=drive_file.name,
                class_name=class_name,
                status="failed",
                error=str(exc),
            )

        if not session.success:
            logger.warning("Pipeline failed for %s: %s", drive_file.name, session.error)
            return DriveFileResult(
                drive_file_id=drive_file.id,
                name=drive_file.name,
                class_name=session.class_name,
                status="failed",
                error=session.error,
            )

        self._store.mark_processed(drive_file.id, session.class_name)
        return DriveFileResult(
            drive_file_id=drive_file.id,
            name=drive_file.name,
            class_name=session.class_name,
            status="processed",
        )


class DriveSyncArgs(BaseModel):
    service_account_json: Path
    folder_id: str
    output_dir: Path
    teacher: list[str]
    db_url: str = ""
    roster_path: Path | None = None
    attendance_path: Path | None = None
    model: str = "small"
    allow_cpu: bool = False


_DEFAULT_ROSTER_PATH = Path("data/roster.csv")


def resolve_roster_path(flag_value: Path | None) -> Path | None:
    """Resolve the roster CSV (Name, RollNo, Email) for absent-student context.

    Precedence: an explicit ``--roster`` flag, then the ``ROSTER_CSV`` env var, then
    a ``data/roster.csv`` default if it exists. Returns ``None`` when no roster is
    available, in which case absent students simply get no context object.
    """
    if flag_value is not None:
        return flag_value
    env_val = os.getenv("ROSTER_CSV", "").strip()
    if env_val:
        return Path(env_val)
    return _DEFAULT_ROSTER_PATH if _DEFAULT_ROSTER_PATH.exists() else None


def parse_args(argv: Sequence[str] | None = None) -> DriveSyncArgs:
    parser = argparse.ArgumentParser(
        description="Poll a Google Drive folder for Zoom .zip exports and ingest new ones."
    )
    parser.add_argument("--output-dir", required=True, type=Path, dest="output_dir")
    parser.add_argument("--teacher", action="append", default=[], dest="teacher")
    parser.add_argument("--roster", type=Path, dest="roster_path", default=None)
    parser.add_argument("--attendance", type=Path, dest="attendance_path", default=None)
    parser.add_argument(
        "--db-url",
        default=None,
        dest="db_url",
        help="PostgreSQL connection URL. Falls back to DATABASE_URL env var.",
    )
    parser.add_argument("--model", default="small")
    parser.add_argument("--allow-cpu", action="store_true", dest="allow_cpu")
    namespace = parser.parse_args(argv)

    # Load .env first so GOOGLE_* below resolve from it on the local-scheduler path
    # (the CI path sets them as real env vars, which take precedence either way).
    load_dotenv()

    # Secrets / infra identifiers come from the environment, never CLI flags.
    service_account_json = Path(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip())
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip()

    return DriveSyncArgs(
        service_account_json=service_account_json,
        folder_id=folder_id,
        output_dir=namespace.output_dir,
        teacher=namespace.teacher,
        db_url=resolve_db_url(namespace.db_url),
        roster_path=resolve_roster_path(namespace.roster_path),
        attendance_path=namespace.attendance_path,
        model=namespace.model,
        allow_cpu=namespace.allow_cpu,
    )


def validate_inputs(args: DriveSyncArgs) -> None:
    if not str(args.service_account_json).strip():
        raise ValueError(
            "Service-account JSON path is required. Set GOOGLE_SERVICE_ACCOUNT_JSON in .env."
        )
    if not args.service_account_json.exists():
        raise ValueError(f"Service-account JSON not found: {args.service_account_json}")
    if not args.folder_id:
        raise ValueError("Drive folder id is required. Set GOOGLE_DRIVE_FOLDER_ID in .env.")
    if not args.db_url.strip():
        raise ValueError("Database URL is required. Set DATABASE_URL in .env.")
    if not args.teacher:
        raise ValueError("At least one --teacher name is required.")


def build_run_config(args: DriveSyncArgs) -> RunArgs:
    # input_path is unused by process_single_class (it takes the zip explicitly); set
    # it to the output dir so the RunArgs contract is satisfied without a sentinel.
    return RunArgs(
        input_path=args.output_dir,
        output_dir=args.output_dir,
        teacher=args.teacher,
        roster_path=args.roster_path,
        attendance_path=args.attendance_path,
        db_url=args.db_url,
        model=args.model,
        allow_cpu=args.allow_cpu,
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        args = parse_args(argv)
        validate_inputs(args)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from exc

    drive_client = build_drive_client(args.service_account_json)
    store = connect_processed_files_store(args.db_url)
    service = DriveSyncService(drive_client, store, build_run_config(args), args.folder_id)
    try:
        report = service.sync()
    finally:
        store.close()

    print(
        f"Drive sync complete: processed={report.processed} skipped={report.skipped} "
        f"failed={report.failed} of {report.total_listed} listed"
    )
    if report.failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
