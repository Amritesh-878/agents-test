from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest

from scripts.embed_and_store import chunk_material_blocks, embed_records_deduped
from scripts.ingest_materials import (
    IngestArgs,
    MaterialStudent,
    build_material_records,
    enrolled_students,
    main,
    parse_args,
    resolve_students,
    run_ingest_materials,
    validate_inputs,
)
from scripts.models.identity import IdentityMap, IdentityMapEntry
from scripts.models.pipeline import EmbeddingRecord

GOOD_BLOCK = (
    "The supply function links quantity supplied to the price level and its determinants."
)
LONG_BLOCK = (
    "A supply schedule lists the quantity producers plan to sell at each price. "
    "When input costs fall, firms can offer more output profitably, shifting the whole curve outward. "
    "Improvements in technology raise productivity, letting the same factories deliver larger volumes. "
    "Taxes work in the opposite direction because they raise the effective cost of every unit sold. "
    "Subsidies lower marginal cost and therefore expand planned production at any given price point. "
    "Expectations about future prices also matter, since sellers may withhold stock today hoping for better margins tomorrow. "
    "Finally the number of sellers in the market determines total industry supply, as each entrant adds its own capacity to the aggregate. "
    "Weather and seasonal cycles influence agricultural supply in ways manufacturers rarely experience."
)


class FakeStore:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []
        self.upserted: list[EmbeddingRecord] = []
        self.closed = False

    def delete_student_material_chunks(self, class_name: str, student_id: str) -> int:
        self.deleted.append((class_name, student_id))
        return 0

    def delete_class_chunks(self, class_name: str) -> int:
        raise AssertionError("materials ingest must never purge whole-class chunks")

    def upsert_chunks(self, records: Sequence[EmbeddingRecord]) -> int:
        self.upserted.extend(records)
        return len(records)

    def close(self) -> None:
        self.closed = True


def fake_embed(records: list[EmbeddingRecord], model_name: str) -> list[EmbeddingRecord]:
    for record in records:
        record.embedding = [0.1, 0.2, 0.3]
    return records


def make_identity_map() -> IdentityMap:
    return IdentityMap(
        teacher_name="Nisha",
        teacher_audio_file="audio_teacher.m4a",
        entries=[
            IdentityMapEntry(
                audio_file="audio_teacher.m4a",
                matched_name="Nisha",
                match_method="fuzzy_name",
                match_confidence=1.0,
                is_teacher=True,
            ),
            IdentityMapEntry(
                audio_file="audio_2301.m4a",
                matched_name="Anshi Verma",
                matched_roll_no="2301",
                match_method="roll_no",
                match_confidence=1.0,
            ),
            IdentityMapEntry(
                audio_file="audio_kabir.m4a",
                matched_name="Kabir Rao",
                match_method="fuzzy_name",
                match_confidence=0.8,
            ),
            IdentityMapEntry(
                audio_file="audio_junk.m4a",
                match_method="none",
                match_confidence=0.0,
                is_unmatched=True,
            ),
        ],
        roster_students_without_audio=["Riya Sen"],
    )


def make_args(
    tmp_path: Path,
    identity_map_path: Path | None = None,
    student_ids: list[str] | None = None,
    db_url: str = "postgresql://localhost/test",
) -> IngestArgs:
    return IngestArgs(
        materials_dir=tmp_path,
        class_name="CS101",
        db_url=db_url,
        identity_map_path=identity_map_path,
        student_ids=student_ids or [],
    )


def write_identity_map(tmp_path: Path) -> Path:
    path = tmp_path / "identity_map.json"
    path.write_text(make_identity_map().model_dump_json(), encoding="utf-8")
    return path


def test_enrolled_students_skips_teacher_and_unmatched() -> None:
    students = enrolled_students(make_identity_map())
    assert [s.student_id for s in students] == ["2301", "kabir_rao"]


def test_enrolled_students_id_from_roll_no_else_name_slug() -> None:
    students = {s.student_id: s.student_name for s in enrolled_students(make_identity_map())}
    assert students["2301"] == "Anshi Verma"
    assert students["kabir_rao"] == "Kabir Rao"


def test_enrolled_students_dedupes_repeated_roll_no() -> None:
    identity_map = make_identity_map()
    identity_map.entries.append(
        IdentityMapEntry(
            audio_file="audio_2301_again.m4a",
            matched_name="Anshi Verma",
            matched_roll_no="2301",
            match_method="roll_no",
            match_confidence=1.0,
        )
    )
    students = enrolled_students(identity_map)
    assert [s.student_id for s in students] == ["2301", "kabir_rao"]


def test_resolve_students_from_identity_map_file(tmp_path: Path) -> None:
    identity_path = write_identity_map(tmp_path)
    args = make_args(tmp_path, identity_map_path=identity_path)
    students = resolve_students(args)
    assert [s.student_id for s in students] == ["2301", "kabir_rao"]


def test_resolve_students_warns_for_roster_students_without_audio(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    identity_path = write_identity_map(tmp_path)
    args = make_args(tmp_path, identity_map_path=identity_path)
    with caplog.at_level(logging.WARNING):
        resolve_students(args)
    assert "Riya Sen" in caplog.text


def test_resolve_students_explicit_ids_only(tmp_path: Path) -> None:
    args = make_args(tmp_path, student_ids=["2301", "2301", "  ", "kabir_rao"])
    students = resolve_students(args)
    assert [s.student_id for s in students] == ["2301", "kabir_rao"]
    assert students[0].student_name == "2301"


def test_resolve_students_merges_explicit_ids_without_duplicates(tmp_path: Path) -> None:
    identity_path = write_identity_map(tmp_path)
    args = make_args(tmp_path, identity_map_path=identity_path, student_ids=["2301", "9999"])
    students = resolve_students(args)
    assert [s.student_id for s in students] == ["2301", "kabir_rao", "9999"]


def test_material_records_have_material_type_and_provenance() -> None:
    records = chunk_material_blocks(
        [("supply.pptx", GOOD_BLOCK)],
        student_id="2301",
        student_name="Anshi Verma",
        class_name="CS101",
    )
    assert len(records) == 1
    record = records[0]
    assert record.chunk_type == "material"
    assert record.metadata == {"source_file": "supply.pptx"}
    assert record.speaker == "material"
    assert record.student_id == "2301"
    assert record.class_name == "CS101"
    assert record.text == GOOD_BLOCK


def test_material_records_split_long_blocks() -> None:
    records = chunk_material_blocks(
        [("module.pdf", LONG_BLOCK)],
        student_id="2301",
        student_name="Anshi Verma",
        class_name="CS101",
    )
    assert len(records) > 1
    assert all(len(record.text) <= 700 for record in records)
    assert all(record.metadata["source_file"] == "module.pdf" for record in records)


def test_material_chunk_ids_stable_and_scoped_per_student() -> None:
    blocks = [("supply.pptx", GOOD_BLOCK)]
    first = chunk_material_blocks(
        blocks, student_id="2301", student_name="Anshi Verma", class_name="CS101"
    )
    again = chunk_material_blocks(
        blocks, student_id="2301", student_name="Anshi Verma", class_name="CS101"
    )
    other = chunk_material_blocks(
        blocks, student_id="kabir_rao", student_name="Kabir Rao", class_name="CS101"
    )
    assert first[0].id == again[0].id
    assert first[0].id != other[0].id


def test_build_material_records_fans_out_per_student() -> None:
    students = [
        MaterialStudent(student_id="2301", student_name="Anshi Verma"),
        MaterialStudent(student_id="kabir_rao", student_name="Kabir Rao"),
    ]
    records = build_material_records([("supply.pptx", GOOD_BLOCK)], students, "CS101")
    assert [record.student_id for record in records] == ["2301", "kabir_rao"]
    assert all(record.chunk_type == "material" for record in records)


def test_embed_records_deduped_encodes_each_distinct_text_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class FakeVector:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def tolist(self) -> list[float]:
            return self._values

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def encode(
            self, texts: list[str], show_progress_bar: bool = False
        ) -> list[FakeVector]:
            calls.append(list(texts))
            return [FakeVector([float(len(text)), 1.0]) for text in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    students = [
        MaterialStudent(student_id="2301", student_name="Anshi Verma"),
        MaterialStudent(student_id="kabir_rao", student_name="Kabir Rao"),
    ]
    records = build_material_records([("supply.pptx", GOOD_BLOCK)], students, "CS101")
    embedded = embed_records_deduped(records, "fake-model")
    assert calls == [[GOOD_BLOCK]]
    assert embedded[0].embedding == embedded[1].embedding
    assert embedded[0].embedding != []


def test_embed_records_deduped_empty_is_noop() -> None:
    assert embed_records_deduped([], "fake-model") == []


def test_run_ingest_purges_then_upserts_per_student(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("scripts.ingest_materials.embed_records_deduped", fake_embed)
    store = FakeStore()
    students = [
        MaterialStudent(student_id="2301", student_name="Anshi Verma"),
        MaterialStudent(student_id="kabir_rao", student_name="Kabir Rao"),
    ]
    records = run_ingest_materials(
        [("supply.pptx", GOOD_BLOCK)], students, "CS101", store, "fake-model"
    )
    assert store.deleted == [("CS101", "2301"), ("CS101", "kabir_rao")]
    assert [record.id for record in store.upserted] == [record.id for record in records]
    assert all(record.embedding == [0.1, 0.2, 0.3] for record in store.upserted)


def test_run_ingest_is_idempotent_replace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.ingest_materials.embed_records_deduped", fake_embed)
    students = [MaterialStudent(student_id="2301", student_name="Anshi Verma")]
    first_store = FakeStore()
    second_store = FakeStore()
    first = run_ingest_materials(
        [("supply.pptx", GOOD_BLOCK)], students, "CS101", first_store, "fake-model"
    )
    second = run_ingest_materials(
        [("supply.pptx", GOOD_BLOCK)], students, "CS101", second_store, "fake-model"
    )
    assert [record.id for record in first] == [record.id for record in second]
    assert second_store.deleted == [("CS101", "2301")]


def test_run_ingest_only_touches_material_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("scripts.ingest_materials.embed_records_deduped", fake_embed)
    store = FakeStore()
    students = [MaterialStudent(student_id="2301", student_name="Anshi Verma")]
    run_ingest_materials([("supply.pptx", GOOD_BLOCK)], students, "CS101", store, "fake-model")
    assert all(record.chunk_type == "material" for record in store.upserted)


def test_parse_args_collects_repeatable_student_ids() -> None:
    args = parse_args(
        [
            "--materials-dir", "materials/CS101",
            "--class-name", "CS101",
            "--student-id", "2301",
            "--student-id", "2302",
            "--db-url", "postgresql://localhost/test",
        ]
    )
    assert args.student_ids == ["2301", "2302"]
    assert args.db_url == "postgresql://localhost/test"
    assert args.identity_map_path is None


def test_validate_inputs_requires_materials_dir(tmp_path: Path) -> None:
    args = make_args(tmp_path / "missing", student_ids=["2301"])
    with pytest.raises(ValueError, match="Materials folder not found"):
        validate_inputs(args)


def test_validate_inputs_requires_db_url(tmp_path: Path) -> None:
    args = make_args(tmp_path, student_ids=["2301"], db_url="")
    with pytest.raises(ValueError, match="Database URL"):
        validate_inputs(args)


def test_validate_inputs_requires_student_source(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    with pytest.raises(ValueError, match="--identity-map or at least one --student-id"):
        validate_inputs(args)


def test_validate_inputs_requires_existing_identity_map(tmp_path: Path) -> None:
    args = make_args(tmp_path, identity_map_path=tmp_path / "missing.json")
    with pytest.raises(ValueError, match="Identity map not found"):
        validate_inputs(args)


def test_validate_inputs_requires_class_name(tmp_path: Path) -> None:
    args = make_args(tmp_path, student_ids=["2301"])
    args = args.model_copy(update={"class_name": "  "})
    with pytest.raises(ValueError, match="Class name"):
        validate_inputs(args)


def test_main_end_to_end_with_mocked_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    materials = tmp_path / "materials" / "CS101"
    materials.mkdir(parents=True)
    (materials / "notes.txt").write_text(GOOD_BLOCK, encoding="utf-8")
    identity_path = write_identity_map(tmp_path)
    review_path = tmp_path / "material_chunk_review.csv"

    fake_store = FakeStore()
    monkeypatch.setattr(
        "scripts.utils.pg_store.connect_pg_store", lambda db_url: fake_store
    )
    monkeypatch.setattr("scripts.ingest_materials.embed_records_deduped", fake_embed)

    main(
        [
            "--materials-dir", str(materials),
            "--class-name", "CS101",
            "--identity-map", str(identity_path),
            "--db-url", "postgresql://localhost/test",
            "--chunk-review", str(review_path),
        ]
    )

    assert fake_store.closed is True
    assert len(fake_store.upserted) == 2
    assert {record.student_id for record in fake_store.upserted} == {"2301", "kabir_rao"}
    assert review_path.exists()
    assert "Ingested 2 material chunks" in capsys.readouterr().out


def test_main_errors_when_no_material_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "empty.txt").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--materials-dir", str(materials),
                "--class-name", "CS101",
                "--student-id", "2301",
                "--db-url", "postgresql://localhost/test",
            ]
        )
    assert excinfo.value.code == 2
