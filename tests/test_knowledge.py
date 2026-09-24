import hashlib
import json
from pathlib import Path

import pytest

from backend.ingestion import ParserFailure, UnsupportedSource, extract_registered_pdf
from backend.index_provenance import (
    validate_chunks_manifest,
    validate_manifest,
    write_chunks_manifest,
    write_manifest,
)
from backend.knowledge_models import RetrievalPolicy
from backend.source_registry import (
    IntegrityMismatch,
    RegistryError,
    SourceRegistry,
    StaleArtifact,
    sha256_file,
)


@pytest.fixture
def registry_data(tmp_path: Path):
    path = tmp_path / "source.pdf"
    path.write_bytes(b"small exact source")
    version = {
        "version_id": "guideline-2020",
        "version_label": "2020",
        "published_at": "2020-01-01",
        "updated_at": None,
        "retrieved_at": None,
        "valid_from": None,
        "valid_until": None,
        "status": "current",
        "supersedes_version_id": None,
        "superseded_by_version_id": None,
        "sha256": sha256_file(path),
        "parser_version": "test-parser-v1",
        "ingestion_status": "ingested",
        "source_path": "source.pdf",
        "legacy_source": "source.pdf.txt",
    }
    source = {
        "document_id": "guideline",
        "canonical_title": "Small guideline",
        "publisher": "Test publisher",
        "source_type": "clinical_guideline",
        "jurisdiction": "UK",
        "guideline_code": "T1",
        "canonical_source_url": "https://example.org/guideline",
        "license": None,
        "versions": [version],
    }
    registry_path = tmp_path / "registry.json"

    def save(rows):
        registry_path.write_text(json.dumps({"sources": rows}), encoding="utf-8")
        return registry_path

    save([source])
    return tmp_path, source, version, save


def test_real_registry_and_corpus_provenance():
    registry = SourceRegistry()
    assert not registry.eligible({"version_id": "nice-ng19-2019-10-11"}, RetrievalPolicy.CURRENT_CLINICAL)
    assert registry.eligible({"version_id": "nice-ng19-2019-10-11"}, RetrievalPolicy.HISTORICAL_ONLY)
    validate_chunks_manifest(registry.root / "data/chunks.json", registry)
    chunks = json.loads((registry.root / "data/chunks.json").read_text(encoding="utf-8"))
    assert len(registry.versions) == 9
    assert {c["source"] for c in chunks} == set(registry.by_legacy_source)
    for chunk in chunks:
        document, version = registry.resolve(chunk)
        assert document.publisher and version.sha256 == chunk["source_sha256"]
        assert chunk["source_type"] == document.source_type.value
        assert hashlib.sha256(chunk["text"].encode()).hexdigest() == chunk["content_sha256"]
        assert chunk["page_start"] >= 1


def test_valid_and_optional_unknown_fields(registry_data):
    root, source, version, save = registry_data
    version["updated_at"] = None
    source["license"] = None
    registry = SourceRegistry(save([source]), root)
    assert registry.resolve({"source": "source.pdf.txt"})[0].canonical_title == "Small guideline"


@pytest.mark.parametrize(
    "change",
    [
        lambda s, v: s.update(source_type="unrecognized"),
        lambda s, v: v.update(status="newest"),
        lambda s, v: v.update(published_at="2020-99-99"),
        lambda s, v: s.update(canonical_source_url="not-a-url"),
        lambda s, v: v.update(superseded_by_version_id="missing"),
        lambda s, v: v.update(status="current", superseded_by_version_id="guideline-2021"),
        lambda s, v: v.update(ingestion_status="unverified"),
    ],
)
def test_invalid_registry_fields(registry_data, change):
    root, source, version, save = registry_data
    change(source, version)
    with pytest.raises(RegistryError):
        SourceRegistry(save([source]), root)


def test_malformed_and_duplicate_registry(registry_data):
    root, source, version, save = registry_data
    path = save([source, source])
    with pytest.raises(RegistryError, match="Duplicate document"):
        SourceRegistry(path, root)
    other = json.loads(json.dumps(source))
    other["document_id"] = "other"
    with pytest.raises(RegistryError, match="Duplicate version"):
        SourceRegistry(save([source, other]), root)
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(RegistryError, match="Cannot load"):
        SourceRegistry(path, root)


def test_missing_and_changed_source(registry_data):
    root, source, version, save = registry_data
    path = save([source])
    original = sha256_file(root / "source.pdf")
    assert sha256_file(root / "source.pdf") == original
    (root / "source.pdf").write_bytes(b"changed source")
    assert sha256_file(root / "source.pdf") != original
    with pytest.raises(IntegrityMismatch):
        SourceRegistry(path, root)
    (root / "source.pdf").unlink()
    with pytest.raises(RegistryError, match="Missing local"):
        SourceRegistry(path, root)


def test_lifecycle_and_source_type_boundaries(registry_data):
    root, source, version, save = registry_data
    rows = []
    for ident, kind, status in [
        ("current", "clinical_guideline", "current"),
        ("superseded", "clinical_guideline", "superseded"),
        ("historical", "public_health_report", "historical"),
        ("withdrawn", "clinical_guideline", "withdrawn"),
        ("unknown", "clinical_guideline", "unknown"),
        ("textbook", "textbook", "unknown"),
        ("quarantined", "clinical_guideline", "current"),
    ]:
        row = json.loads(json.dumps(source))
        row["document_id"] = ident
        row["source_type"] = kind
        row["versions"][0]["version_id"] = ident + "-v1"
        row["versions"][0]["legacy_source"] = ident + ".txt"
        (root / f"{ident}.pdf").write_bytes((root / "source.pdf").read_bytes())
        row["versions"][0]["source_path"] = f"{ident}.pdf"
        row["versions"][0]["status"] = status
        if ident == "quarantined":
            row["versions"][0]["ingestion_status"] = "quarantined"
        rows.append(row)
    registry = SourceRegistry(save(rows), root)

    def eligible(ident, policy):
        return registry.eligible({"version_id": ident + "-v1"}, policy)

    assert eligible("current", RetrievalPolicy.CURRENT_CLINICAL)
    assert all(
        not eligible(ident, RetrievalPolicy.CURRENT_CLINICAL)
        for ident in ("superseded", "historical", "withdrawn", "unknown", "textbook", "quarantined")
    )
    assert eligible("superseded", RetrievalPolicy.HISTORICAL_ONLY)
    assert eligible("historical", RetrievalPolicy.HISTORICAL_ONLY)
    assert not eligible("withdrawn", RetrievalPolicy.HISTORICAL_ALLOWED)
    assert eligible("textbook", RetrievalPolicy.REFERENCE)
    assert not eligible("textbook", RetrievalPolicy.HISTORICAL_ALLOWED)


def test_manifest_staleness(registry_data):
    root, source, version, save = registry_data
    registry = SourceRegistry(save([source]), root)
    index = root / "tiny.index"
    metadata = root / "tiny.json"
    index.write_bytes(b"vectors")
    metadata.write_text(json.dumps([{"version_id": version["version_id"], "text": "a"}]), encoding="utf-8")
    write_manifest(index, metadata, registry)
    assert validate_manifest(index, metadata, registry)["chunk_count"] == 1
    index.write_bytes(b"different vectors")
    with pytest.raises(StaleArtifact):
        validate_manifest(index, metadata, registry)


def test_changed_processed_source_invalidates_chunks(registry_data):
    root, source, version, save = registry_data
    registry = SourceRegistry(save([source]), root)
    processed = root / "data/processed/source.json"
    processed.parent.mkdir(parents=True)
    processed.write_text('[{"page":1,"text":"evidence"}]', encoding="utf-8")
    processed.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "version_id": version["version_id"],
                "source_sha256": version["sha256"],
                "processed_sha256": sha256_file(processed),
                "parser_version": version["parser_version"],
            }
        ),
        encoding="utf-8",
    )
    chunks = root / "chunks.json"
    chunks.write_text(json.dumps([{"version_id": version["version_id"], "text": "evidence"}]), encoding="utf-8")
    write_chunks_manifest(chunks, registry)
    validate_chunks_manifest(chunks, registry)
    processed.write_text('[{"page":1,"text":"changed"}]', encoding="utf-8")
    with pytest.raises(StaleArtifact):
        validate_chunks_manifest(chunks, registry)


def test_ingestion_success_failure_and_integrity(registry_data):
    root, source, version, save = registry_data
    registry = SourceRegistry(save([source]), root)
    output = root / "processed"
    target = extract_registered_pdf(
        registry, version["version_id"], output, lambda _: [{"page": 1, "text": "Useful evidence"}]
    )
    assert json.loads(target.read_text(encoding="utf-8"))[0]["page"] == 1
    assert target.with_suffix(".provenance.json").exists()
    before = target.read_bytes()
    assert (
        extract_registered_pdf(
            registry,
            version["version_id"],
            output,
            lambda _: (_ for _ in ()).throw(AssertionError("unchanged source reparsed")),
        )
        == target
    )
    target.with_suffix(".provenance.json").unlink()
    with pytest.raises(ParserFailure):
        extract_registered_pdf(
            registry, version["version_id"], output, lambda _: (_ for _ in ()).throw(ValueError("bad PDF"))
        )
    assert target.read_bytes() == before
    with pytest.raises(ParserFailure):
        extract_registered_pdf(registry, version["version_id"], output, lambda _: [])
    with pytest.raises(RegistryError, match="Unregistered"):
        extract_registered_pdf(registry, "missing", output, lambda _: [])
    (root / "source.pdf").write_bytes(b"changed")
    with pytest.raises(IntegrityMismatch):
        extract_registered_pdf(registry, version["version_id"], output, lambda _: [])


def test_unsupported_source(registry_data):
    root, source, version, save = registry_data
    (root / "source.pdf").rename(root / "source.txt")
    version["source_path"] = "source.txt"
    registry = SourceRegistry(save([source]), root)
    with pytest.raises(UnsupportedSource):
        extract_registered_pdf(registry, version["version_id"], root / "out", lambda _: [])
