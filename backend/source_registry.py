"""Offline source registry, integrity checks, and retrieval policy."""

import hashlib
import json
import logging
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from backend.knowledge_models import (
    IngestionStatus,
    KnowledgeDocument,
    KnowledgeDocumentVersion,
    Lifecycle,
    RetrievalPolicy,
    SourceType,
)

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "data" / "source_registry.json"


class RegistryError(ValueError):
    """The registry is malformed or internally inconsistent."""


class IntegrityMismatch(RegistryError):
    """A local source no longer matches its registered exact bytes."""


class StaleArtifact(RegistryError):
    """An index or extracted artifact needs rebuilding."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _date(value: str | None) -> None:
    if value is not None:
        date.fromisoformat(value)


def _url(value: str | None) -> None:
    if value is not None:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.netloc or " " in value:
            raise RegistryError("Source URL must be an absolute HTTPS URL")


class SourceRegistry:
    def __init__(self, path: Path = REGISTRY_PATH, root: Path = ROOT, *, verify_files: bool = True):
        self.path = path
        self.root = root.resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload["sources"]
            if not isinstance(rows, list):
                raise TypeError("sources must be a list")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RegistryError("Cannot load source registry") from exc
        self.documents: dict[str, KnowledgeDocument] = {}
        self.versions: dict[str, KnowledgeDocumentVersion] = {}
        self.by_legacy_source: dict[str, KnowledgeDocumentVersion] = {}
        source_paths: set[str] = set()
        for row in rows:
            try:
                doc = KnowledgeDocument(
                    document_id=row["document_id"],
                    canonical_title=row["canonical_title"],
                    publisher=row["publisher"],
                    source_type=SourceType(row["source_type"]),
                    jurisdiction=row["jurisdiction"],
                    guideline_code=row.get("guideline_code"),
                    canonical_source_url=row.get("canonical_source_url"),
                    license=row.get("license"),
                )
                _url(doc.canonical_source_url)
                if not all((doc.document_id, doc.canonical_title, doc.publisher, doc.jurisdiction)):
                    raise RegistryError("Required document field is empty")
                if doc.document_id in self.documents:
                    raise RegistryError("Duplicate document ID")
                self.documents[doc.document_id] = doc
                for item in row["versions"]:
                    version = KnowledgeDocumentVersion(
                        version_id=item["version_id"],
                        document_id=doc.document_id,
                        version_label=item.get("version_label"),
                        published_at=item.get("published_at"),
                        updated_at=item.get("updated_at"),
                        retrieved_at=item.get("retrieved_at"),
                        valid_from=item.get("valid_from"),
                        valid_until=item.get("valid_until"),
                        status=Lifecycle(item["status"]),
                        supersedes_version_id=item.get("supersedes_version_id"),
                        superseded_by_version_id=item.get("superseded_by_version_id"),
                        sha256=item["sha256"],
                        parser_version=item["parser_version"],
                        ingestion_status=IngestionStatus(item["ingestion_status"]),
                        source_path=item["source_path"],
                        legacy_source=item["legacy_source"],
                    )
                    for field in (
                        version.published_at,
                        version.updated_at,
                        version.retrieved_at,
                        version.valid_from,
                        version.valid_until,
                    ):
                        _date(field)
                    if not re.fullmatch(r"[a-f0-9]{64}", version.sha256):
                        raise RegistryError("Invalid SHA-256")
                    if version.version_id in self.versions or version.legacy_source in self.by_legacy_source:
                        raise RegistryError("Duplicate version ID or legacy source")
                    if version.source_path in source_paths:
                        raise RegistryError("Duplicate local source path")
                    source_paths.add(version.source_path)
                    if version.status is Lifecycle.CURRENT and version.superseded_by_version_id:
                        raise RegistryError("Current version cannot be superseded")
                    self.versions[version.version_id] = version
                    self.by_legacy_source[version.legacy_source] = version
            except (KeyError, TypeError, ValueError) as exc:
                raise RegistryError(f"Invalid source registry entry: {exc}") from exc
        for version in self.versions.values():
            for link in (version.supersedes_version_id, version.superseded_by_version_id):
                if link and (
                    link == version.version_id
                    or link not in self.versions
                    or self.versions[link].document_id != version.document_id
                ):
                    raise RegistryError("Invalid supersession link")
            if (
                version.superseded_by_version_id
                and self.versions[version.superseded_by_version_id].supersedes_version_id != version.version_id
            ):
                raise RegistryError("Supersession links must be reciprocal")
            if (
                version.supersedes_version_id
                and self.versions[version.supersedes_version_id].superseded_by_version_id != version.version_id
            ):
                raise RegistryError("Supersession links must be reciprocal")
            if verify_files:
                source = self.source_path(version)
                if not source.is_file():
                    raise RegistryError(f"Missing local source for {version.version_id}")
                if sha256_file(source) != version.sha256:
                    raise IntegrityMismatch(f"Checksum mismatch for {version.version_id}")
                LOG.info(
                    "source_validated document_id=%s version_id=%s status=%s",
                    version.document_id,
                    version.version_id,
                    version.status,
                )

    def source_path(self, version: KnowledgeDocumentVersion) -> Path:
        path = (self.root / version.source_path).resolve()
        if not path.is_relative_to(self.root):
            raise RegistryError("Source path escapes repository")
        return path

    def resolve(self, chunk: dict) -> tuple[KnowledgeDocument, KnowledgeDocumentVersion]:
        version_id = chunk.get("version_id")
        if not version_id and chunk.get("source") in self.by_legacy_source:
            version_id = self.by_legacy_source[chunk["source"]].version_id
        if version_id not in self.versions:
            raise RegistryError("Unregistered evidence chunk")
        version = self.versions[version_id]
        doc = self.documents[version.document_id]
        if chunk.get("document_id", doc.document_id) != doc.document_id:
            raise RegistryError("Evidence document/version mismatch")
        return doc, version

    def eligible(self, chunk: dict, policy: RetrievalPolicy) -> bool:
        doc, version = self.resolve(chunk)
        if policy is RetrievalPolicy.CURRENT_CLINICAL:
            return (
                doc.source_type is SourceType.CLINICAL_GUIDELINE
                and version.status is Lifecycle.CURRENT
                and version.ingestion_status is IngestionStatus.INGESTED
            )
        if policy is RetrievalPolicy.HISTORICAL_ONLY:
            return (
                version.status in (Lifecycle.SUPERSEDED, Lifecycle.HISTORICAL)
                and version.ingestion_status is IngestionStatus.INGESTED
            )
        if policy is RetrievalPolicy.HISTORICAL_ALLOWED:
            return (
                version.status in (Lifecycle.CURRENT, Lifecycle.SUPERSEDED, Lifecycle.HISTORICAL)
                and version.ingestion_status is IngestionStatus.INGESTED
            )
        if policy is RetrievalPolicy.REFERENCE:
            return (
                doc.source_type in (SourceType.TEXTBOOK, SourceType.PATIENT_EDUCATION)
                and version.status is not Lifecycle.WITHDRAWN
                and version.ingestion_status is IngestionStatus.INGESTED
            )
        raise RegistryError("Unsupported retrieval policy")

    def enrich(self, chunk: dict) -> dict:
        doc, version = self.resolve(chunk)
        result = dict(chunk)
        result.update(
            document_id=doc.document_id,
            version_id=version.version_id,
            canonical_title=doc.canonical_title,
            publisher=doc.publisher,
            source_type=doc.source_type.value,
            jurisdiction=doc.jurisdiction,
            lifecycle_status=version.status.value,
            source_url=doc.canonical_source_url,
            published_at=version.published_at,
            updated_at=version.updated_at,
            source_sha256=version.sha256,
            section=chunk.get("section"),
            recommendation_id=chunk.get("recommendation_id"),
        )
        return result

    def display_name(self, version: KnowledgeDocumentVersion) -> str:
        document = self.documents[version.document_id]
        prefix = f"{document.publisher} {document.guideline_code}" if document.guideline_code else document.publisher
        return f"{prefix} — {document.canonical_title}"


if __name__ == "__main__":
    loaded = SourceRegistry()
    print(f"Validated {len(loaded.documents)} documents and {len(loaded.versions)} source checksums")
