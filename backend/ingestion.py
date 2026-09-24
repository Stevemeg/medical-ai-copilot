"""Validated, atomic PDF extraction for registered local sources."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable

from backend.source_registry import IntegrityMismatch, RegistryError, SourceRegistry, sha256_file

LOG = logging.getLogger(__name__)


class UnsupportedSource(RegistryError):
    """The registered source format cannot be parsed."""


class ParserFailure(RegistryError):
    """A source could not be converted to valid page text."""


def extract_registered_pdf(
    registry: SourceRegistry,
    version_id: str,
    output_dir: Path,
    parser: Callable[[Path], list[dict]],
) -> Path:
    try:
        version = registry.versions[version_id]
    except KeyError as exc:
        raise RegistryError("Unregistered source version") from exc
    source = registry.source_path(version)
    if source.suffix.lower() != ".pdf":
        raise UnsupportedSource("Only PDF extraction is supported")
    if not source.is_file() or sha256_file(source) != version.sha256:
        raise IntegrityMismatch(f"Source checksum mismatch for {version_id}")
    target = output_dir / (source.stem + ".json")
    sidecar = target.with_suffix(".provenance.json")
    if target.is_file() and sidecar.is_file():
        try:
            recorded = json.loads(sidecar.read_text(encoding="utf-8"))
            if (
                recorded.get("version_id") == version_id
                and recorded.get("source_sha256") == version.sha256
                and recorded.get("processed_sha256") == sha256_file(target)
                and recorded.get("parser_version") == version.parser_version
            ):
                LOG.info("ingestion_unchanged document_id=%s version_id=%s", version.document_id, version_id)
                return target
        except (OSError, ValueError):
            pass
    try:
        pages = parser(source)
        if (
            not isinstance(pages, list)
            or not pages
            or any(
                not isinstance(row, dict)
                or not isinstance(row.get("page"), int)
                or row["page"] < 1
                or not isinstance(row.get("text"), str)
                or not row["text"].strip()
                for row in pages
            )
        ):
            raise ValueError("Parser returned invalid or empty pages")
    except Exception as exc:
        LOG.error("ingestion_failed document_id=%s version_id=%s", version.document_id, version_id)
        raise ParserFailure(f"PDF parser failed for {version_id}") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    os.close(fd)
    temporary = Path(temp_name)
    try:
        temporary.write_text(json.dumps(pages, indent=2, ensure_ascii=False), encoding="utf-8")
        processed_hash = sha256_file(temporary)
        temporary.replace(target)
        sidecar.write_text(
            json.dumps(
                {
                    "version_id": version_id,
                    "source_sha256": version.sha256,
                    "processed_sha256": processed_hash,
                    "parser_version": version.parser_version,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        temporary.unlink(missing_ok=True)
    LOG.info("ingestion_complete document_id=%s version_id=%s pages=%d", version.document_id, version_id, len(pages))
    return target
