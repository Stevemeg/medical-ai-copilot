"""Reproducible FAISS artifact manifests; legacy indexes fail closed."""

import json
from pathlib import Path

from backend.source_registry import StaleArtifact, SourceRegistry, sha256_file
from embeddings.recommendation_extractor import EXTRACTOR_VERSION

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNKER_VERSION = "sentence-token-page-v1:cl100k_base:600:100"
NORMALIZATION = "l2_unit"
SIMILARITY_METRIC = "cosine_inner_product"
INDEX_TYPE = "IndexFlatIP"
EMBEDDING_DIMENSION = 384
EVIDENCE_SCHEMA_VERSION = 1


def _processed_snapshot(registry: SourceRegistry) -> list[dict]:
    rows = []
    for version in registry.versions.values():
        processed = registry.root / "data/processed" / (Path(version.source_path).stem + ".json")
        sidecar = processed.with_suffix(".provenance.json")
        try:
            recorded = json.loads(sidecar.read_text(encoding="utf-8"))
            processed_hash = sha256_file(processed)
        except (OSError, ValueError) as exc:
            raise StaleArtifact("Processed source provenance missing or corrupt") from exc
        if (
            recorded.get("version_id") != version.version_id
            or recorded.get("source_sha256") != version.sha256
            or recorded.get("processed_sha256") != processed_hash
            or recorded.get("parser_version") != version.parser_version
        ):
            raise StaleArtifact("Processed source provenance is stale")
        rows.append({"version_id": version.version_id, "processed_sha256": processed_hash})
    return sorted(rows, key=lambda row: row["version_id"])


def write_chunks_manifest(chunks_path: Path, registry: SourceRegistry) -> None:
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if {registry.resolve(c)[1].version_id for c in chunks} != set(registry.versions):
        raise StaleArtifact("Chunks do not cover every registered source version")
    chunks_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "chunks_sha256": sha256_file(chunks_path),
                "registry_sha256": sha256_file(registry.path),
                "chunker_version": CHUNKER_VERSION,
                "processed_sources": _processed_snapshot(registry),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def validate_chunks_manifest(chunks_path: Path, registry: SourceRegistry) -> None:
    try:
        manifest = json.loads(chunks_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("chunks_sha256") != sha256_file(chunks_path)
            or manifest.get("registry_sha256") != sha256_file(registry.path)
            or manifest.get("chunker_version") != CHUNKER_VERSION
            or manifest.get("processed_sources") != _processed_snapshot(registry)
        ):
            raise StaleArtifact("Chunk provenance differs from current sources")
    except (OSError, ValueError, TypeError) as exc:
        raise StaleArtifact("Chunk provenance missing or corrupt") from exc


def write_manifest(index_path: Path, metadata_path: Path, registry: SourceRegistry) -> None:
    chunks = json.loads(metadata_path.read_text(encoding="utf-8"))
    version_ids = sorted({registry.resolve(chunk)[1].version_id for chunk in chunks})
    manifest = {
        "schema_version": 2,
        "index_sha256": sha256_file(index_path),
        "metadata_sha256": sha256_file(metadata_path),
        "registry_sha256": sha256_file(registry.path),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "normalization": NORMALIZATION,
        "similarity_metric": SIMILARITY_METRIC,
        "index_type": INDEX_TYPE,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "recommendation_extractor_version": EXTRACTOR_VERSION,
        "evidence_units_sha256": sha256_file(registry.root / "data/evidence_units.json")
        if (registry.root / "data/evidence_units.json").exists()
        else None,
        "chunker_version": CHUNKER_VERSION,
        "document_versions": [
            {
                "version_id": ident,
                "source_sha256": registry.versions[ident].sha256,
                "parser_version": registry.versions[ident].parser_version,
            }
            for ident in version_ids
        ],
        "chunk_count": len(chunks),
    }
    index_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def validate_manifest(index_path: Path, metadata_path: Path, registry: SourceRegistry) -> dict:
    manifest_path = index_path.with_suffix(".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "index_sha256": sha256_file(index_path),
            "metadata_sha256": sha256_file(metadata_path),
            "registry_sha256": sha256_file(registry.path),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "normalization": NORMALIZATION,
            "similarity_metric": SIMILARITY_METRIC,
            "index_type": INDEX_TYPE,
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "recommendation_extractor_version": EXTRACTOR_VERSION,
            "evidence_units_sha256": sha256_file(registry.root / "data/evidence_units.json")
            if (registry.root / "data/evidence_units.json").exists()
            else None,
            "chunker_version": CHUNKER_VERSION,
            "chunk_count": len(chunks),
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise StaleArtifact("Index manifest differs from current artifacts or registry")
        actual_versions = sorted({registry.resolve(c)[1].version_id for c in chunks})
        recorded_versions = sorted(row["version_id"] for row in manifest["document_versions"])
        if actual_versions != recorded_versions:
            raise StaleArtifact("Index version membership differs from manifest")
        for row in manifest["document_versions"]:
            version = registry.versions[row["version_id"]]
            if row["source_sha256"] != version.sha256 or row["parser_version"] != version.parser_version:
                raise StaleArtifact("Index source provenance differs from registry")
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StaleArtifact("Index provenance missing or corrupt; rebuild indexes") from exc


if __name__ == "__main__":
    from backend.source_registry import ROOT

    loaded = SourceRegistry()
    validate_chunks_manifest(ROOT / "data/chunks.json", loaded)
    print("Validated chunk provenance")
    for name in ("clinical", "anatomy"):
        base = ROOT / "data/vector_store"
        validate_manifest(base / f"{name}_faiss.index", base / f"{name}_metadata.json", loaded)
        print(f"Validated {name} index provenance")
