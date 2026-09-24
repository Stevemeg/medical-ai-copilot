"""One-time provenance adoption for the exact committed FAISS/chunk snapshot.

This validates vector alignment against the legacy chunk metadata before adding
source identities. It does not recalculate embeddings or alter the vectors.
"""

import hashlib
import json
from pathlib import Path

import faiss

from backend.index_provenance import write_chunks_manifest, write_manifest
from backend.source_registry import ROOT, RegistryError, SourceRegistry, sha256_file


def enrich(chunk: dict, registry: SourceRegistry) -> dict:
    doc, version = registry.resolve(chunk)
    result = registry.enrich(chunk)
    result["chunk_id"] = f"{version.version_id}:{chunk['chunk_id']}"
    result["content_sha256"] = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
    result["legacy_chunk_id"] = chunk["chunk_id"]
    result["section"] = None
    result["recommendation_id"] = None
    return result


def main() -> None:
    registry = SourceRegistry()
    chunks_path = ROOT / "data/chunks.json"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    vector_dir = ROOT / "data/vector_store"
    stores = []
    for name in ("clinical", "anatomy"):
        index_path = vector_dir / f"{name}_faiss.index"
        metadata_path = vector_dir / f"{name}_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if faiss.read_index(str(index_path)).ntotal != len(metadata):
            raise RegistryError("FAISS and metadata count differ")
        stores.append((index_path, metadata_path, metadata))
    combined = [chunk for _, _, metadata in stores for chunk in metadata]
    if len(combined) != len(chunks) or sorted((c["source"], c["chunk_id"], c["text"]) for c in combined) != sorted(
        (c["source"], c["chunk_id"], c["text"]) for c in chunks
    ):
        raise RegistryError("Legacy metadata is not the indexed chunk snapshot")
    for index_path, metadata_path, metadata in stores:
        metadata_path.write_text(
            json.dumps([enrich(c, registry) for c in metadata], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        write_manifest(index_path, metadata_path, registry)
    chunks_path.write_text(
        json.dumps([enrich(c, registry) for c in chunks], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for version in registry.versions.values():
        processed = ROOT / "data/processed" / (Path(version.source_path).stem + ".json")
        if not processed.is_file():
            raise RegistryError("Missing processed source")
        sidecar = processed.with_suffix(".provenance.json")
        sidecar.write_text(
            json.dumps(
                {
                    "version_id": version.version_id,
                    "source_sha256": version.sha256,
                    "processed_sha256": sha256_file(processed),
                    "parser_version": version.parser_version,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    write_chunks_manifest(chunks_path, registry)


if __name__ == "__main__":
    main()
