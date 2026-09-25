"""Conservative extraction of NICE-style numbered recommendation paragraphs."""

import json
import re
from pathlib import Path

from backend.evidence_models import EvidenceUnit, EvidenceUnitType
from backend.knowledge_models import SourceType
from backend.source_registry import SourceRegistry

EXTRACTOR_VERSION = "nice-numbered-action-v3"
ROOT = Path(__file__).resolve().parent.parent
# Number alone is insufficient: require a paragraph-start action verb and a
# neighboring sequential recommendation in the same local guideline section.
MARKER = re.compile(
    r"(?<![\w.])(?P<id>\d{1,2}\.\d{1,2}\.\d{1,3})\s+"
    r"(?P<verb>Offer|Ask|Consider|Do not|Discuss|Assess|Measure|Refer|Ensure|Advise|Use|"
    r"Record|Review|Tell|Explain|Provide|Perform|Check|Monitor|Repeat|Start|Stop|"
    r"Investigate|Recommend|Encourage|Inform|Take|Give|For|If|When)\b",
    re.IGNORECASE,
)
DATE_NOTE = re.compile(r"\[(?:\d{4}|[A-Za-z]+ \d{4})[^\]]{0,35}\]")


def _number(identifier: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in identifier.split("."))  # type: ignore[return-value]


def extract_recommendations(pages: list[dict], document_id: str, version_id: str) -> list[dict]:
    """Return only structurally corroborated recommendations, with page spans."""
    joined = ""
    offsets: list[tuple[int, int, int]] = []
    for page in pages:
        start = len(joined)
        joined += " " + re.sub(r"\s+", " ", page["text"])
        offsets.append((start, len(joined), page["page"]))
    matches = list(MARKER.finditer(joined))
    rows: list[dict] = []
    for i, match in enumerate(matches):
        # PDF extraction may flatten a table caption or a prose reference
        # onto the same line as an apparent action verb. Neither establishes
        # a recommendation boundary.
        prefix = joined[max(0, match.start() - 18) : match.start()]
        if re.search(r"\b(?:table|figure|section|recommendation|page)\s+$", prefix, re.IGNORECASE):
            continue
        current = _number(match.group("id"))
        neighbors = [matches[j] for j in (i - 1, i + 1) if 0 <= j < len(matches)]
        sequential = any(
            _number(other.group("id"))[:2] == current[:2] and abs(_number(other.group("id"))[2] - current[2]) == 1
            for other in neighbors
        )
        if not sequential:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(joined)
        passage = joined[match.end("id") : end].strip()
        dated = DATE_NOTE.search(passage)
        if dated and passage[dated.end() :].strip() and re.match(r"^[A-Z][a-z]", passage[dated.end() :].strip()):
            passage = passage[: dated.end()].strip()
            end = match.end("id") + len(passage)
        # Long intervals often contain headers/tables rather than one paragraph.
        if not (25 <= len(passage) <= 2500):
            continue
        pages_seen = [number for start, stop, number in offsets if start < end and stop > match.start()]
        if not pages_seen:
            continue
        rows.append(
            {
                "recommendation_id": match.group("id"),
                "document_id": document_id,
                "version_id": version_id,
                "page_start": min(pages_seen),
                "page_end": max(pages_seen),
                "section": ".".join(match.group("id").split(".")[:2]),
                "text": passage,
                "extractor_version": EXTRACTOR_VERSION,
            }
        )
    # Duplicate printed IDs are ambiguous: keep neither.
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["recommendation_id"]] = counts.get(row["recommendation_id"], 0) + 1
    return [row for row in rows if counts[row["recommendation_id"]] == 1]


def build_units(registry: SourceRegistry, chunks: list[dict], processed_dir: Path) -> tuple[list[EvidenceUnit], dict]:
    units: list[EvidenceUnit] = []
    for chunk in chunks:
        c = registry.enrich(chunk)
        kind = (
            EvidenceUnitType.REFERENCE_CHUNK
            if c["source_type"] in (SourceType.TEXTBOOK.value, SourceType.PATIENT_EDUCATION.value)
            else EvidenceUnitType.CONTEXT_CHUNK
        )
        units.append(
            EvidenceUnit(
                evidence_unit_id=c["chunk_id"],
                unit_type=kind,
                chunk_id=c["chunk_id"],
                source_chunk_ids=[c["chunk_id"]],
                document_id=c["document_id"],
                version_id=c["version_id"],
                text=c["text"],
                section=c.get("section"),
                page_start=c.get("page_start"),
                page_end=c.get("page_end"),
                source_type=c["source_type"],
                jurisdiction=c["jurisdiction"],
                lifecycle_status=c["lifecycle_status"],
                canonical_source_url=c.get("source_url"),
                publisher=c["publisher"],
                canonical_title=c["canonical_title"],
                published_at=c.get("published_at"),
                updated_at=c.get("updated_at"),
            )
        )
    extracted = 0
    ambiguous = 0
    document_status = []
    for version in registry.versions.values():
        doc = registry.documents[version.document_id]
        if doc.source_type is not SourceType.CLINICAL_GUIDELINE or doc.publisher != "NICE":
            continue
        path = processed_dir / (Path(version.source_path).stem + ".json")
        pages = json.loads(path.read_text(encoding="utf-8"))
        recommendations = extract_recommendations(pages, doc.document_id, version.version_id)
        extracted += len(recommendations)
        candidates = sum(len(MARKER.findall(page["text"])) for page in pages)
        ambiguous += max(0, candidates - len(recommendations))
        document_status.append(
            {
                "document_id": doc.document_id,
                "version_id": version.version_id,
                "extraction_status": "verified_structure" if recommendations else "ambiguous" if candidates else "none",
                "verified_count": len(recommendations),
                "ambiguous_candidates_rejected": max(0, candidates - len(recommendations)),
            }
        )
        for rec in recommendations:
            sources = [
                c["chunk_id"]
                for c in chunks
                if c["version_id"] == version.version_id
                and c["page_start"] <= rec["page_end"]
                and c["page_end"] >= rec["page_start"]
            ]
            if not sources:
                continue
            units.append(
                EvidenceUnit(
                    evidence_unit_id=f"{version.version_id}:rec:{rec['recommendation_id']}",
                    unit_type=EvidenceUnitType.RECOMMENDATION,
                    recommendation_id=rec["recommendation_id"],
                    source_chunk_ids=sources,
                    document_id=doc.document_id,
                    version_id=version.version_id,
                    text=rec["text"],
                    section=rec["section"],
                    page_start=rec["page_start"],
                    page_end=rec["page_end"],
                    source_type=doc.source_type,
                    jurisdiction=doc.jurisdiction,
                    lifecycle_status=version.status,
                    canonical_source_url=doc.canonical_source_url,
                    publisher=doc.publisher,
                    canonical_title=doc.canonical_title,
                    published_at=version.published_at,
                    updated_at=version.updated_at,
                    extractor_version=EXTRACTOR_VERSION,
                )
            )
    return units, {
        "extractor_version": EXTRACTOR_VERSION,
        "documents_processed": len(document_status),
        "recommendations_extracted": extracted,
        "ambiguous_candidates_rejected": ambiguous,
        "documents": document_status,
    }


if __name__ == "__main__":
    registry = SourceRegistry()
    chunks = json.loads((ROOT / "data/chunks.json").read_text(encoding="utf-8"))
    units, report = build_units(registry, chunks, ROOT / "data/processed")
    (ROOT / "data/evidence_units.json").write_text(
        json.dumps([unit.model_dump(mode="json") for unit in units], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (ROOT / "data/recommendation_extraction_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(report | {"total_units": len(units)})
