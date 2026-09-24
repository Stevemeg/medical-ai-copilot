"""Governed, offline recommendation snapshots for deterministic clinical rules."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import date
from enum import StrEnum
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, field_validator

from backend.source_registry import ROOT

REGISTRY_PATH = ROOT / "data" / "rule_evidence_registry.json"
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
RECOMMENDATION_ID = re.compile(r"\d+(?:\.\d+)+\Z")
NICE_HOSTS = {"www.nice.org.uk", "nice.org.uk"}


class VerificationStatus(StrEnum):
    VERIFIED_CURRENT = "verified_current"
    SUPERSEDED = "superseded"
    WITHDRAWN = "withdrawn"
    UNVERIFIED = "unverified"
    DRIFT_DETECTED = "drift_detected"


class EvidenceBasis(StrEnum):
    LOCAL_CURRENT_DOCUMENT = "local_current_document"
    AUTHORITATIVE_RECOMMENDATION_SNAPSHOT = "authoritative_recommendation_snapshot"


class RecommendationEvidenceError(ValueError):
    pass


def validate_nice_url(value: str, guideline_code: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in NICE_HOSTS
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.lower() != f"/guidance/{guideline_code.lower()}/chapter/recommendations"
    ):
        raise ValueError("NICE evidence must use its official HTTPS recommendations page")
    return value


class VerifiedRecommendationEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str
    document_id: str
    publisher: str
    guideline_code: str
    recommendation_id: str
    canonical_source_url: str
    jurisdiction: str
    verified_on: date
    verification_status: VerificationStatus
    recommendation_sha256: str
    summary: str
    source_revision_note: str | None = None
    supersedes_evidence_id: str | None = None
    verification_method: str = "official_nice_html_article_v1"

    @field_validator(
        "evidence_id", "document_id", "publisher", "guideline_code", "recommendation_id", "jurisdiction", "summary"
    )
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("required recommendation evidence field is empty")
        return value

    @field_validator("recommendation_id")
    @classmethod
    def recommendation_format(cls, value: str) -> str:
        if not RECOMMENDATION_ID.fullmatch(value):
            raise ValueError("invalid recommendation ID")
        return value

    @field_validator("recommendation_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not SHA256.fullmatch(value):
            raise ValueError("invalid recommendation SHA-256")
        return value

    @field_validator("canonical_source_url")
    @classmethod
    def absolute_https(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("recommendation URL must be absolute HTTPS")
        return value


class RecommendationEvidenceRegistry:
    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = path
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
            if payload["schema_version"] != 1 or not isinstance(payload["recommendations"], list):
                raise ValueError("unsupported recommendation registry schema")
            rows = payload["recommendations"]
            self.evidence: dict[str, VerifiedRecommendationEvidence] = {}
            for row in rows:
                entry = VerifiedRecommendationEvidence.model_validate(row)
                if entry.evidence_id in self.evidence:
                    raise ValueError("duplicate recommendation evidence ID")
                self.evidence[entry.evidence_id] = entry
            self._loaded_sha256 = hashlib.sha256(raw).hexdigest()
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise RecommendationEvidenceError("Cannot load recommendation evidence registry") from exc

    def refresh_if_changed(self) -> None:
        """Observe local governance changes during a long-running application process."""
        try:
            current_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        except OSError as exc:
            raise RecommendationEvidenceError("Recommendation evidence registry is unavailable") from exc
        if current_sha256 != self._loaded_sha256:
            refreshed = RecommendationEvidenceRegistry(self.path)
            self.evidence = refreshed.evidence
            self._loaded_sha256 = refreshed._loaded_sha256


def validate_current_recommendation(entry: VerifiedRecommendationEvidence) -> None:
    """Only the explicitly trusted current NICE recommendation path can activate a rule."""
    if entry.verification_status is not VerificationStatus.VERIFIED_CURRENT:
        raise RecommendationEvidenceError("recommendation evidence is not verified current")
    if entry.publisher != "NICE" or entry.jurisdiction != "UK":
        raise RecommendationEvidenceError("recommendation publisher or jurisdiction is not trusted")
    if not re.fullmatch(r"NG\d+", entry.guideline_code):
        raise RecommendationEvidenceError("unsupported NICE guideline code")
    if (
        entry.document_id != f"nice-{entry.guideline_code.lower()}"
        or entry.evidence_id != f"{entry.document_id}-rec-{entry.recommendation_id}"
        or entry.verification_method != "official_nice_html_article_v1"
    ):
        raise RecommendationEvidenceError("recommendation identity or verification method is invalid")
    try:
        validate_nice_url(entry.canonical_source_url, entry.guideline_code)
    except ValueError as exc:
        raise RecommendationEvidenceError("recommendation URL is not authoritative") from exc
    if not SHA256.fullmatch(entry.recommendation_sha256):
        raise RecommendationEvidenceError("recommendation fingerprint is invalid")


def normalize_recommendation_content(content: str) -> str:
    """NFC Unicode, collapse all whitespace, and normalize HTML list markers to U+2022."""
    normalized = unicodedata.normalize("NFC", content)
    normalized = re.sub(r"\s*[•●]\s*", " • ", normalized)
    return " ".join(normalized.split())


def recommendation_fingerprint(content: str) -> str:
    return hashlib.sha256(normalize_recommendation_content(content).encode("utf-8")).hexdigest()


class _RecommendationParser(HTMLParser):
    VOID_TAGS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self, article_id: str):
        super().__init__(convert_charrefs=True)
        self.article_id = article_id
        self.depth = 0
        self.body_depth: int | None = None
        self.parts: list[str] = []
        self.found = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if self.depth == 0:
            if tag == "article" and attributes.get("id") == self.article_id:
                self.depth = 1
                self.found = True
            return
        if tag not in self.VOID_TAGS:
            self.depth += 1
        if tag == "div" and "recommendation__body" in (attributes.get("class") or "").split():
            self.body_depth = self.depth
        elif self.body_depth is not None and tag == "li":
            self.parts.append(" • ")
        elif self.body_depth is not None and tag == "br":
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self.depth == 0 or tag in self.VOID_TAGS:
            return
        if self.body_depth is not None and self.depth >= self.body_depth and tag in {"p", "li"}:
            self.parts.append(" ")
        if self.body_depth == self.depth and tag == "div":
            self.body_depth = None
        self.depth -= 1

    def handle_data(self, data: str) -> None:
        if self.body_depth is not None:
            self.parts.append(data)


def extract_recommendation(html: str, guideline_code: str, recommendation_id: str) -> str | None:
    article_id = f"{guideline_code.lower()}-{recommendation_id.replace('.', '_')}"
    parser = _RecommendationParser(article_id)
    parser.feed(html)
    content = normalize_recommendation_content("".join(parser.parts))
    return content if parser.found and content else None
