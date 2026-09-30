"""Versioned offline engineering evaluation with explicit, failing release gates."""

import argparse
import json
import os
import platform
import subprocess
import tempfile
from importlib.metadata import version as package_version
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from backend.clinical_models import EvaluationRequest, RecordCoverage
from backend.clinical_review import ClinicalReviewEngine
from backend.clinical_rules import default_registry
from backend.evidence_models import (
    AnswerClaim,
    RankedEvidence,
    RetrievalDiagnostics,
    RetrievalRequest,
    RetrievalResult,
    SupportStatus,
)
from backend.evidence_pipeline import EvidenceAnswerService, public_answer
from backend.fhir_adapter import parse_bundle
from backend.grounding import NLIClaimVerifier, verify_claim
from backend.knowledge_models import Lifecycle
from backend.patient_store import SQLitePatientRepository
from backend.rule_evidence import VerificationStatus
from eval.benchmark import metrics as retrieval_metrics, summarize
from eval.grounding_benchmark import unit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "eval/cases/v1"


def load(suite):
    rows = [json.loads(line) for line in (DATA / f"{suite}.jsonl").read_text(encoding="utf-8").splitlines()]
    if suite == "grounding":
        rows += [
            json.loads(line) for line in (DATA / "grounding_governed.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    for row in rows:
        if (
            not {
                "case_id",
                "suite",
                "description",
                "input",
                "expected_behavior",
                "tags",
                "version",
                "provenance",
                "split",
            }
            <= row.keys()
        ):
            raise ValueError("Incomplete case provenance")
    if len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate evaluation ID")
    return rows


def git_state():
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return {"sha": sha, "dirty": dirty}


class FixedVerifier:
    def __init__(self, status=SupportStatus.SUPPORTED):
        self.status = status

    def verify(self, claim, evidence):
        return self.status


class FixedRetriever:
    def __init__(self, result):
        self.result = result

    def retrieve(self, request):
        return self.result.model_copy(update={"request": request})


class FixedGenerator:
    def __init__(self, text, evidence_ids):
        self.text, self.evidence_ids = text, evidence_ids

    def generate(self, system, prompt):
        return json.dumps(
            {"status": "grounded", "claims": [{"claim_id": "c1", "text": self.text, "evidence_ids": self.evidence_ids}]}
        )


def retrieval_for(units):
    return RetrievalResult(
        request=RetrievalRequest(query="Synthetic engineering query"),
        accepted=True,
        evidence=[RankedEvidence(unit=u, rrf_score=1) for u in units],
        diagnostics=RetrievalDiagnostics(),
    )


def grounding(cases, verifier):
    rows = []
    for case in cases:
        units = [unit(ident, value) for ident, value in case["evidence"].items()]
        registry = None
        if "governed_unit" in case:
            from backend.evidence_models import EvidenceUnit
            from backend.source_registry import SourceRegistry

            units = [EvidenceUnit.model_validate(case["governed_unit"])]
            registry = SourceRegistry()
        if case.get("failure") == "stale":
            units[0].lifecycle_status = Lifecycle.SUPERSEDED
        claim = AnswerClaim(claim_id=case["case_id"], text=case["claim"], evidence_ids=case["cited_ids"])
        local_verifier = (
            FixedVerifier(SupportStatus.UNCERTAIN) if case.get("failure") == "verifier_unavailable" else verifier
        )
        started = perf_counter()
        retrieval = retrieval_for(units)
        if case.get("failure") == "wrong_jurisdiction":
            # Exercise registry metadata rejection before semantics, using an
            # actual governed unit whose jurisdiction has been tampered with.
            from backend.source_registry import SourceRegistry
            from backend.evidence_models import EvidenceUnit

            registered = next(
                u
                for u in json.loads((ROOT / "data/evidence_units.json").read_text(encoding="utf-8"))
                if u["lifecycle_status"] == "current"
            )
            registered["jurisdiction"] = "IN"
            units = [EvidenceUnit.model_validate(registered)]
            claim.evidence_ids = [units[0].evidence_unit_id]
            actual = verify_claim(claim, retrieval_for(units), local_verifier, SourceRegistry()).support_status.value
        else:
            actual = verify_claim(claim, retrieval, local_verifier, registry).support_status.value
        rows.append({**case, "actual": actual, "latency_ms": (perf_counter() - started) * 1000})
    positive = [r for r in rows if r["expected"] == "supported"]
    negative = [r for r in rows if r["expected"] != "supported"]
    predicted = [r for r in rows if r["actual"] == "supported"]
    tp = sum(r["expected"] == "supported" for r in predicted)
    citations = {(r["case_id"], ident) for r in predicted for ident in r["cited_ids"]}
    relevant = {(r["case_id"], ident) for r in positive for ident in r["relevant_ids"]}
    return {
        "cases": len(rows),
        "metrics": {
            "supported_precision": tp / len(predicted) if predicted else None,
            "supported_recall": tp / len(positive),
            "unsupported_detection": sum(r["actual"] != "supported" for r in negative) / len(negative),
            "false_supported_rate": sum(r["actual"] == "supported" for r in negative) / len(negative),
            "false_unsupported_rate": sum(r["actual"] == "unsupported" for r in positive) / len(positive),
            "uncertain_rate": sum(r["actual"] == "uncertain" for r in rows) / len(rows),
            "citation_precision": len(citations & relevant) / len(citations) if citations else None,
            "citation_recall": len(citations & relevant) / len(relevant),
            "verification_p50_ms": float(np.median([r["latency_ms"] for r in rows])),
            "verification_p95_ms": float(np.percentile([r["latency_ms"] for r in rows], 95)),
        },
        "results": rows,
    }


def injections(cases):
    rows = []
    for case in cases:
        value = case["input"]
        result = retrieval_for([unit("a", value["retrieved_text"])])
        # An adversarial generator repeats the hostile instruction. Even an
        # over-permissive semantic verifier must not let control text survive.
        service = EvidenceAnswerService(
            FixedRetriever(result), FixedGenerator(value["draft_claim"], [value["citation"]]), FixedVerifier()
        )
        response = public_answer(service.answer(result.request))
        passed = (
            response["status"] == "abstained"
            and not response["claims"]
            and "CANARY_SYNTHETIC_SECRET" not in json.dumps(response)
        )
        rows.append(
            {"case_id": case["case_id"], "split": case["split"], "passed": passed, "status": response["status"]}
        )
    return {"cases": len(rows), "pass_rate": sum(r["passed"] for r in rows) / len(rows), "results": rows}


def clinical_rules(cases):
    rows = []
    manifest = json.loads((ROOT / "data/synthetic_fhir/manifest.json").read_text(encoding="utf-8"))["fixtures"]
    with tempfile.TemporaryDirectory() as directory:
        for index, case in enumerate(cases):
            inp = case["input"]
            key = f"syn_pat_{inp['fixture']:03}"
            raw = json.loads((ROOT / f"data/synthetic_fhir/{key}.json").read_text(encoding="utf-8"))
            metadata = manifest[key]
            coverage = (
                RecordCoverage.model_validate(metadata["record_coverage"]) if metadata["record_coverage"] else None
            )
            mutation = inp["mutation"]
            patient = next(e["resource"] for e in raw["entry"] if e["resource"]["resourceType"] == "Patient")
            if mutation == "no_birthdate":
                patient.pop("birthDate", None)
            if mutation == "minor":
                patient["birthDate"] = "2020-01-01"
            if mutation == "no_coverage":
                coverage = None
            if mutation == "coverage_boundary":
                coverage = coverage.model_copy(update={"start_date": date(2025, 9, 25)})
            if mutation in {"boundary_event", "old_event", "undated_event", "future_event"}:
                kind = "Procedure" if inp["domain"] == "diabetes" else "Encounter"
                event = next(e["resource"] for e in raw["entry"] if e["resource"]["resourceType"] == kind)
                when = {
                    "boundary_event": "2025-09-24",
                    "old_event": "2025-09-23",
                    "future_event": "2026-09-25",
                    "undated_event": None,
                }[mutation]
                if kind == "Procedure":
                    event.pop("performedDateTime", None)
                    if when:
                        event["performedDateTime"] = when
                else:
                    event.pop("period", None)
                    if when:
                        event["period"] = {"start": when}
            registry = default_registry()
            rule = next(r for r in registry.current_rules() if r.definition.domain == inp["domain"])
            if mutation == "stale_rule":
                ident = rule.definition.source_version_id
                if ident:
                    registry.evidence_registry.versions[ident] = replace(
                        registry.evidence_registry.versions[ident], status=Lifecycle.SUPERSEDED
                    )
                else:
                    entry = registry.recommendation_registry.evidence[rule.definition.evidence_id]
                    registry.recommendation_registry.evidence[entry.evidence_id] = entry.model_copy(
                        update={"verification_status": VerificationStatus.SUPERSEDED}
                    )
            if mutation == "drifted_rule":
                entry = registry.recommendation_registry.evidence[rule.definition.evidence_id]
                registry.recommendation_registry.evidence[entry.evidence_id] = entry.model_copy(
                    update={"verification_status": VerificationStatus.DRIFT_DETECTED}
                )
            repo = SQLitePatientRepository(Path(directory) / f"{index}.db")
            pid = repo.import_context(parse_bundle(raw)[0])[0]
            review = repo.create_review(pid)
            findings = ClinicalReviewEngine(repo, registry).evaluate(
                review.review_id, EvaluationRequest(as_of=metadata["evaluation_as_of"], record_coverage=coverage)
            )
            actual = next(f.status.value for f in findings if f.domain == inp["domain"])
            rows.append(
                {
                    "case_id": case["case_id"],
                    "split": case["split"],
                    "expected": case["expected"],
                    "actual": actual,
                    "passed": actual == case["expected"],
                }
            )
    missing = [r for r in rows if r["expected"] == "insufficient_data"]
    tp = sum(r["expected"] == r["actual"] == "potential_care_gap" for r in rows)
    return {
        "cases": len(rows),
        "accuracy": sum(r["passed"] for r in rows) / len(rows),
        "missing_data_accuracy": sum(r["passed"] for r in missing) / len(missing),
        "finding_precision": tp / max(1, sum(r["actual"] == "potential_care_gap" for r in rows)),
        "finding_recall": tp / max(1, sum(r["expected"] == "potential_care_gap" for r in rows)),
        "results": rows,
    }


def safety():
    from backend.source_registry import SourceRegistry
    from backend.knowledge_models import RetrievalPolicy

    registry = SourceRegistry()
    stale = [v for v in registry.versions.values() if v.status is Lifecycle.SUPERSEDED]
    checks = {}
    for version in stale:
        checks["stale-" + version.version_id] = not registry.eligible(
            {"version_id": version.version_id}, RetrievalPolicy.CURRENT_CLINICAL
        )
        checks["historical-" + version.version_id] = registry.eligible(
            {"version_id": version.version_id}, RetrievalPolicy.HISTORICAL_ONLY
        )
    good = retrieval_for([unit("a", "The synthetic marker is blue.")])

    class FailedRetriever:
        def retrieve(self, request):
            raise RuntimeError("synthetic failure")

    class FailedGenerator:
        def generate(self, *args):
            raise RuntimeError("synthetic failure")

    services = {
        "verifier_unavailable": EvidenceAnswerService(
            FixedRetriever(good), FixedGenerator("The marker is blue.", ["a"]), FixedVerifier(SupportStatus.UNCERTAIN)
        ),
        "provider_failure": EvidenceAnswerService(FixedRetriever(good), FailedGenerator(), FixedVerifier()),
        "retriever_failure": EvidenceAnswerService(FailedRetriever(), FixedGenerator("unused", ["a"]), FixedVerifier()),
        "insufficient_current": EvidenceAnswerService(
            FixedRetriever(good.model_copy(update={"accepted": False, "evidence": []})),
            FixedGenerator("unused", ["a"]),
            FixedVerifier(),
        ),
    }
    for name, service in services.items():
        checks[name] = service.answer(good.request).status.value == "abstained"
    return {"cases": len(checks), "pass_rate": sum(checks.values()) / len(checks), "results": checks}


def gate(report, baseline):
    failures = []
    for path, bound in baseline["gates"].items():
        value = report
        try:
            for part in path.split("."):
                value = value[part]
        except KeyError:
            failures.append(path + ": missing")
            continue
        if value is None or ("min" in bound and value < bound["min"]) or ("max" in bound and value > bound["max"]):
            failures.append(f"{path}: {value} violates {bound}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", action="store_true", help="Use cached local embedding/reranker/NLI models; never a provider API"
    )
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--postgres", action="store_true", help="Include real PostgreSQL evaluation contracts")
    parser.add_argument("--output", type=Path, default=ROOT / "eval/final_system_report.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "eval/baselines/v1.json")
    args = parser.parse_args()
    state = git_state()
    if args.release and state["dirty"]:
        raise SystemExit("Release evaluation requires a clean working tree")
    from eval.contracts import run as run_contracts

    report = {
        "dataset_version": "1.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": state,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "packages": {
                name: package_version(name) for name in ("transformers", "sentence-transformers", "torch", "numpy")
            },
        },
        "mode": "local_models" if args.models else "deterministic_contracts",
        "clinical_rules": clinical_rules(load("clinical_rules")),
        "prompt_injection": injections(load("prompt_injection")),
        "safety": safety(),
        "contracts": run_contracts(args.postgres),
    }
    if args.models:
        from embeddings.retrieve import EvidenceRetriever

        retriever = EvidenceRetriever()
        verifier = NLIClaimVerifier()
        report["retrieval"], report["grounding"] = {}, {}
        for split in ("development", "held_out"):
            cases = [c for c in load("retrieval") if c["split"] == split]
            rows, latency = summarize(cases, retriever)
            if any(r["accepted"] and not r["reranker_used"] for r in rows):
                raise RuntimeError("Model evaluation requires the reranker")
            report["retrieval"][split] = {
                "cases": len(rows),
                "metrics": retrieval_metrics(rows),
                "latency": latency,
                "results": rows,
                "historical_count": sum(c["intent"] == "historical" for c in cases),
                "stale_traps": sum("stale_trap" in c["tags"] for c in cases),
            }
            report["grounding"][split] = grounding([c for c in load("grounding") if c["split"] == split], verifier)
        if verifier._model is None:
            raise RuntimeError("NLI model unavailable; cannot report model metrics")
        from eval.benchmark import cases as legacy_cases

        legacy_rows, legacy_latency = summarize(legacy_cases("held_out"), retriever)
        report["phase5_comparison"] = {
            "description": "Unchanged Phase 5 held-out queries, retained pool size 30; warm local models",
            "cases": len(legacy_rows),
            "metrics": retrieval_metrics(legacy_rows),
            "latency": legacy_latency,
            "prior": json.loads((ROOT / "eval/final_phase5_report.json").read_text(encoding="utf-8")),
        }
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if not args.models:
        baseline = {
            **baseline,
            "gates": {k: v for k, v in baseline["gates"].items() if not k.startswith(("retrieval.", "grounding."))},
        }
    failures = gate(report, baseline)
    report.update(
        baseline_version=baseline["version"], thresholds=baseline["gates"], gate_failures=failures, passed=not failures
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"passed": not failures, "failures": failures, "report": str(args.output)}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
