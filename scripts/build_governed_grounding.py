"""Curate distinct governed passages; labels and split precede model evaluation."""

import json
from pathlib import Path

CASES = [
    (
        "development",
        "nice-ng136",
        "1.2.1",
        "supported",
        "Both arms should be measured when considering a hypertension diagnosis.",
    ),
    (
        "development",
        "nice-ng136",
        "1.2.2",
        "contradicted",
        "Record the higher of the last two clinic blood-pressure readings.",
    ),
    (
        "development",
        "nice-ng136",
        "1.2.6",
        "supported",
        "ABPM confirmation should average at least 14 readings taken during waking hours.",
    ),
    (
        "development",
        "nice-ng136",
        "1.2.7",
        "contradicted",
        "HBPM averaging should include the measurements from the first day.",
    ),
    (
        "development",
        "nice-ng136",
        "1.2.10",
        "supported",
        "If hypertension is not diagnosed, clinic blood pressure should be measured at least every 5 years subsequently.",
    ),
    ("development", "nice-ng238", "1.1.5", "contradicted", "Formal cardiovascular risk assessment cannot be declined."),
    (
        "development",
        "nice-ng238",
        "1.3.4",
        "contradicted",
        "Dietary advice should ignore comorbidities and drug treatment.",
    ),
    (
        "development",
        "nice-ng238",
        "1.3.5",
        "supported",
        "Aerobic and muscle-strengthening activities are advised for people at high risk of or with CVD.",
    ),
    (
        "held_out",
        "nice-ng136",
        "1.2.3",
        "supported",
        "Offer ABPM to confirm hypertension if clinic blood pressure is between 140/90 mmHg and 180/120 mmHg.",
    ),
    (
        "held_out",
        "nice-ng136",
        "1.2.12",
        "supported",
        "Measure blood pressure at least annually in adults with type 2 diabetes without previously diagnosed hypertension or renal disease.",
    ),
    (
        "held_out",
        "nice-ng136",
        "1.4.1",
        "contradicted",
        "Lifestyle advice should stop after hypertension is diagnosed.",
    ),
    (
        "held_out",
        "nice-ng136",
        "1.4.6",
        "contradicted",
        "Calcium supplements should be offered to reduce blood pressure.",
    ),
    (
        "held_out",
        "nice-ng238",
        "1.1.6",
        "contradicted",
        "Use opportunistic assessment as the main primary-care strategy to identify CVD risk in unselected people.",
    ),
    (
        "held_out",
        "nice-ng238",
        "1.1.7",
        "supported",
        "Use QRISK3 to estimate CVD risk over the next 10 years for people aged 25 to 84 without CVD.",
    ),
    ("held_out", "nice-ng238", "1.1.8", "supported", "Use QRISK3 for people with type 2 diabetes aged 25 to 84."),
    (
        "held_out",
        "nice-ng238",
        "1.3.2",
        "contradicted",
        "For people at high risk of or with CVD, the recommended saturated-fat limit is 70% of total energy intake.",
    ),
]


def main():
    units = json.loads(Path("data/evidence_units.json").read_text(encoding="utf-8"))
    rows = []
    for split, document, recommendation, expected, claim in CASES:
        matches = [
            u
            for u in units
            if u["document_id"] == document
            and u.get("recommendation_id") == recommendation
            and u["lifecycle_status"] == "current"
        ]
        if len(matches) != 1:
            raise ValueError("Governed recommendation must resolve uniquely")
        evidence = matches[0]
        ident = evidence["evidence_unit_id"]
        rows.append(
            {
                "case_id": "governed-" + document + "-" + recommendation,
                "suite": "grounding",
                "version": "1.1.0",
                "split": split,
                "description": "Independent paraphrase or deliberate contradiction of " + ident,
                "input": {"claim": claim, "evidence_unit_id": ident},
                "expected_behavior": expected,
                "expected": expected,
                "claim": claim,
                "evidence": {ident: evidence["text"]},
                "cited_ids": [ident],
                "relevant_ids": [ident] if expected == "supported" else [],
                "governed_unit": evidence,
                "tags": ["governed_passage", "paraphrase" if expected == "supported" else "deliberate_contradiction"],
                "provenance": {
                    "document_id": document,
                    "version_id": evidence["version_id"],
                    "recommendation_id": recommendation,
                    "canonical_url": evidence["canonical_source_url"],
                    "label_note": "Contradicted claims are synthetic negative assertions, never publisher recommendations.",
                },
            }
        )
    Path("eval/cases/v1/grounding_governed.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
