"""Synthetic patient review workspace with governed Ask Evidence access."""

import json
import os
from pathlib import Path

import streamlit as st

from backend.fhir_adapter import FHIRInputError, parse_bundle
from backend.knowledge_models import RetrievalPolicy
from backend.patient_api import DEMO_LABELS, FIXTURES
from backend.patient_context import active_conditions, active_medications, data_availability, known_allergies, timeline
from backend.patient_store import SQLitePatientRepository
from backend.rag_pipeline import answer_question
from backend.source_registry import SourceRegistry

st.set_page_config(page_title="Medical AI Copilot", layout="wide")
st.markdown(
    """<style>
    .stApp {background:#f6f8f7;color:#182b27}
    .block-container {max-width:1200px;padding-top:1.8rem}
    .synthetic {display:inline-block;background:#d6eee8;color:#075c4c;border-radius:6px;
        padding:.35rem .7rem;font-size:.75rem;font-weight:700;letter-spacing:.06em}
    div[data-testid="stMetric"] {background:white;border:1px solid #d9e5e0;border-radius:10px;padding:1rem}
</style>""",
    unsafe_allow_html=True,
)

st.title("Medical AI Copilot")
st.caption("Governed evidence retrieval and structured synthetic patient review")
st.markdown('<span class="synthetic">SYNTHETIC DEMO DATA ONLY</span>', unsafe_allow_html=True)
st.info(
    "Prototype for clinician review. The patient workspace presents supplied records; it does not evaluate care gaps or make treatment decisions."
)

repository = SQLitePatientRepository(
    os.environ.get("PATIENT_DB_PATH", str(Path(__file__).resolve().parent / "data" / "patient_context.db"))
)
registry = SourceRegistry()


def import_bundle(raw):
    context, _ = parse_bundle(raw)
    patient_id, _, _ = repository.import_context(context)
    st.session_state.selected_patient_id = patient_id
    st.session_state.pop("selected_review_id", None)


def render_citations(citations):
    if citations:
        with st.expander("Source citations"):
            st.table(
                [
                    {
                        "Source": item.get("canonical_title") or "Unknown source",
                        "Publisher": item.get("publisher") or "Unknown",
                        "Version": item.get("version_id") or "Unknown",
                        "Lifecycle": item.get("lifecycle_status") or "Unknown",
                        "Pages": f"{item.get('page_start') or '?'}–{item.get('page_end') or item.get('page_start') or '?'}",
                    }
                    for item in citations
                ]
            )


patients_tab, review_tab, evidence_tab, sources_tab = st.tabs(
    ["Patients", "Clinical Review", "Ask Evidence", "Knowledge Sources"]
)

with patients_tab:
    st.subheader("Select a synthetic patient")
    demo_key = st.selectbox("Bundled demo records", options=list(DEMO_LABELS), format_func=lambda key: DEMO_LABELS[key])
    if st.button("Load demo patient", type="primary"):
        import_bundle(json.loads((FIXTURES / f"{demo_key}.json").read_text(encoding="utf-8")))
        st.rerun()
    uploaded = st.file_uploader("Or import a supported synthetic FHIR R4 collection Bundle", type="json")
    if uploaded is not None and st.button("Validate and import Bundle"):
        try:
            import_bundle(json.load(uploaded))
            st.success("Synthetic patient imported")
            st.rerun()
        except (ValueError, FHIRInputError, json.JSONDecodeError) as exc:
            st.error(f"Import failed: {exc}")

    patients = repository.list_patients()
    if patients:
        labels = {
            pid: f"{context.patient.synthetic_label} · {context.patient.source_patient_id}"
            for pid, context, _ in patients
        }
        selected_id = st.selectbox(
            "Imported patients",
            options=list(labels),
            index=list(labels).index(st.session_state.selected_patient_id)
            if st.session_state.get("selected_patient_id") in labels
            else 0,
            format_func=lambda pid: labels[pid],
        )
        st.session_state.selected_patient_id = selected_id
        context, digest = repository.get_patient(selected_id)
        st.subheader(context.patient.synthetic_label)
        a, b, c = st.columns(3)
        a.metric("Synthetic ID", context.patient.source_patient_id)
        b.metric("Birth date", context.patient.birth_date or "Not supplied")
        c.metric("Gender as recorded", context.patient.gender or "Not supplied")
        st.caption(f"Context snapshot: {digest[:16]}…")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Conditions")
            st.table(
                [
                    {
                        "Condition": x.code.display,
                        "Status": x.clinical_status or "Unknown",
                        "Onset": x.onset or "Not supplied",
                    }
                    for x in active_conditions(context)
                ]
            )
            st.markdown("#### Medications")
            st.table(
                [
                    {"Medication": x.code.display, "Status": x.status, "Authored": x.authored_on or "Not supplied"}
                    for x in active_medications(context)
                ]
            )
            st.markdown("#### Allergies")
            allergies = known_allergies(context)
            if allergies:
                st.table(
                    [
                        {
                            "Allergy": x.code.display,
                            "Status": x.clinical_status or "Unknown",
                            "Verification": x.verification_status or "Unknown",
                        }
                        for x in allergies
                    ]
                )
            else:
                st.caption("No allergy record was supplied in this snapshot.")
        with right:
            st.markdown("#### Recent observations")
            recent = sorted(
                (x for x in context.observations if x.status not in ("entered-in-error", "cancelled")),
                key=lambda x: (x.effective or "", x.observation_id),
                reverse=True,
            )[:8]
            st.table(
                [
                    {
                        "Observation": x.code.display,
                        "Value": (
                            f"{x.value.value:g} {x.value.unit or ''}"
                            if x.value
                            else ", ".join(
                                f"{c.code.display}: {c.value.value:g} {c.value.unit or ''}" for c in x.components
                            )
                        ),
                        "Date": x.effective or "Undated",
                    }
                    for x in recent
                ]
            )
            st.markdown("#### Encounters")
            st.table(
                [
                    {
                        "Type": x.type.display if x.type else "Encounter",
                        "Status": x.status,
                        "Start": x.start or "Undated",
                    }
                    for x in context.encounters
                ]
            )
            st.markdown("#### Data availability")
            availability = data_availability(context)
            st.write("Supplied: " + (", ".join(availability.available_data_types) or "none"))
            st.write("Not supplied: " + (", ".join(availability.missing_data_types) or "none"))
        st.markdown("#### Timeline")
        st.table(
            [{"When": e.timestamp or "Undated", "Type": e.event_type, "Event": e.title} for e in timeline(context)]
        )
    else:
        st.caption("Load one of the bundled synthetic records to begin.")

with review_tab:
    st.subheader("Clinical Review")
    selected_id = st.session_state.get("selected_patient_id")
    if selected_id:
        if st.button("Create review snapshot", type="primary"):
            review = repository.create_review(selected_id)
            st.session_state.selected_review_id = review.review_id
        reviews = repository.list_reviews(selected_id)
        review_ids = [item.review_id for item in reviews]
        if review_ids:
            review_id = st.selectbox(
                "Saved reviews",
                options=review_ids,
                index=review_ids.index(st.session_state.selected_review_id)
                if st.session_state.get("selected_review_id") in review_ids
                else 0,
                format_func=lambda value: next(
                    f"{item.created_at} · {item.status.value}" for item in reviews if item.review_id == value
                ),
            )
            review = repository.get_review(review_id)
            st.caption(
                f"Review {review.review_id} · {review.status.value} · snapshot {review.patient_context_hash[:16]}…"
            )
            st.markdown("#### Findings")
            st.write("No rule-based clinical findings are available for this review.")
            st.markdown("#### Supplied and unavailable record information")
            st.write("Supplied: " + (", ".join(review.data_availability.available_data_types) or "none"))
            st.write("Not supplied: " + (", ".join(review.data_availability.missing_data_types) or "none"))
            st.markdown("#### Snapshot timeline")
            st.table(
                [{"When": e.timestamp or "Undated", "Type": e.event_type, "Event": e.title} for e in review.timeline]
            )
            if review.status.value != "completed" and st.button("Mark review completed"):
                repository.complete_review(review_id)
                st.rerun()
    else:
        st.caption("Select a synthetic patient in Patients to create a review.")

with evidence_tab:
    st.subheader("Ask Evidence")
    st.caption(
        "General evidence queries use the governed current clinical guideline policy. Patient records are not sent to this Q&A feature."
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_citations(message.get("citations", []))
    if question := st.chat_input("Ask a general evidence question"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            with st.spinner("Searching current clinical guidelines"):
                result = answer_question(question, policy=RetrievalPolicy.CURRENT_CLINICAL)
            st.markdown(result["answer"])
            render_citations(result.get("citations", []))
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "citations": result.get("citations", [])}
        )

with sources_tab:
    st.subheader("Knowledge Sources")
    st.caption("The source registry controls lifecycle eligibility for evidence retrieval.")
    st.table(
        [
            {
                "Title": registry.documents[v.document_id].canonical_title,
                "Publisher": registry.documents[v.document_id].publisher,
                "Type": registry.documents[v.document_id].source_type.value,
                "Lifecycle": v.status.value,
                "Version": v.version_label or "Unknown",
            }
            for v in registry.versions.values()
        ]
    )
