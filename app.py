import streamlit as st
import sys
import os
import re
import time

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.rag_pipeline import answer_question

st.set_page_config(
    page_title="Medical AI Copilot",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -----------------------------------
# Design tokens
# -----------------------------------
# See PHASE_5_DESIGN_NOTES.md for the full design rationale. Summary:
# this app's visual language borrows from how clinical guidelines (NICE,
# WHO) are actually typeset -- numbered recommendation chips, a document-
# width reading column, and a single restrained clinical-teal accent --
# rather than generic chatbot or AI-tool defaults.

PAPER = "#F7F7F4"
INK = "#1A2421"
TEAL = "#0E6E5C"
SLATE = "#5B6B66"
AMBER = "#B6762C"
HAIRLINE = "#DBDDD7"

# -----------------------------------
# Source name mapping
# -----------------------------------
# Maps real raw source filenames (as they exist in the indexed corpus) to
# clean, recruiter-readable display names. Falls back to a cleaned-up
# version of the raw filename for any source not in this map, so a new
# corpus document never breaks display -- it just looks slightly less
# polished until this map is updated.
SOURCE_DISPLAY_NAMES = {
    "nice_diabetic_foot_guideline.pdf.pdf.txt": "NICE NG19 — Diabetic Foot Problems",
    "nice_hypertension_guideline.pdf.pdf.txt": "NICE NG136 — Hypertension",
    "nice_cvd_lipid_guideline.pdf.pdf.txt": "NICE NG238 — Cardiovascular Risk & Lipids",
    "nice_type2_diabetes_guideline.pdf.txt": "NICE NG28 — Type 2 Diabetes",
    "moh_diabetes_mellitus_guideline.pdf.txt": "MoH — Diabetes Mellitus Guideline",
    "who_doc_1.pdf.txt": "WHO — Tuberculosis Report",
    "who_doc_2.pdf.txt": "WHO — Malaria Report",
    "cdc_chronic_disease_overview.pdf.txt": "CDC — Chronic Disease Overview",
    "openstax_anatomy_physiology.pdf.txt": "OpenStax — Anatomy & Physiology",
}


def clean_fallback_name(raw_source: str) -> str:
    """Best-effort readable name for any source not in the explicit map."""
    name = raw_source.split(",")[0]
    name = re.sub(r"\.pdf(\.pdf)?\.txt$", "", name)
    name = name.replace("_", " ").strip()
    return name.title()


def parse_source_entry(raw_entry: str):
    """
    Splits a source string like "nice_diabetic_foot_guideline.pdf.pdf.txt,
    pages 15-16" into a (display_name, page_label) pair. Page label is an
    empty string if the source has no page info (graceful degradation,
    consistent with format_page_range's own behavior in rag_pipeline.py).
    """
    if "," in raw_entry:
        filename, page_part = raw_entry.split(",", 1)
        page_part = page_part.strip()
    else:
        filename, page_part = raw_entry, ""

    display_name = SOURCE_DISPLAY_NAMES.get(filename, clean_fallback_name(filename))
    return display_name, page_part


def render_source_chips(sources: list) -> str:
    """
    Groups (name, page) pairs by document name and renders ONE chip per
    unique document, with all its page ranges combined -- e.g.
    "NICE NG19 — Diabetic Foot Problems · pp.6-7, 13-16, 27-29, 31-33"
    instead of 5 separate, nearly-identical chips that differ only in page
    range. Found in testing: a single real answer cited 5 chunks from the
    same guideline, rendering as 5 repetitive chips before this fix.

    Preserves the order documents first appear in `sources`, and the order
    pages were cited within each document (not re-sorted), since citation
    order can reflect how the answer actually references them.
    """
    grouped = {}
    for name, page in sources:
        page_number = re.sub(r"^pages?\s*", "", page).strip()
        grouped.setdefault(name, [])
        if page_number and page_number not in grouped[name]:
            grouped[name].append(page_number)

    def page_sort_key(page_str):
        # Sort by the first number in the range (e.g. "13-15" -> 13),
        # so chips read in natural page order rather than citation order.
        match = re.search(r"\d+", page_str)
        return int(match.group()) if match else 0

    chips = []
    for name, pages in grouped.items():
        pages_sorted = sorted(pages, key=page_sort_key)
        if not pages_sorted:
            suffix = ""
        elif len(pages_sorted) == 1:
            suffix = f" · p.{pages_sorted[0]}"
        else:
            suffix = f" · pp.{', '.join(pages_sorted)}"
        chips.append(f'<span class="source-chip">{name}{suffix}</span>')

    return "".join(chips)


# -----------------------------------
# Example questions
# -----------------------------------
# Real questions verified against the actual corpus throughout this
# project's development -- not placeholder text. Clicking one fills the
# chat input and submits immediately, so a visitor can see the tool work
# without typing anything.
EXAMPLE_QUESTIONS = [
    "How should a diabetic foot ulcer be managed?",
    "When should statins be offered for cardiovascular risk reduction?",
    "What is the SINBAD classification?",
    "Explain insulin resistance.",
]

# -----------------------------------
# Custom styling
# -----------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400..600&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* Ambient background: a soft, fixed radial glow behind everything.
       Strengthened after testing showed the original was too faint to
       register against the white cards covering most of the page --
       this needs to be visible in the GAPS between cards, not just
       theoretically present in the CSS. */
    .stApp {{
        background-color: {PAPER};
        background-image:
            radial-gradient(circle at 15% 0%, {TEAL}26 0%, transparent 50%),
            radial-gradient(circle at 100% 25%, {AMBER}1f 0%, transparent 45%),
            radial-gradient(circle at 50% 100%, {TEAL}14 0%, transparent 55%);
        background-attachment: fixed;
    }}

    .block-container {{
        max-width: 760px;
        padding-top: 1.5rem;
        padding-bottom: 6rem;
    }}

    /* Shared shadow system: layered, soft, multi-stop. Strengthened after
       testing showed the original shadows were too soft to read as real
       depth against the light paper background -- light-on-light shadows
       need more contrast than the same values would on pure white. */
    :root {{
        --shadow-soft: 0 1px 3px rgba(26,36,33,0.07), 0 6px 16px rgba(26,36,33,0.07);
        --shadow-lifted: 0 3px 6px rgba(26,36,33,0.09), 0 16px 36px rgba(26,36,33,0.14);
        --shadow-glow: 0 0 0 1px {TEAL}3d, 0 10px 28px {TEAL}33;
        --inset-highlight: inset 0 1px 0 rgba(255,255,255,0.9);
        --ease-premium: cubic-bezier(0.22, 1, 0.36, 1);
    }}

    /* Header -- glass surface, sticky, with backdrop blur so content
       scrolling underneath shows through softly. @supports fallback
       keeps it fully opaque and readable on browsers without blur support. */
    .app-header {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        border-bottom: 1px solid {HAIRLINE};
        padding: 0.95rem 0;
        margin-bottom: 1.3rem;
        background: rgba(247,247,244,0.62);
        position: sticky;
        top: 0;
        z-index: 20;
        margin-left: -1rem;
        margin-right: -1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        box-shadow: 0 8px 24px -8px rgba(26,36,33,0.10);
    }}
    @supports (backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px)) {{
        .app-header {{
            backdrop-filter: blur(14px) saturate(140%);
            -webkit-backdrop-filter: blur(14px) saturate(140%);
        }}
    }}
    .app-header h1 {{
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.7rem;
        color: {INK};
        margin: 0;
        letter-spacing: -0.015em;
    }}
    .app-header .tag {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: {TEAL};
        text-transform: uppercase;
        letter-spacing: 0.07em;
        border: 1px solid {TEAL}40;
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        background: linear-gradient(135deg, {TEAL}14, {TEAL}05);
        box-shadow: var(--shadow-soft);
    }}

    /* Disclaimer */
    .disclaimer {{
        font-size: 0.82rem;
        color: {AMBER};
        background: linear-gradient(135deg, #B6762C14, #B6762C08);
        border: 1px solid #B6762C2A;
        border-radius: 10px;
        padding: 0.6rem 0.85rem;
        margin-bottom: 1.4rem;
        line-height: 1.5;
        box-shadow: var(--shadow-soft);
    }}

    /* Example question chips */
    .example-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: {SLATE};
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.55rem;
    }}
    /* .stButton > button is a long-stable Streamlit selector pattern
       (confirmed in widespread, multi-year community use) -- preferred
       here over newer data-testid-based button selectors, which use a
       different attribute naming convention (stbaseButton-secondary)
       than what was first assumed, to keep this on the simplest, most
       proven selector available. */
    .stButton > button {{
        background: white;
        color: {INK};
        border: 1px solid {HAIRLINE};
        border-radius: 10px;
        font-size: 0.85rem;
        padding: 0.55rem 1rem;
        font-family: 'Inter', sans-serif;
        box-shadow: var(--shadow-soft);
        transition: border-color 0.25s var(--ease-premium),
                    color 0.25s var(--ease-premium),
                    box-shadow 0.25s var(--ease-premium),
                    transform 0.2s var(--ease-premium);
    }}
    .stButton > button:hover {{
        border-color: {TEAL}90;
        color: {TEAL};
        box-shadow: var(--shadow-glow);
        transform: translateY(-1px);
    }}
    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* Chat message containers. div[data-testid="stChatMessage"] is an
       internal (undocumented) Streamlit attribute, but it's in current,
       confirmed community use for exactly this kind of styling as of
       Streamlit 1.5x. The "name" argument to st.chat_message is
       documented as an accessibility label only though, so rather than
       guess at how role is exposed, an empty marker element with a class
       I control (.assistant-marker, emitted as the first thing inside
       each assistant message) is used with :has() to detect and style
       the assistant message specifically -- this only depends on the one
       confirmed selector, not on guessing Streamlit's avatar/role markup. */
    div[data-testid="stChatMessage"] {{
        background: transparent;
        padding: 0.25rem 0;
    }}

    div[data-testid="stChatMessage"]:has(.assistant-marker) {{
        background: linear-gradient(165deg, white, {PAPER}80);
        border-left: 3px solid {TEAL};
        border-radius: 4px 12px 12px 4px;
        padding: 1rem 1.2rem;
        box-shadow: var(--shadow-lifted), var(--inset-highlight);
        transition: box-shadow 0.3s var(--ease-premium);
    }}

    .assistant-marker {{
        display: none;
    }}

    /* Source chips */
    .source-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.75rem;
        padding-top: 0.65rem;
        border-top: 1px solid {HAIRLINE};
    }}
    .source-chip {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: {TEAL};
        border: 1px solid {TEAL}55;
        border-radius: 20px;
        padding: 0.22rem 0.65rem;
        white-space: nowrap;
        background: linear-gradient(135deg, {TEAL}10, {TEAL}05);
        transition: box-shadow 0.2s var(--ease-premium), transform 0.2s var(--ease-premium);
    }}
    .source-chip:hover {{
        box-shadow: var(--shadow-glow);
        transform: translateY(-1px);
    }}

    .empty-sources-note {{
        font-size: 0.78rem;
        color: {SLATE};
        font-style: italic;
        margin-top: 0.5rem;
    }}

    /* Indexed-sources panel (empty state) -- same chip language as
       citations, but muted slate rather than teal, since these are an
       index listing, not an active citation on a specific answer. */
    .corpus-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.55rem;
        margin-bottom: 1.6rem;
    }}
    .corpus-chip {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: {SLATE};
        border: 1px solid {HAIRLINE};
        border-radius: 20px;
        padding: 0.22rem 0.65rem;
        white-space: nowrap;
        background: white;
        box-shadow: var(--shadow-soft);
        transition: border-color 0.2s var(--ease-premium);
    }}
    .corpus-chip:hover {{
        border-color: {SLATE}70;
    }}

    /* Footer */
    .app-footer {{
        text-align: center;
        font-size: 0.74rem;
        color: {SLATE};
        margin-top: 2.5rem;
    }}

    /* Entrance animation for the empty state -- a single, deliberate
       moment on load rather than scattered effects everywhere. */
    @keyframes premiumFadeUp {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    div[data-testid="stVerticalBlock"] > div:has(.example-label) {{
        animation: premiumFadeUp 0.5s var(--ease-premium);
    }}

    @media (prefers-reduced-motion: reduce) {{
        * {{ animation: none !important; transition: none !important; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------
# Header
# -----------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🩺 Medical AI Copilot</h1>
        <span class="tag">Evidence-Grounded</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer">
        ⚠️ Educational and research tool only — not a substitute for professional
        medical advice. Answers are generated from indexed clinical guidelines
        (NICE, WHO, MoH) and reference material, with sources cited below each answer.
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------
# Session state
# -----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def queue_question(question: str):
    st.session_state.pending_question = question


# -----------------------------------
# Example questions (only before the first message, so it's an empty-state
# feature, not a permanent fixture competing with real chat history)
# -----------------------------------
if not st.session_state.messages and not st.session_state.pending_question:
    st.markdown('<div class="example-label">Try asking</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            st.button(q, key=f"example_{i}", on_click=queue_question, args=(q,), use_container_width=True)

    st.markdown('<div class="example-label" style="margin-top:1.6rem;">Indexed sources</div>', unsafe_allow_html=True)
    corpus_chips = "".join(
        f'<span class="corpus-chip">{name}</span>'
        for name in sorted(set(SOURCE_DISPLAY_NAMES.values()))
    )
    st.markdown(f'<div class="corpus-row">{corpus_chips}</div>', unsafe_allow_html=True)

# -----------------------------------
# Render chat history
# -----------------------------------
for message in st.session_state.messages:
    # Single-codepoint emoji used deliberately -- the ZWJ-based
    # "health worker" emoji (🧑‍⚕️) was found to render as a generic
    # silhouette on some Windows font configurations, since ZWJ emoji
    # composition support varies by system. 👤 is a single codepoint and
    # renders consistently.
    avatar = "👤" if message["role"] == "user" else "🩺"
    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "assistant":
            # Empty marker, first element in the message -- lets the CSS
            # above detect and style this specific message via :has(),
            # without depending on Streamlit's internal avatar/role markup.
            st.markdown('<div class="assistant-marker"></div>', unsafe_allow_html=True)
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            chips_html = render_source_chips(message["sources"])
            st.markdown(f'<div class="source-row">{chips_html}</div>', unsafe_allow_html=True)


def stream_words(text: str):
    """
    Yields the already-complete answer word by word with a short delay,
    purely for a polished "typewriter" presentation. The backend
    (Groq via answer_question) returns the full answer in one call, not
    a true token stream -- this is a presentation-layer effect on a
    complete response, not simulated/fake content.

    Delay tuned down from an earlier 0.012s/word after a user confirmed
    the reveal itself (not the API call) was the source of perceived
    slowness -- 0.012s/word added up to ~2 real seconds of pure
    artificial delay on a realistic ~165-word clinical answer. 0.004s/word
    keeps a visible reveal effect without it competing with actual wait
    time.
    """
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.004)


def run_turn(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🩺"):
        st.markdown('<div class="assistant-marker"></div>', unsafe_allow_html=True)

        with st.spinner("Reviewing clinical guidelines..."):
            result = answer_question(question)

        st.write_stream(stream_words(result["answer"]))

        parsed_sources = [parse_source_entry(s) for s in result["sources"]]
        if parsed_sources:
            chips_html = render_source_chips(parsed_sources)
            st.markdown(f'<div class="source-row">{chips_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="empty-sources-note">No matching source found in the indexed guidelines.</div>',
                unsafe_allow_html=True,
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": parsed_sources}
    )


# -----------------------------------
# Handle a queued example-question click
# -----------------------------------
if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    run_turn(q)

# -----------------------------------
# Chat input
# -----------------------------------
if prompt := st.chat_input("Ask about a clinical guideline, condition, or treatment..."):
    run_turn(prompt)

st.markdown(
    """
    <div class="app-footer">
        Built on NICE, WHO, MoH, and OpenStax reference material · Retrieval-augmented, citation-grounded
    </div>
    """,
    unsafe_allow_html=True,
)