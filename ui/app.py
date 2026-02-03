import streamlit as st
import sys
import os

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.rag_pipeline import answer_question

st.set_page_config(
    page_title="Medical AI Copilot",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical AI Copilot")
st.markdown(
    """
    Ask medical questions based on **trusted clinical documents**
    (WHO, NICE, Ministry of Health, OpenStax).

    ⚠️ *This tool is for educational purposes only and is not a substitute for professional medical advice.*
    """
)

st.divider()

question = st.text_input(
    "Enter your medical question:",
    placeholder="e.g., What is diabetes?"
)

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing medical documents..."):
            result = answer_question(question)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        for src in result["sources"]:
            st.write(f"- {src}")
