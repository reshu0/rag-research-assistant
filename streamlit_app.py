import requests
import streamlit as st

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="🔎",
    layout="centered",
)

st.title("🔎 RAG Research Assistant")
st.caption("Hybrid retrieval + RRF + Cross-Encoder Reranking + Local LLM")

question = st.text_input(
    "Ask a question",
    placeholder="How does Tesla make money?",
)

if st.button("Ask", type="primary") and question:

    with st.spinner("Retrieving and generating answer..."):

        try:
            response = requests.post(
                "http://127.0.0.1:8000/query",
                json={"question": question},
                timeout=120,
            )

            response.raise_for_status()

            data = response.json()

            st.subheader("Answer")
            st.write(data["answer"])

        except requests.exceptions.RequestException as e:
            st.error(f"Backend error: {e}")