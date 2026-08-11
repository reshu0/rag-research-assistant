import streamlit as st
import requests

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="🔎",
    layout="centered"
)

st.title("🔎 RAG Research Assistant")
st.caption("Ask questions about the documents in the knowledge base.")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input("Ask a question...")

if question:
    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/query",
                    json={"question": question},
                    timeout=120
                )

                response.raise_for_status()

                data = response.json()
                answer = data.get("answer", "No answer returned.")

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except requests.exceptions.RequestException as e:
                error = f"Backend connection failed: {e}"
                st.error(error)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error
                })