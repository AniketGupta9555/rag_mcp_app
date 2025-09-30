import streamlit as st
import requests

st.set_page_config(page_title="🩺 Prescription Chatbot", layout="centered")

st.title("🩺 Prescription Chatbot (Local RAG with MCP)")

st.markdown(
    "Ask questions based on preloaded prescription PDFs. "
    "⚠️ *Disclaimer: This is not medical advice. Always consult a licensed doctor.*"
)

# Chat input
question = st.text_input("💬 Your Question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            res = requests.post(
                "http://127.0.0.1:8000/chat",
                params={"question": question}
            )
            if res.status_code == 200:
                data = res.json()

                # Display Answer
                st.markdown("### 💡 Answer")
                st.write(data["answer"])

                # Display Sources
                st.markdown("### 📚 Sources")
                for i, src in enumerate(data["sources"], 1):
                    with st.container():
                        st.markdown(f"**{i}. {src['file']} — Page {src['page']}**")
                        with st.expander("🔎 View referenced text"):
                            st.write(src["chunk"])

            else:
                st.error(f"Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")
