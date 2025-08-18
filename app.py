import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import re
import os

# Load env vars
load_dotenv()

st.set_page_config(page_title="PDF Q&A Chatbot", layout="centered")
st.title("📄 PDF Q&A Chatbot (Qwen)")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []

# Load model
@st.cache_resource
def get_model():
    api_key = st.secrets["OPENAI_API_KEY"]
    base_url = st.secrets["OPENAI_BASE_URL"]

    return ChatOpenAI(
        model="qwen/qwq-32b:free",
        temperature=0.0,
        streaming=True,
        api_key=api_key,
        base_url=base_url,
    )

model = get_model()

# Helper function to clean <think> tags
def clean_response(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# --- PDF UPLOADER ON RIGHT SIDE ---
col_left, col_right = st.columns([2, 1])

with col_right:
    st.markdown("### 📂 Upload a PDF file")
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if uploaded_file:
        text = ""
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.pdf_chunks = splitter.split_text(text)

        st.success(f"✅ PDF split into {len(st.session_state.pdf_chunks)} parts")

        # Options after upload
        task = st.radio("Choose an action:", ["None", "Generate Questions", "Summarize PDF"])

        if task == "Generate Questions":
            if st.button("Generate"):
                with col_left:
                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        raw_output = ""

                        prompt = "Generate 8–10 exam-style questions based on this PDF:\n\n" + text
                        st.session_state.messages.append(HumanMessage(content=prompt))

                        for chunk in model.stream(st.session_state.messages):
                            raw_output += chunk.content or ""
                            response_placeholder.markdown(clean_response(raw_output) + "▌")

                        final_output = clean_response(raw_output)
                        response_placeholder.markdown(final_output)
                        st.session_state.messages.append(AIMessage(content=final_output))

        elif task == "Summarize PDF":
            if st.button("Summarize"):
                with col_left:
                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        raw_output = ""

                        prompt = "Summarize this PDF in simple language:\n\n" + text
                        st.session_state.messages.append(HumanMessage(content=prompt))

                        for chunk in model.stream(st.session_state.messages):
                            raw_output += chunk.content or ""
                            response_placeholder.markdown(clean_response(raw_output) + "▌")

                        final_output = clean_response(raw_output)
                        response_placeholder.markdown(final_output)
                        st.session_state.messages.append(AIMessage(content=final_output))

# --- USER INPUT AT BOTTOM ---
st.markdown("---")
st.markdown("### 💬 Ask your own question")
user_input = st.chat_input("Type your question here...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        raw_output = ""

        try:
            if st.session_state.pdf_chunks:
                relevant = max(
                    st.session_state.pdf_chunks,
                    key=lambda chunk: sum(word.lower() in chunk.lower() for word in user_input.split())
                )
                prompt = f"Answer using this PDF content:\n\n{relevant}\n\nQuestion: {user_input}"
            else:
                prompt = user_input

            st.session_state.messages.append(HumanMessage(content=prompt))

            for chunk in model.stream(st.session_state.messages):
                raw_output += chunk.content or ""
                response_placeholder.markdown(clean_response(raw_output) + "▌")

            final_output = clean_response(raw_output)
            response_placeholder.markdown(final_output)

        except Exception as e:
            st.error(f"Error: {e}")

        st.session_state.messages.append(AIMessage(content=final_output))
