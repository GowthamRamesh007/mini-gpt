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
    return ChatOpenAI(
        model="qwen/qwq-32b:free",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.0,
        streaming=True,
    )

model = get_model()

# --- SPACE + CENTERED USER INPUT ---
st.markdown("### 💬 Ask your question")

col1, col2, col3 = st.columns([1, 2, 1])  # center alignment
with col2:
    user_input = st.text_input("Type your question 👇", placeholder="Ask something about the PDF...")
    ask_button = st.button("Ask")

# --- PROCESS USER INPUT ---
if ask_button and user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        raw_output = ""

        try:
            # Choose relevant chunk
            if st.session_state.pdf_chunks:
                relevant = max(
                    st.session_state.pdf_chunks,
                    key=lambda chunk: sum(word.lower() in chunk.lower() for word in user_input.split())
                )
                prompt = f"Answer using this PDF content:\n\n{relevant}\n\nQuestion: {user_input}"
            else:
                prompt = user_input

            st.session_state.messages.append(HumanMessage(content=prompt))

            # Stream + clean response
            for chunk in model.stream(st.session_state.messages):
                raw_output += chunk.content or ""
                cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
                response_placeholder.markdown(cleaned + "▌")

            final_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
            response_placeholder.markdown(final_output)

        except Exception as e:
            st.error(f"Error: {e}")

        st.session_state.messages.append(AIMessage(content=final_output))

# --- PDF UPLOADER BELOW INPUT ---
st.markdown("---")
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
