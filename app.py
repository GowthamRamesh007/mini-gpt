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

st.set_page_config(page_title="PDF Q&A Chatbot", layout="wide")
st.title("📄 PDF Q&A Chatbot (Qwen)")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []
if "task" not in st.session_state:
    st.session_state.task = None
if "length" not in st.session_state:
    st.session_state.length = None
if "generated_questions" not in st.session_state:
    st.session_state.generated_questions = None

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

# Layout: left (PDF uploader + tasks) | right (generated content)
col_pdf, col_display = st.columns([1, 3])

# --- PDF UPLOADER (LEFT) ---
with col_pdf:
    st.markdown("### 📂 Upload PDF")
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if uploaded_file:
        text = ""
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.pdf_chunks = splitter.split_text(text)

        st.success(f"✅ PDF split into {len(st.session_state.pdf_chunks)} parts")

        # Choose task
        st.markdown("### 📝 Select Task")
        if st.button("Generate Questions"):
            st.session_state.task = "questions"
            st.session_state.length = None
            relevant = " ".join(st.session_state.pdf_chunks[:5])

            prompt = f"Generate 8-10 meaningful exam-style questions based on the following PDF content:\n\n{relevant}"
            with st.spinner("🔎 Generating questions..."):
                response = model.invoke([HumanMessage(content=prompt)])
                st.session_state.generated_questions = response.content

        if st.button("Summarize PDF"):
            st.session_state.task = "summary"
            st.session_state.length = None
            st.session_state.generated_questions = None

# --- DISPLAY AREA (MIDDLE/RIGHT) ---
with col_display:
    if st.session_state.task == "questions" and st.session_state.generated_questions:
        st.markdown("## 📌 Generated Questions")
        st.markdown(st.session_state.generated_questions)

        st.markdown("### ⏱ Select Answer Length")
        if st.button("2m (short ~50 words)"):
            st.session_state.length = "short"
        if st.button("13m (medium ~130 words)"):
            st.session_state.length = "medium"
        if st.button("15m (long ~300 words)"):
            st.session_state.length = "long"

    elif st.session_state.task == "summary":
        st.markdown("## 📄 PDF Summary will be generated when you ask a question.")

    else:
        st.markdown("## 💬 Ask anything about the PDF or general questions")

    # Display chat history
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

# --- USER INPUT AT BOTTOM ---
st.markdown("---")
user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        raw_output = ""

        try:
            # Decide prompt
            if st.session_state.task and st.session_state.pdf_chunks:
                relevant = " ".join(st.session_state.pdf_chunks[:5])  # take first few chunks

                if st.session_state.task == "summary":
                    task_prompt = "Summarize this PDF content."
                elif st.session_state.task == "questions" and st.session_state.length:
                    task_prompt = "Answer this question based on the PDF."
                else:
                    task_prompt = "Answer the following question."

                # Add word length condition
                if st.session_state.length == "short":
                    length_prompt = "Write the answer in about 50 words (3 lines)."
                elif st.session_state.length == "medium":
                    length_prompt = "Write the answer in about 130 words (10 lines)."
                elif st.session_state.length == "long":
                    length_prompt = "Write the answer in about 300 words (20 lines)."
                else:
                    length_prompt = ""

                prompt = f"{task_prompt}\n\nPDF Content:\n{relevant}\n\n{length_prompt}\n\nQuestion: {user_input}"
            else:
                prompt = user_input  # general questions

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
