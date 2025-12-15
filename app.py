import os
import streamlit as st

from ocr_utils import ocr_pdf
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- CONFIG ----------------
DB_DIR = "chroma_langchain_db"

st.set_page_config(
    page_title="PDF RAG with Ollama",
    layout="wide"
)

st.title("📄 PDF Question Answering (RAG + Ollama)")

# ---------------- MODELS ----------------
llm = OllamaLLM(model="llama3")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    docs = []

    try:
        # Try normal PDF text extraction
        loader = PyPDFLoader("temp.pdf")
        pdf_docs = loader.load()

        # If text exists → use normal loader
        if any(doc.page_content.strip() for doc in pdf_docs):
            docs.extend(pdf_docs)
        else:
            # Image-based PDF → OCR
            docs.extend(ocr_pdf("temp.pdf"))

    except Exception:
        # Fallback to OCR if loader fails
        docs.extend(ocr_pdf("temp.pdf"))

    # Safety check
    if not docs:
        st.error("❌ No text could be extracted from this PDF.")
        st.stop()

    # ---------------- TEXT SPLITTING ----------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # ---------------- VECTOR STORE ----------------
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    st.success("✅ PDF processed successfully (Text + Scanned supported)")

# ---------------- QUESTION ----------------
query = st.text_input("Ask a question from the PDF:")

if query:
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{query}
"""

    answer = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(answer)
