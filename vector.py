import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pytesseract
from PIL import Image
import fitz  # PyMuPDF for reading image-based PDFs

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

def extract_text_with_ocr(pdf_path):
    """Extract text from scanned/image PDFs using OCR"""
    text = ""
    pdf = fitz.open(pdf_path)
    for page in pdf:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)
    pdf.close()
    return text

if add_documents:
    documents = []
    ids = []

    # --- 1️⃣ CSV loader (optional) ---
    if os.path.exists("realistic_restaurant_reviews.csv"):
        df = pd.read_csv("realistic_restaurant_reviews.csv")
        for i, row in df.iterrows():
            document = Document(
                page_content=row["Title"] + " " + row["Review"],
                metadata={"rating": row["Rating"], "date": row["Date"]},
                id=str(i)
            )
            ids.append(str(i))
            documents.append(document)

    # --- 2️⃣ Load ALL PDFs (text + image based) ---
    pdf_files = [f for f in os.listdir() if f.lower().endswith(".pdf")]
    pdf_index = len(ids)

    for pdf_file in pdf_files:
        print(f"📄 Processing PDF: {pdf_file}")

        # Try reading as text first
        loader = PyPDFLoader(pdf_file)
        pdf_docs = loader.load()

        # If no text found → run OCR
        if not pdf_docs or all(not d.page_content.strip() for d in pdf_docs):
            print(f"🔍 No text found — using OCR for {pdf_file}")
            text = extract_text_with_ocr(pdf_file)
            pdf_docs = [Document(page_content=text, metadata={"source": pdf_file})]

        for doc in pdf_docs:
            doc.metadata["source"] = pdf_file
            documents.append(doc)
            ids.append(str(pdf_index))
            pdf_index += 1

# --- 3️⃣ Store embeddings ---
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    print("🧠 Adding documents to vector database...")
    vector_store.add_documents(documents=documents, ids=ids)
    print("✅ Vector database created successfully!")

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
