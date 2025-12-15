import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document

def ocr_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)

    documents = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)

        documents.append(
            Document(
                page_content=text,
                metadata={"page": i + 1}
            )
        )

    return documents
