"""
PDF-QA-Bot - PDF Processor Module
Handles text extraction from PDF files and document splitting into chunked passages.
Preserves page numbers and file source metadata.
"""

import io
from typing import List, Tuple
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pdf_documents(uploaded_files) -> List[Document]:
    """
    Extracts text page-by-page from single or multiple uploaded PDF files.
    Wraps each page into a LangChain Document with source filename and page metadata.
    """
    documents = []

    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        file_bytes = uploaded_file.read()
        
        # Reset file pointer for re-read if needed
        uploaded_file.seek(0)
        
        try:
            pdf_file = io.BytesIO(file_bytes)
            reader = PdfReader(pdf_file)
            
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    doc = Document(
                        page_content=page_text.strip(),
                        metadata={
                            "source": filename,
                            "page": page_num,
                            "total_pages": len(reader.pages)
                        }
                    )
                    documents.append(doc)
        except Exception as e:
            print(f"[PDFProcessor] Error reading {filename}: {e}")

    return documents


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits page-level documents into smaller chunked Document passages using RecursiveCharacterTextSplitter.
    Preserves metadata (source filename and page number).
    """
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def get_combined_pdf_text(documents: List[Document]) -> str:
    """
    Combines text contents from all document pages into a single raw text string.
    """
    return "\n\n".join([doc.page_content for doc in documents])
