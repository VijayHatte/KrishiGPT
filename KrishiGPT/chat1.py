"""
chat1.py
--------
Knowledge-base ingestion + FAISS vector store for the KrishiGPT RAG pipeline.

Open-source stack:
  * Embeddings   : sentence-transformers `all-MiniLM-L6-v2` (free, local).
  * Vector store : FAISS (Facebook AI Similarity Search) -- fast, in-memory,
                   and fully open-source.

Content is pulled from agriculture websites and PDF handbooks, chunked, then
embedded into FAISS for similarity retrieval at query time.
"""

import requests
import PyPDF2
from itertools import chain

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


def fetch_website_content(url: str) -> str:
    """Fetch raw HTML/text content from a website URL."""
    response = requests.get(url, timeout=30)
    return response.text


def extract_pdf_text(pdf_file: str) -> str:
    """Extract concatenated text from every page of a PDF file."""
    with open(pdf_file, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100):
    """Split a long document into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def initialize_vector_store(contents):
    """Embed all content chunks into a FAISS vector store and return it."""
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    chunks = list(chain.from_iterable(split_text(c) for c in contents if c))
    db = FAISS.from_texts(chunks, embedding_function)
    return db
