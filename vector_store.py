"""
PDF-QA-Bot - Vector Store & Retriever Module
Generates embeddings and builds a FAISS vector index for fast semantic document retrieval.
Includes robust text search fallback.
"""

import os
import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


class DeterministicSimpleEmbeddings(Embeddings):
    """
    Lightweight, deterministic character/word n-gram embedding fallback.
    Ensures FAISS vector retrieval works offline without external API keys or heavy models.
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> List[float]:
        vec = [0.0] * 128
        words = text.lower().split()
        for word in words:
            idx = sum(ord(c) for c in word) % 128
            vec[idx] += 1.0
        norm = sum(x*x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


def get_embeddings(provider: str = "gemini", api_key: str = "") -> Embeddings:
    """
    Returns embedding model instance based on provider and available API keys.
    """
    selected_provider = provider.lower()

    if selected_provider == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        if key:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                return GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=key
                )
            except Exception as e:
                print(f"[VectorStore] Gemini embeddings init failed: {e}. Using fallback.")

    elif selected_provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        if key:
            try:
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=key
                )
            except Exception as e:
                print(f"[VectorStore] OpenAI embeddings init failed: {e}. Using fallback.")

    return DeterministicSimpleEmbeddings()


def create_vector_store(chunks: List[Document], provider: str = "gemini", api_key: str = "") -> FAISS:
    """
    Builds an in-memory FAISS vector index from chunked Document passages.
    """
    if not chunks:
        raise ValueError("Cannot create vector store from empty document chunks.")

    embeddings = get_embeddings(provider=provider, api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def retrieve_relevant_chunks(vector_store: Optional[FAISS], query: str, chunks: Optional[List[Document]] = None, top_k: int = 4) -> List[Document]:
    """
    Retrieves top_k most semantically relevant Document chunks for a user query.
    Includes fallback keyword search across chunks if vector search is empty.
    """
    if not query or not query.strip():
        return []

    docs = []
    if vector_store is not None:
        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.invoke(query)
        except Exception as e:
            print(f"[VectorStore] Vector retrieval error: {e}")

    # Fallback to text matching if vector retrieval returned no results
    if not docs and chunks:
        query_words = set(re.findall(r'\w+', query.lower())) - {"what", "is", "the", "a", "an", "of", "and", "in", "to", "how", "why", "who", "where"}
        scored_chunks = []
        for chunk in chunks:
            chunk_words = set(re.findall(r'\w+', chunk.page_content.lower()))
            overlap = len(query_words.intersection(chunk_words))
            scored_chunks.append((overlap, chunk))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        docs = [chunk for score, chunk in scored_chunks[:top_k] if score > 0]
        if not docs and chunks:
            docs = chunks[:top_k]

    return docs
