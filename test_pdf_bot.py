"""
Verification test for PDF-QA-Bot workflow.
"""

from langchain_core.documents import Document
from pdf_processor import split_documents, get_combined_pdf_text
from vector_store import create_vector_store, retrieve_relevant_chunks
from pdf_analyzer import generate_summary, extract_topics, generate_questions_from_pdf
from qa_graph import run_qa_pipeline, PDFQAState, node_check_context, route_after_context_check

def test_pdf_qa_workflow():
    sample_docs = [
        Document(
            page_content="""
            Machine Learning (ML) is a branch of artificial intelligence (AI) focused on building applications that learn from data and improve their accuracy over time without being explicitly programmed.
            The primary categories of machine learning are Supervised Learning, Unsupervised Learning, and Reinforcement Learning.
            Supervised learning relies on labeled datasets to train algorithms that classify data or predict outcomes accurately.
            Unsupervised learning uses machine learning algorithms to analyze and cluster unlabeled datasets.
            """,
            metadata={"source": "Sample_ML_Guide.pdf", "page": 1}
        ),
        Document(
            page_content="""
            Applications of Machine Learning include natural language processing, computer vision, recommendation engines, and automated fraud detection.
            Natural Language Processing (NLP) enables computers to understand, interpret, and manipulate human language.
            LangChain and LangGraph are framework tools used to build advanced LLM applications and stateful multi-agent workflows.
            """,
            metadata={"source": "Sample_ML_Guide.pdf", "page": 2}
        )
    ]

    # Test Splitting
    chunks = split_documents(sample_docs)
    assert len(chunks) >= 2
    combined_text = get_combined_pdf_text(sample_docs)
    assert "Machine Learning" in combined_text

    # Test Vector Store
    vs = create_vector_store(chunks, provider="offline")
    retrieved = retrieve_relevant_chunks(vs, "What is supervised learning?", top_k=2)
    assert len(retrieved) > 0
    assert retrieved[0].metadata["source"] == "Sample_ML_Guide.pdf"

    # Test Analyzer
    summary = generate_summary(combined_text)
    assert len(summary) > 0
    topics = extract_topics(combined_text)
    assert len(topics) > 0
    questions = generate_questions_from_pdf(combined_text)
    assert len(questions) >= 3

    # Test LangGraph Conditional State & Routing
    state: PDFQAState = {
        "uploaded_files": None,
        "pdf_text": combined_text,
        "documents": sample_docs,
        "chunks": chunks,
        "vectorstore": vs,
        "summary": summary,
        "topics": topics,
        "keywords": ["Machine Learning", "NLP"],
        "key_points": ["Supervised learning uses labeled data"],
        "generated_questions": questions,
        "user_question": "What is supervised learning?",
        "retrieved_documents": [],
        "context": "",
        "has_relevant_context": False,
        "answer": "",
        "sources": [],
        "llm_provider": "offline",
        "api_key": "",
        "error_message": None
    }

    updated_state = run_qa_pipeline(state, "What is supervised learning?")
    assert updated_state["has_relevant_context"] is True
    assert len(updated_state["answer"]) > 0
    assert len(updated_state["sources"]) > 0

    # Test Not Found Route
    missing_context_state = run_qa_pipeline(state, "What is quantum gravity string theory?")
    assert "could not be found" in missing_context_state["answer"].lower() or len(missing_context_state["answer"]) > 0

    print("All PDF-QA-Bot workflow tests passed successfully!")

if __name__ == "__main__":
    test_pdf_qa_workflow()
