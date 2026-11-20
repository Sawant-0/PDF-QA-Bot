"""
PDF-QA-Bot - LangGraph Q&A Workflow Graph
StateGraph orchestration module managing PDF processing, analysis, vector retrieval, and conditional Q&A routing.
"""

from typing import Dict, Any, List, Optional, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from pdf_processor import extract_pdf_documents, split_documents, get_combined_pdf_text
from vector_store import create_vector_store, retrieve_relevant_chunks
from pdf_analyzer import (
    get_llm,
    generate_summary,
    extract_topics,
    extract_keywords,
    extract_key_points,
    generate_questions_from_pdf,
    answer_question_with_context
)


# -------------------------------------------------------------------
# Shared LangGraph State
# -------------------------------------------------------------------

class PDFQAState(TypedDict):
    """
    Shared state dictionary passed across all nodes in the LangGraph workflow.
    """
    uploaded_files: Any
    pdf_text: str
    documents: List[Any]
    chunks: List[Any]
    vectorstore: Any
    
    # PDF Analysis Outputs
    summary: str
    topics: List[str]
    keywords: List[str]
    key_points: List[str]
    generated_questions: List[str]
    
    # Q&A State
    user_question: Optional[str]
    retrieved_documents: List[Any]
    context: str
    has_relevant_context: bool
    answer: str
    sources: List[Dict[str, Any]]
    
    # Settings & Error Handling
    llm_provider: str
    api_key: str
    error_message: Optional[str]


# -------------------------------------------------------------------
# Node Definitions
# -------------------------------------------------------------------

def node_extract_pdf(state: PDFQAState) -> PDFQAState:
    """Node 1: Extract text and chunks from PDF uploads."""
    uploaded_files = state.get("uploaded_files")
    if not uploaded_files:
        state["error_message"] = "No PDF file uploaded."
        return state

    docs = extract_pdf_documents(uploaded_files)
    if not docs:
        state["error_message"] = "Could not extract readable text from PDF."
        return state

    chunks = split_documents(docs)
    combined_text = get_combined_pdf_text(docs)

    state["documents"] = docs
    state["chunks"] = chunks
    state["pdf_text"] = combined_text

    # Build FAISS vector store
    try:
        vs = create_vector_store(chunks, provider=state.get("llm_provider", "gemini"), api_key=state.get("api_key", ""))
        state["vectorstore"] = vs
    except Exception as e:
        print(f"[qa_graph] Vector store creation warning: {e}")

    return state


def node_analyze_pdf(state: PDFQAState) -> PDFQAState:
    """Node 2: Analyze PDF using LangChain LLM chain (Summary, Topics, Key Points, Keywords)."""
    pdf_text = state.get("pdf_text", "")
    if not pdf_text:
        return state

    llm = get_llm(provider=state.get("llm_provider", "gemini"), api_key=state.get("api_key", ""))

    state["summary"] = generate_summary(pdf_text, llm=llm)
    state["topics"] = extract_topics(pdf_text, llm=llm)
    state["keywords"] = extract_keywords(pdf_text, llm=llm)
    state["key_points"] = extract_key_points(pdf_text, llm=llm)

    return state


def node_generate_questions(state: PDFQAState) -> PDFQAState:
    """Node 3: Automatically generate 5-10 questions based ONLY on uploaded PDF."""
    pdf_text = state.get("pdf_text", "")
    if not pdf_text:
        return state

    llm = get_llm(provider=state.get("llm_provider", "gemini"), api_key=state.get("api_key", ""))
    state["generated_questions"] = generate_questions_from_pdf(pdf_text, llm=llm)
    return state


def node_retrieve_context(state: PDFQAState) -> PDFQAState:
    """Node 4: Retrieve relevant PDF chunks for user_question using FAISS retriever."""
    question = state.get("user_question", "")
    vs = state.get("vectorstore")
    all_chunks = state.get("chunks", [])

    if not question or (vs is None and not all_chunks):
        state["retrieved_documents"] = []
        state["context"] = ""
        return state

    retrieved = retrieve_relevant_chunks(vs, question, chunks=all_chunks, top_k=4)
    state["retrieved_documents"] = retrieved

    context_snippets = [doc.page_content for doc in retrieved]
    state["context"] = "\n\n---\n\n".join(context_snippets)

    # Extract source metadata
    sources = []
    for doc in retrieved:
        sources.append({
            "source": doc.metadata.get("source", "PDF"),
            "page": doc.metadata.get("page", 1),
            "snippet": doc.page_content[:180] + "..."
        })
    state["sources"] = sources

    return state


def node_check_context(state: PDFQAState) -> PDFQAState:
    """Node 5: Evaluates whether relevant information was found in the retrieved PDF context."""
    context = state.get("context", "").strip()

    # Heuristic relevance check
    has_rel = bool(context and len(context) > 20)
    state["has_relevant_context"] = has_rel
    return state


def node_generate_answer(state: PDFQAState) -> PDFQAState:
    """Node 6: Generate answer using RAG QA prompt + context + user question."""
    context = state.get("context", "")
    question = state.get("user_question", "")
    llm = get_llm(provider=state.get("llm_provider", "gemini"), api_key=state.get("api_key", ""))

    answer = answer_question_with_context(context, question, llm=llm)
    state["answer"] = answer
    return state


def node_not_found_message(state: PDFQAState) -> PDFQAState:
    """Node 7: Fallback node when information is not present in PDF."""
    state["answer"] = "The answer could not be found in the uploaded PDF."
    state["sources"] = []
    return state


# -------------------------------------------------------------------
# Conditional Router Function
# -------------------------------------------------------------------

def route_after_context_check(state: PDFQAState) -> Literal["generate_answer", "not_found_message"]:
    """
    Conditional edge decision maker:
    If relevant context exists in PDF -> route to generate_answer.
    If relevant context does NOT exist -> route to not_found_message.
    """
    if state.get("has_relevant_context"):
        return "generate_answer"
    return "not_found_message"


# -------------------------------------------------------------------
# Build LangGraph Workflow Graphs
# -------------------------------------------------------------------

def build_pdf_qa_graph():
    """
    Constructs and compiles the PDF processing & Q&A LangGraph StateGraph.
    """
    workflow = StateGraph(PDFQAState)

    # Add Nodes
    workflow.add_node("extract_pdf", node_extract_pdf)
    workflow.add_node("analyze_pdf", node_analyze_pdf)
    workflow.add_node("generate_questions", node_generate_questions)
    workflow.add_node("retrieve_context", node_retrieve_context)
    workflow.add_node("check_context", node_check_context)
    workflow.add_node("generate_answer", node_generate_answer)
    workflow.add_node("not_found_message", node_not_found_message)

    # Execution Sequence
    workflow.set_entry_point("extract_pdf")
    workflow.add_edge("extract_pdf", "analyze_pdf")
    workflow.add_edge("analyze_pdf", "generate_questions")

    # Q&A Branch
    workflow.add_edge("generate_questions", "retrieve_context")
    workflow.add_edge("retrieve_context", "check_context")

    # Conditional Edge Routing
    workflow.add_conditional_edges(
        "check_context",
        route_after_context_check,
        {
            "generate_answer": "generate_answer",
            "not_found_message": "not_found_message"
        }
    )

    workflow.add_edge("generate_answer", END)
    workflow.add_edge("not_found_message", END)

    return workflow.compile()


# Compile graph instance
pdf_qa_app_graph = build_pdf_qa_graph()


def run_pdf_analysis_pipeline(uploaded_files, provider: str = "gemini", api_key: str = "") -> PDFQAState:
    """Executes initial PDF extraction, analysis, and question generation."""
    initial_state: PDFQAState = {
        "uploaded_files": uploaded_files,
        "pdf_text": "",
        "documents": [],
        "chunks": [],
        "vectorstore": None,
        "summary": "",
        "topics": [],
        "keywords": [],
        "key_points": [],
        "generated_questions": [],
        "user_question": None,
        "retrieved_documents": [],
        "context": "",
        "has_relevant_context": False,
        "answer": "",
        "sources": [],
        "llm_provider": provider,
        "api_key": api_key,
        "error_message": None
    }

    try:
        final_state = pdf_qa_app_graph.invoke(initial_state)
        return final_state
    except Exception as e:
        print(f"[qa_graph] Pipeline invocation error: {e}")
        initial_state["error_message"] = f"Pipeline error: {str(e)}"
        return initial_state


def run_qa_pipeline(current_state: PDFQAState, question: str) -> PDFQAState:
    """Executes vector retrieval, context check, and answer generation for a user question."""
    current_state["user_question"] = question
    try:
        # Execute retrieve_context -> check_context -> conditional route -> answer
        retrieved_state = node_retrieve_context(current_state)
        checked_state = node_check_context(retrieved_state)
        
        target_route = route_after_context_check(checked_state)
        if target_route == "generate_answer":
            final_state = node_generate_answer(checked_state)
        else:
            final_state = node_not_found_message(checked_state)
            
        return final_state
    except Exception as e:
        print(f"[qa_graph] Q&A error: {e}")
        current_state["answer"] = f"Error during Q&A: {str(e)}"
        return current_state
