"""
PDF-QA-Bot - Streamlit Web Application Interface
Clean, beginner-friendly UI for uploading PDFs, viewing PDF analysis,
clicking generated questions, asking custom questions, and viewing RAG source citations.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from qa_graph import run_pdf_analysis_pipeline, run_qa_pipeline

load_dotenv()

# Page Setup
st.set_page_config(
    page_title="AI PDF Analyzer & Q&A Bot",
    page_icon="📚",
    layout="wide"
)

# Custom Glassmorphic Styling
st.markdown("""
<style>
    .main-title {
        color: #1E293B;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 4px;
    }
    .sub-title {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 24px;
    }
    .answer-box {
        background-color: #F0FDF4;
        border-left: 5px solid #22C55E;
        padding: 18px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 16px;
    }
    .source-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .keyword-pill {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin: 3px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "graph_state" not in st.session_state:
    st.session_state["graph_state"] = None
if "current_question" not in st.session_state:
    st.session_state["current_question"] = ""
if "qa_answer" not in st.session_state:
    st.session_state["qa_answer"] = None
if "qa_sources" not in st.session_state:
    st.session_state["qa_sources"] = []


def execute_qa(question: str):
    """Executes Q&A pipeline for a question and stores results in session state."""
    gstate = st.session_state.get("graph_state")
    if not gstate or not question.strip():
        return

    with st.spinner(f"Retrieving context & generating answer for: '{question}'..."):
        updated_state = run_qa_pipeline(gstate, question)
        st.session_state["graph_state"] = updated_state
        st.session_state["current_question"] = question
        st.session_state["qa_answer"] = updated_state.get("answer")
        st.session_state["qa_sources"] = updated_state.get("sources", [])


def main():
    # Sidebar Configuration
    st.sidebar.title("⚙️ Engine Settings")
    provider = st.sidebar.selectbox(
        "Select LLM Provider",
        options=["gemini", "openai", "offline"],
        index=0,
        help="Google Gemini, OpenAI, or Offline Fallback Mode"
    )

    api_key = st.sidebar.text_input(
        "API Key (Optional if set in .env)",
        value=os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        help="Enter GEMINI_API_KEY or OPENAI_API_KEY"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### ℹ️ How it works
    1. **Upload PDF(s)** (Single or Multiple)
    2. **Analyze PDF**: LangGraph extracts text, splits chunks, creates FAISS embeddings, and generates summary, topics, keywords & 5-10 questions.
    3. **Q&A**: Click any generated question OR type your own question.
    4. **Context & Sources**: RAG retrieves relevant PDF chunks and cites exact filename & page numbers!
    """)

    # Main Header
    st.markdown('<div class="main-title">📚 AI PDF Analyzer & Q&A Bot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload PDFs to get an instant executive summary, key topics, auto-generated questions, and context-grounded Q&A using <strong>LangChain</strong> & <strong>LangGraph</strong>.</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # PDF Upload & Processing Section
    # -----------------------------------------------------------------
    uploaded_files = st.file_uploader(
        "Choose one or multiple PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload single or multiple PDF documents"
    )

    if st.button("🚀 Analyze PDF", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("⚠️ Please upload at least one PDF file before clicking Analyze PDF.")
            return

        with st.spinner("Processing PDF, building FAISS vector store, and running LangGraph pipeline..."):
            final_state = run_pdf_analysis_pipeline(
                uploaded_files=uploaded_files,
                provider=provider,
                api_key=api_key
            )

            if final_state.get("error_message"):
                st.error(f"❌ {final_state['error_message']}")
                return

            st.session_state["graph_state"] = final_state
            st.session_state["current_question"] = ""
            st.session_state["qa_answer"] = None
            st.session_state["qa_sources"] = []
            st.success("✅ PDF Analysis complete! Explore the analysis and questions below.")

    gstate = st.session_state.get("graph_state")

    if gstate and gstate.get("summary"):
        st.markdown("---")

        # -------------------------------------------------------------
        # Section 1: PDF Analysis
        # -------------------------------------------------------------
        st.markdown("## 📄 PDF Analysis")

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("### Executive Summary")
            st.write(gstate.get("summary", "Summary unavailable."))

            st.markdown("### 🔑 Important Keywords")
            kws = gstate.get("keywords", [])
            if kws:
                for kw in kws:
                    st.markdown(f'<span class="keyword-pill">{kw}</span>', unsafe_allow_html=True)

        with col_a2:
            st.markdown("### 📌 Main Topics")
            topics = gstate.get("topics", [])
            for t in topics:
                st.markdown(f"- 🔹 **{t}**")

            st.markdown("### 💡 Key Takeaways & Points")
            points = gstate.get("key_points", [])
            for p in points:
                st.markdown(f"- ✨ {p}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        # -------------------------------------------------------------
        # Section 2: Questions Based on PDF (Clickable Buttons)
        # -------------------------------------------------------------
        st.markdown("## ❓ Questions Based on This PDF")
        st.write("Click any generated question below to automatically retrieve its answer from the PDF:")

        gen_qs = gstate.get("generated_questions", [])
        if gen_qs:
            for idx, q_text in enumerate(gen_qs, start=1):
                if st.button(f"{idx}. {q_text}", key=f"gen_q_{idx}", use_container_width=True):
                    execute_qa(q_text)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        # -------------------------------------------------------------
        # Section 3: Custom Question Form
        # -------------------------------------------------------------
        st.markdown("## 💬 Ask Your Own Question")

        custom_q = st.text_input(
            "Type any question about the uploaded PDF:",
            key="custom_q_input",
            placeholder="e.g., What are the main findings mentioned in the document?"
        )

        if st.button("Ask Question 🔍", type="primary", key="btn_ask_custom"):
            if not custom_q.strip():
                st.warning("Please type a question before clicking Ask Question.")
            else:
                execute_qa(custom_q)

        # -------------------------------------------------------------
        # Section 4: Answer & Source Display
        # -------------------------------------------------------------
        if st.session_state.get("qa_answer"):
            st.markdown("<br>", unsafe_allow_html=True)
            active_q = st.session_state.get("current_question", "Question")
            st.markdown(f"## 🤖 Answer for: *\"{active_q}\"*")

            ans_text = st.session_state["qa_answer"]
            if "could not be found" in ans_text.lower():
                st.warning(f"⚠️ {ans_text}")
            else:
                st.markdown(f"""
                <div class="answer-box">
                    <p style="font-size: 1.1rem; line-height: 1.6; color: #14532D; margin:0;">
                        {ans_text}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Display Sources
            sources = st.session_state.get("qa_sources", [])
            if sources and "could not be found" not in ans_text.lower():
                st.markdown("### 📖 Source Citations & Relevant Snippets")
                for s_idx, src in enumerate(sources, start=1):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {s_idx}:</strong> <code>{src.get('source')}</code> | <strong>Page:</strong> {src.get('page')}
                        <br><em style="color:#1E40AF;">"{src.get('snippet')}"</em>
                    </div>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
