"""
PDF-QA-Bot - Prompts Module
Defines LangChain PromptTemplates for PDF document analysis and context-grounded Q&A.
"""

from langchain_core.prompts import PromptTemplate


# 1. Summary Prompt
SUMMARY_PROMPT_TEMPLATE = """You are an expert Document Analyst.
Provide a clear, concise executive summary of the following PDF document content.

Document Content:
---
{pdf_text}
---

Instructions:
- Summarize the main purpose, background, and core message in 3-4 sentences.
- Output ONLY the summary text.
"""

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["pdf_text"],
    template=SUMMARY_PROMPT_TEMPLATE
)


# 2. Topic Extraction Prompt
TOPICS_PROMPT_TEMPLATE = """Extract 4-6 primary topics or themes covered in the following PDF document.

Document Content:
---
{pdf_text}
---

Return ONLY a valid JSON list of topic strings:
["Topic 1", "Topic 2", "Topic 3", "Topic 4"]
"""

TOPICS_PROMPT = PromptTemplate(
    input_variables=["pdf_text"],
    template=TOPICS_PROMPT_TEMPLATE
)


# 3. Keyword Extraction Prompt
KEYWORDS_PROMPT_TEMPLATE = """Extract 6-10 important technical keywords and domain terms from the following PDF document.

Document Content:
---
{pdf_text}
---

Return ONLY a valid JSON list of keyword strings:
["Keyword 1", "Keyword 2", "Keyword 3", "Keyword 4", "Keyword 5"]
"""

KEYWORDS_PROMPT = PromptTemplate(
    input_variables=["pdf_text"],
    template=KEYWORDS_PROMPT_TEMPLATE
)


# 4. Key Points Extraction Prompt
KEY_POINTS_PROMPT_TEMPLATE = """Extract 4-6 essential key takeaways or main points from the following PDF document.

Document Content:
---
{pdf_text}
---

Return ONLY a valid JSON list of key point strings:
["Key point 1", "Key point 2", "Key point 3", "Key point 4"]
"""

KEY_POINTS_PROMPT = PromptTemplate(
    input_variables=["pdf_text"],
    template=KEY_POINTS_PROMPT_TEMPLATE
)


# 5. Question Generation Prompt
QUESTION_GENERATION_PROMPT_TEMPLATE = """Read the supplied document content and generate 5 to 10 meaningful, specific questions that can be answered directly from the document.
Cover the most important concepts and core information.
Do not use outside information.

Document Content:
---
{pdf_text}
---

Return ONLY a valid JSON list of question strings:
[
  "Question 1?",
  "Question 2?",
  "Question 3?",
  "Question 4?",
  "Question 5?"
]
"""

QUESTION_GENERATION_PROMPT = PromptTemplate(
    input_variables=["pdf_text"],
    template=QUESTION_GENERATION_PROMPT_TEMPLATE
)


# 6. Context-Grounded Q&A Answer Prompt
QA_PROMPT_TEMPLATE = """You are a strict Document Q&A Assistant.
Answer the user's question using ONLY the provided PDF context below.
If the answer is not present in the context, say exactly:
"The answer could not be found in the uploaded PDF."
Do not invent or hallucinate information outside the provided PDF context.

PDF Context:
---
{context}
---

User Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=QA_PROMPT_TEMPLATE
)
