"""
app/prompts/templates.py
─────────────────────────
All LLM prompt templates in one place.
Keeping prompts separate from logic makes them easy to iterate.
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ─────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are HR PolicyBot, a knowledgeable and professional \
AI assistant for answering questions about company HR policies, leave rules, \
codes of conduct, and onboarding guidelines.

INSTRUCTIONS:
1. Answer ONLY based on the provided context from HR documents.
2. If the answer is not in the context, say clearly:
   "I couldn't find that information in the available HR documents."
3. Always cite the source document at the end of your answer.
4. Keep answers concise, professional, and actionable.
5. Never fabricate policies or rules not present in the documents.
6. If a question is ambiguous, ask for clarification.

CITATION FORMAT:
End every answer with:
📄 Source: [document name], Page [number] (if available)
"""


# ─────────────────────────────────────────────────────
# RAG Prompt — used by the ConversationalRetrievalChain
# ─────────────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """Use the following retrieved HR policy context to \
answer the employee's question. Be precise and professional.

Context:
{context}

Question: {question}

Answer (cite the source document at the end):"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)


# ─────────────────────────────────────────────────────
# Condense Question Prompt — rewrites follow-up questions
# using conversation history so the retriever gets full context
# ─────────────────────────────────────────────────────

CONDENSE_QUESTION_TEMPLATE = """Given the following conversation history \
and a follow-up question, rephrase the follow-up question to be a \
standalone question that contains all necessary context.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=CONDENSE_QUESTION_TEMPLATE,
)


# ─────────────────────────────────────────────────────
# No-Context Response — when retrieval returns nothing
# ─────────────────────────────────────────────────────

NO_CONTEXT_RESPONSE = (
    "I couldn't find relevant information in the HR documents to answer "
    "your question. Please contact the HR department directly or rephrase "
    "your question with more specific terms."
)
