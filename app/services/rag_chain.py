"""
app/services/rag_chain.py
──────────────────────────
Core RAG pipeline using modern LangChain (0.2+) LCEL-style approach.
Replaces deprecated ConversationalRetrievalChain with explicit chain steps.
"""

from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.core.config import get_settings
from app.core.logger import get_logger
from app.services.vector_store import VectorStoreManager
from app.services.chat_memory import ConversationMemory
from app.prompts.templates import NO_CONTEXT_RESPONSE

logger = get_logger(__name__)
settings = get_settings()

RAG_PROMPT = PromptTemplate.from_template(
    """You are HR PolicyBot, a professional AI assistant for HR policy questions.
Answer ONLY based on the context below. If the answer is not in the context,
say: "I couldn't find that information in the available HR documents."
Always cite the source document at the end of your answer.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer (cite source at the end with: 📄 Source: [filename], Page [number]):"""
)


class RAGChain:
    """
    Manages the full RAG pipeline using modern LangChain LCEL.

    Attributes:
        vector_store: Source of retrieved document chunks.
        memory_manager: Per-session conversation history.
        llm: OpenAI chat model.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        memory_manager: ConversationMemory,
    ) -> None:
        self.vector_store = vector_store
        self.memory_manager = memory_manager
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.1,
            max_tokens=settings.max_tokens,
        )
        logger.info("RAGChain initialised.")

    def query(self, question: str, session_id: str) -> dict[str, Any]:
        """
        Process a question through the full RAG pipeline.

        Args:
            question: User's natural language question.
            session_id: Unique identifier for conversation continuity.

        Returns:
            Dict with keys: answer, sources, session_id.

        Raises:
            RuntimeError: If no documents have been ingested yet.
        """
        if self.vector_store.collection_is_empty():
            raise RuntimeError(
                "No documents found in the vector store. "
                "Please ingest HR documents first via POST /ingest."
            )

        logger.info(f"Query | session='{session_id}' q='{question[:80]}'")

        # 1. Retrieve relevant chunks
        retriever = self.vector_store.get_retriever()
        source_docs = retriever.invoke(question)

        # 2. Format context from retrieved chunks
        context = "\n\n".join(
            f"[{doc.metadata.get('source', 'Unknown')}, "
            f"Page {doc.metadata.get('page', 0)}]\n{doc.page_content}"
            for doc in source_docs
        )

        # 3. Format conversation history
        history_tuples = self.memory_manager.format_history_as_tuples(session_id)
        chat_history = "\n".join(
            f"Human: {h}\nAssistant: {a}"
            for h, a in history_tuples
        ) if history_tuples else "No prior conversation."

        # 4. Call LLM
        chain = RAG_PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "question": question,
        })

        # 5. Save exchange to memory
        self.memory_manager.add_exchange(session_id, question, answer)

        # 6. Extract source citations
        sources = self._extract_sources(source_docs)

        logger.info(f"Answered | session='{session_id}' sources={len(sources)}")

        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
        }

    def _extract_sources(self, source_documents: list) -> list[dict]:
        """Deduplicate and format source citations."""
        seen = set()
        sources = []
        for doc in source_documents:
            meta = doc.metadata
            key = (meta.get("source", "Unknown"), meta.get("page", 0))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": meta.get("source", "Unknown"),
                    "page": meta.get("page", 0),
                })
        return sources