"""
app/services/chat_memory.py
────────────────────────────
Per-session conversation memory manager.
Uses simple message lists — compatible with all LangChain versions.
"""

from typing import Dict, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from app.core.logger import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    """
    Manages conversation history per session_id using plain message lists.
    Avoids deprecated LangChain memory classes entirely.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, List[BaseMessage]] = {}
        logger.info("ConversationMemory initialised.")

    def get_history(self, session_id: str) -> List[BaseMessage]:
        """Return message history for a session, creating it if new."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
            logger.info(f"New memory session created: {session_id}")
        return self._sessions[session_id]

    def add_exchange(
        self,
        session_id: str,
        question: str,
        answer: str,
    ) -> None:
        """Append a human/AI exchange to the session history."""
        history = self.get_history(session_id)
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))

    def clear_session(self, session_id: str) -> None:
        """Clear history for a specific session."""
        if session_id in self._sessions:
            self._sessions[session_id] = []
            logger.info(f"Memory cleared for session: {session_id}")

    def clear_all(self) -> None:
        """Clear all sessions."""
        self._sessions.clear()
        logger.warning("All conversation memories cleared.")

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def format_history_as_tuples(
        self,
        session_id: str,
    ) -> List[tuple]:
        """
        Return history as (human, ai) string tuples.
        Used by LangChain chains that expect this format.
        """
        history = self.get_history(session_id)
        pairs = []
        for i in range(0, len(history) - 1, 2):
            human = history[i].content
            ai = history[i + 1].content if i + 1 < len(history) else ""
            pairs.append((human, ai))
        return pairs