"""
app/api/routes/chat.py
───────────────────────
POST /chat — ask a question, get a grounded answer.
DELETE /chat/session/{session_id} — clear conversation history.
GET  /chat/history/{session_id} — view session message history.
"""

from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatRequest, ChatResponse, SourceCitation
from app.core.logger import get_logger
from app.services.chat_memory import ConversationMemory
from app.services.rag_chain import RAGChain
from app.services.vector_store import VectorStoreManager

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

# Module-level singletons
_vector_store = VectorStoreManager()
_memory_manager = ConversationMemory()
_rag_chain = RAGChain(
    vector_store=_vector_store,
    memory_manager=_memory_manager,
)


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Ask a question about HR policies.

    - **question**: Your question (1–1000 chars).
    - **session_id**: Optional session ID for conversation continuity.
      Defaults to "default". Use unique IDs per user for multi-user setups.

    Returns the answer and source document citations.
    """
    try:
        result = _rag_chain.query(
            question=request.question,
            session_id=request.session_id,
        )
        return ChatResponse(
            answer=result["answer"],
            sources=[SourceCitation(**s) for s in result["sources"]],
            session_id=result["session_id"],
        )

    except RuntimeError as exc:
        # No documents ingested yet
        raise HTTPException(status_code=503, detail=str(exc))

    except Exception as exc:
        logger.error(f"Chat error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/session/{session_id}", status_code=200)
async def clear_session(session_id: str) -> dict:
    """
    Clear the conversation history for a specific session.

    - **session_id**: The session to clear.
    """
    _memory_manager.clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared successfully."}


@router.get("/history/{session_id}")
async def get_history(session_id: str) -> dict:
    """
    Retrieve the conversation history for a session.

    - **session_id**: The session to inspect.
    """
    messages = _memory_manager.get_history(session_id)
    formatted = [
        {
            "role": msg.__class__.__name__.replace("Message", "").lower(),
            "content": msg.content,
        }
        for msg in messages
    ]
    return {
        "session_id": session_id,
        "message_count": len(formatted),
        "history": formatted,
    }
