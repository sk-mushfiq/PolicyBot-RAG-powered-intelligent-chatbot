"""
frontend/streamlit_app.py
──────────────────────────
Streamlit chat UI for HR PolicyBot.
Communicates with the FastAPI backend via HTTP.

Run with:
    streamlit run frontend/streamlit_app.py
"""

import uuid
import requests
import streamlit as st

# ─────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────

API_BASE_URL = "http://localhost:8000"
INGEST_URL   = f"{API_BASE_URL}/ingest"
CHAT_URL     = f"{API_BASE_URL}/chat"
HEALTH_URL   = f"{API_BASE_URL}/health"

st.set_page_config(
    page_title="HR PolicyBot",
    page_icon="📋",
    layout="centered",
)


# ─────────────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []


# ─────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────

def check_api_health() -> dict | None:
    """Return health data dict or None if API is unreachable."""
    try:
        response = requests.get(HEALTH_URL, timeout=3)
        return response.json() if response.status_code == 200 else None
    except requests.exceptions.ConnectionError:
        return None


def ingest_file(uploaded_file) -> dict | None:
    """Upload a file to the /ingest endpoint and return the response."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(INGEST_URL, files=files, timeout=60)
        return response.json()
    except Exception as exc:
        return {"error": str(exc)}


def send_chat_message(question: str, session_id: str) -> dict | None:
    """Send a question to the /chat endpoint and return the response."""
    try:
        payload = {"question": question, "session_id": session_id}
        response = requests.post(CHAT_URL, json=payload, timeout=30)
        return response.json()
    except Exception as exc:
        return {"error": str(exc)}


def format_sources(sources: list) -> str:
    """Format source citations into a readable string."""
    if not sources:
        return ""
    lines = ["\n\n---\n📄 **Sources:**"]
    for s in sources:
        page_info = f", Page {s['page']}" if s.get("page") else ""
        lines.append(f"- `{s['source']}`{page_info}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────
# Sidebar — Status & File Upload
# ─────────────────────────────────────────────────────

with st.sidebar:
    st.title("📋 HR PolicyBot")
    st.caption("Ask questions about company HR policies")
    st.divider()

    # API Health Status
    st.subheader("🔌 API Status")
    health = check_api_health()
    if health:
        st.success("API is running ✅")
        store_status = "✅ Documents loaded" if health.get("vector_store_ready") \
                       else "⚠️ No documents ingested yet"
        st.info(store_status)
    else:
        st.error("API offline ❌ — Start the backend server.")
        st.code("uvicorn app.main:app --reload", language="bash")

    st.divider()

    # File Upload
    st.subheader("📁 Upload HR Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload HR policy documents to teach the bot.",
    )

    if uploaded_files:
        if st.button("📤 Ingest Documents", use_container_width=True):
            for file in uploaded_files:
                if file.name not in st.session_state.ingested_files:
                    with st.spinner(f"Processing {file.name}..."):
                        result = ingest_file(file)
                    if "error" in result:
                        st.error(f"❌ {file.name}: {result['error']}")
                    elif "chunks_created" in result:
                        st.success(
                            f"✅ {file.name} — {result['chunks_created']} chunks"
                        )
                        st.session_state.ingested_files.append(file.name)
                else:
                    st.info(f"ℹ️ {file.name} already ingested.")

    # Ingested Files List
    if st.session_state.ingested_files:
        st.divider()
        st.subheader("📚 Ingested Documents")
        for fname in st.session_state.ingested_files:
            st.markdown(f"- `{fname}`")

    st.divider()

    # Session Info & Reset
    st.subheader("⚙️ Session")
    st.caption(f"Session ID: `{st.session_state.session_id}`")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ─────────────────────────────────────────────────────
# Main Chat Interface
# ─────────────────────────────────────────────────────

st.title("💬 HR Policy Assistant")
st.caption(
    "Ask anything about leave policies, code of conduct, onboarding, and more."
)

# Suggested starter questions
if not st.session_state.messages:
    st.info(
        "👋 **Welcome!** Upload HR documents in the sidebar, then ask questions like:\n\n"
        "- *How many annual leave days am I entitled to?*\n"
        "- *What is the process for requesting sick leave?*\n"
        "- *What counts as a code of conduct violation?*"
    )

# Render conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about HR policies..."):

    # Guard: check API is reachable
    if not check_api_health():
        st.error("❌ Cannot reach the API. Please start the backend server first.")
        st.stop()

    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API and display response
    with st.chat_message("assistant"):
        with st.spinner("Searching HR documents..."):
            result = send_chat_message(prompt, st.session_state.session_id)

        if result and "answer" in result:
            answer = result["answer"]
            sources_text = format_sources(result.get("sources", []))
            full_response = answer + sources_text

            st.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

        elif result and "detail" in result:
            # API returned an error
            error_msg = f"⚠️ {result['detail']}"
            st.warning(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
        else:
            fallback = "⚠️ Unable to get a response. Please try again."
            st.warning(fallback)
            st.session_state.messages.append(
                {"role": "assistant", "content": fallback}
            )
