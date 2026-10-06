import os
import time
import uuid

import logfire
import requests
import streamlit as st
import tiktoken

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

env_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".env")
)

load_dotenv(dotenv_path=env_path)


# ============================================================
# CACHED RESOURCES
# ============================================================

@st.cache_resource
def initialize_logfire():
    """
    Initialize Logfire once and reuse it across Streamlit reruns.
    """
    try:
        token = os.getenv("LOGFIRE_TOKEN")

        if not token:
            return "LOGFIRE_TOKEN not configured"

        logfire.configure(
            token=token,
            service_name="rag-ui",
        )
        logfire.instrument_requests()

        return "Connected & Tracing"

    except Exception as e:
        return f"Standby: {e}"


@st.cache_resource
def get_backend_url():
    """
    Cache the backend URL.
    """
    return os.getenv(
        "BACKEND_URL",
        "http://localhost:8000"
    )


LOGFIRE_STATUS = initialize_logfire()
BACKEND_URL = get_backend_url()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Enterprise Research Agent",
    page_icon="🔬",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

AI_AVATAR = "🤖"
USER_AVATAR = "👤"
MAX_QUERY_TOKENS = 2500
TOKEN_ENCODING = tiktoken.get_encoding("o200k_harmony")


def count_tokens(text: str) -> int:
    """Count query tokens using the GPT-OSS tokenizer."""
    return len(TOKEN_ENCODING.encode(text, disallowed_special=()))


def render_sources(sources):
    """Render retrieved context below an assistant response."""
    if not sources:
        return

    with logfire.span("Render Retrieved Sources", source_count=len(sources)):
        with st.expander(f"Retrieved research documents ({len(sources)})"):
            for i, source in enumerate(sources):
                if isinstance(source, dict):
                    source_name = str(source.get("source") or "Unknown document")
                    source_text = str(source.get("content") or source.get("text") or "")
                    score = source.get("score")
                else:
                    source_name = "Retrieved chunk"
                    source_text = str(source)
                    score = None

                with logfire.span(
                    "Render Retrieved Source",
                    source_index=i + 1,
                    has_score=isinstance(score, (int, float)),
                    content_characters=len(source_text),
                ):
                    with st.expander(f"{i + 1}. {source_name}"):
                        if isinstance(score, (int, float)):
                            st.caption(f"Vector similarity: {score:.3f}")
                        st.write(source_text)


# ============================================================
# SESSION MANAGEMENT
# ============================================================

if "session_id" not in st.session_state:

    st.session_state.session_id = str(uuid.uuid4())

    with logfire.span("Create Research Session"):
        logfire.info(
            "New research session created",
            conversation_id=st.session_state.session_id
        )


if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🔬 Research Agent")

    st.markdown("---")

    st.success(
        f"Logfire: {LOGFIRE_STATUS}"
    )

    st.info(
        f"Memory ID: {st.session_state.session_id[:8]}"
    )

    st.markdown("---")

    if st.button(
        "🗑️ Clear History & Memory",
        width="stretch",
        type="primary"
    ):

        with logfire.span(
            "Clear Research History",
            conversation_id=st.session_state.session_id,
            cleared_message_count=len(st.session_state.messages),
        ):
            st.session_state.messages = []

            st.session_state.session_id = str(
                uuid.uuid4()
            )

            logfire.info("Research session reset")

        st.rerun()


# ============================================================
# MAIN APPLICATION
# ============================================================

st.title("🔬 Enterprise Research Agent")

st.caption(
    "Ask technical questions and research topics using "
    "your enterprise knowledge base."
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

with logfire.span(
    "Render Chat History",
    message_count=len(st.session_state.messages),
    conversation_id=st.session_state.session_id,
):
    for index, message in enumerate(st.session_state.messages):
        role = message["role"]
        avatar = AI_AVATAR if role == "assistant" else USER_AVATAR

        with logfire.span(
            "Render Chat Message",
            message_index=index,
            role=role,
            content_characters=len(message["content"]),
        ):
            with st.chat_message(role, avatar=avatar):
                st.markdown(message["content"])
                if role == "assistant":
                    render_sources(message.get("sources", []))


# ============================================================
# CHAT INPUT
# ============================================================

if prompt := st.chat_input(
    "Ask a research question on kubernetes..."
):

    query_token_count = count_tokens(prompt)
    with logfire.span(
        "Validate User Query",
        query_tokens=query_token_count,
        query_token_limit=MAX_QUERY_TOKENS,
        accepted=query_token_count <= MAX_QUERY_TOKENS,
    ):
        if query_token_count > MAX_QUERY_TOKENS:
            st.warning(
                f"Your query is {query_token_count:,} tokens. "
                f"Please shorten it to {MAX_QUERY_TOKENS:,} tokens or fewer."
            )
            st.stop()

    with logfire.span(
        "User Research Interaction",
        query_tokens=query_token_count,
        conversation_id=st.session_state.session_id
    ):

        # ----------------------------------------------------
        # Store user message
        # ----------------------------------------------------

        with logfire.span("Store User Message", content_characters=len(prompt)):
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )

        with logfire.span("Render Current User Message"):
            with st.chat_message(
                "user",
                avatar=USER_AVATAR
            ):
                st.markdown(prompt)


        # ----------------------------------------------------
        # Assistant response
        # ----------------------------------------------------

        with st.chat_message(
            "assistant",
            avatar=AI_AVATAR
        ):

            with st.status(
                "🔍 Research Agent is working...",
                expanded=True
            ) as status:

                try:

                    # ============================================
                    # BACKEND REQUEST
                    # ============================================

                    with logfire.span(
                        "Calling Research Agent Backend",
                        backend_url=BACKEND_URL,
                        timeout_seconds=60,
                    ) as backend_span:

                        url = f"{BACKEND_URL}/query"

                        payload = {
                            "q": prompt,
                            "thread_id": (
                                st.session_state.session_id
                            )
                        }

                        response = requests.post(
                            url,
                            json=payload,
                            timeout=60
                        )

                        response.raise_for_status()

                        data = response.json()
                        backend_span.set_attribute(
                            "http.status_code",
                            response.status_code,
                        )
                        backend_span.set_attribute(
                            "response.content_length",
                            len(response.content),
                        )


                    # ============================================
                    # AGENT PLAN
                    # ============================================

                    with logfire.span("Parse Backend Response"):
                        steps = data.get(
                            "thought_process",
                            []
                        )
                        sources = data.get(
                            "sources",
                            []
                        )
                        logfire.info(
                            "Backend response parsed",
                            plan_step_count=len(steps),
                            source_count=len(sources),
                            answer_characters=len(data.get("answer", "")),
                            backend_status=data.get("status", "unknown"),
                        )

                    if steps:

                        with logfire.span(
                            "Render Agent Plan",
                            step_count=len(steps),
                        ):
                            st.markdown("**Agent Process**")
                            for step_index, step in enumerate(steps):
                                with logfire.span(
                                    "Render Agent Plan Step",
                                    step_index=step_index,
                                ):
                                    st.write(f"⚙️ {step}")


                    # ============================================
                    # STATUS
                    # ============================================

                    with logfire.span("Update Research Status", state="complete"):
                        status.update(
                            label="✅ Research completed",
                            state="complete",
                            expanded=False
                        )


                # ================================================
                # CONNECTION ERROR
                # ================================================

                except requests.exceptions.ConnectionError:

                    logfire.error(
                        "Research backend unavailable"
                    )

                    status.update(
                        label="❌ Backend unavailable",
                        state="error"
                    )

                    st.error(
                        "Research backend is offline."
                    )

                    st.stop()


                # ================================================
                # TIMEOUT ERROR
                # ================================================

                except requests.exceptions.Timeout:

                    logfire.error(
                        "Research backend request timed out"
                    )

                    status.update(
                        label="❌ Request timed out",
                        state="error"
                    )

                    st.error(
                        "The research request timed out."
                    )

                    st.stop()


                # ================================================
                # GENERAL ERROR
                # ================================================

                except Exception as e:

                    logfire.exception(
                        "Research agent request failed",
                        error=str(e)
                    )

                    status.update(
                        label="❌ Research failed",
                        state="error"
                    )

                    st.error(
                        "An error occurred while processing "
                        "your research request."
                    )

                    st.stop()


            # ====================================================
            # FINAL ANSWER
            # ====================================================

            with logfire.span("Prepare Final Answer"):
                answer_placeholder = st.empty()
                full_answer = data.get(
                    "answer",
                    "No response generated."
                )
                answer_character_count = len(full_answer)

            with logfire.span(
                "Stream Answer to UI",
                answer_characters=answer_character_count,
            ):
                current_text = ""
                for char in full_answer:
                    current_text += char
                    answer_placeholder.markdown(
                        current_text + "▌"
                    )
                    time.sleep(0.005)

                answer_placeholder.markdown(full_answer)

            if sources:
                render_sources(sources)
            elif not any("Skipped" in str(step) for step in steps):
                with logfire.span("Render Empty Retrieval Notice"):
                    if any("Retrieval Failed" in str(step) for step in steps):
                        st.warning("Qdrant retrieval failed; this answer has no retrieved source documents.")
                    else:
                        st.info("No Qdrant source documents were returned for this answer.")


            # ====================================================
            # SAVE ASSISTANT RESPONSE
            # ====================================================

            with logfire.span(
                "Store Assistant Response",
                answer_characters=answer_character_count,
                source_count=len(sources),
            ):
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_answer,
                        "sources": sources,
                    }
                )

            logfire.info(
                "Research cycle completed",
                conversation_id=st.session_state.session_id,
                answer_characters=answer_character_count,
                source_count=len(sources),
            )
