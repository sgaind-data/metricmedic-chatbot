import logging

import streamlit as st

from chat_store import ChatStore
from metricmedic import MetricMedicRAG
from metricmedic.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("metricmedic.app")


st.set_page_config(
    page_title="MetricMedic",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def load_chatbot():
    return MetricMedicRAG()


@st.cache_resource
def load_chat_store():
    return ChatStore()


chatbot = load_chatbot()
chat_store = load_chat_store()


def create_new_chat():
    """Create a new chat and make it the active conversation."""
    chat_id = chat_store.create_chat()

    chat_store.add_message(
        chat_id,
        "assistant",
        (
            "Hi! Describe a SQL, dashboard, KPI, ETL, or data-quality issue "
            "and I'll retrieve relevant troubleshooting context and generate "
            "a grounded response."
        ),
    )

    st.session_state.active_chat_id = chat_id
    st.rerun()


def generate_title(user_input):
    """Generate a short local title from the first user message."""
    title = user_input.strip()

    if len(title) > 42:
        title = title[:42].rstrip() + "..."

    return title or "New Chat"


# -------------------------------------------------------------------
# Session setup
# -------------------------------------------------------------------

if "active_chat_id" not in st.session_state:
    existing_chats = chat_store.get_chats()

    if existing_chats:
        st.session_state.active_chat_id = existing_chats[0][0]
    else:
        chat_id = chat_store.create_chat()

        chat_store.add_message(
            chat_id,
            "assistant",
            (
                "Hi! Describe a SQL, dashboard, KPI, ETL, or data-quality "
                "issue and I'll retrieve relevant troubleshooting context "
                "and generate a grounded response."
            ),
        )

        st.session_state.active_chat_id = chat_id


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:
    st.title("🩺 MetricMedic")

    if st.button(
        "＋ New Chat",
        use_container_width=True,
        type="primary",
    ):
        create_new_chat()

    st.divider()

    st.subheader("Previous Chats")

    chats = chat_store.get_chats()

    if not chats:
        st.caption("No previous conversations yet.")

    for chat_id, title, _created_at, _updated_at in chats:
        col1, col2 = st.columns([5, 1])

        with col1:
            if st.button(
                title,
                key=f"chat_{chat_id}",
                use_container_width=True,
            ):
                st.session_state.active_chat_id = chat_id
                st.rerun()

        with col2:
            if st.button(
                "🗑️",
                key=f"delete_{chat_id}",
                help="Delete chat",
            ):
                deleting_active_chat = chat_id == st.session_state.active_chat_id

                chat_store.delete_chat(chat_id)

                remaining_chats = chat_store.get_chats()

                if deleting_active_chat:
                    if remaining_chats:
                        st.session_state.active_chat_id = remaining_chats[0][0]
                    else:
                        new_chat_id = chat_store.create_chat()

                        chat_store.add_message(
                            new_chat_id,
                            "assistant",
                            (
                                "Hi! Describe a SQL, dashboard, KPI, ETL, "
                                "or data-quality issue and I'll retrieve "
                                "relevant troubleshooting context and "
                                "generate a grounded response."
                            ),
                        )

                        st.session_state.active_chat_id = new_chat_id

                st.rerun()


# -------------------------------------------------------------------
# Main chat interface
# -------------------------------------------------------------------

st.title("🩺 MetricMedic")

st.caption(
    "RAG-powered troubleshooting assistant for SQL, dashboards, KPIs, ETL, and data-quality issues."
)

active_chat_id = st.session_state.active_chat_id

messages = chat_store.get_messages(active_chat_id)

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------------------------------------------------------
# User input
# -------------------------------------------------------------------

user_input = st.chat_input("Describe your SQL, dashboard, KPI, ETL, or data-quality issue...")

if user_input:
    existing_messages = chat_store.get_messages(active_chat_id)

    # Save user message.
    chat_store.add_message(
        active_chat_id,
        "user",
        user_input,
    )

    # Automatically rename a new conversation using the first user prompt.
    user_message_count = sum(1 for message in existing_messages if message["role"] == "user")

    if user_message_count == 0:
        chat_store.update_chat_title(
            active_chat_id,
            generate_title(user_input),
        )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context and analyzing your issue..."):
            try:
                result = chatbot.answer(
                    user_input,
                    session_id=f"chat-{active_chat_id}",
                    tags=["streamlit"],
                )

                answer = result["answer"]
                sources = result["sources"]

            except Exception:
                answer = (
                    "I encountered an unexpected error while analyzing "
                    "your issue. Please try again."
                )

                sources = []

                logger.exception("Unexpected error while answering")

        st.markdown(answer)

        if sources:
            with st.expander("🔎 Retrieved troubleshooting context"):
                for index, source in enumerate(
                    sources,
                    start=1,
                ):
                    st.markdown(f"### {index}. {source['issue']}")

                    st.markdown(f"**Similarity score:** `{source['score']:.3f}`")

                    st.markdown(f"**Guidance:** {source['response']}")

                    if index < len(sources):
                        st.divider()

    # Save assistant response.
    chat_store.add_message(
        active_chat_id,
        "assistant",
        answer,
    )

    st.rerun()
