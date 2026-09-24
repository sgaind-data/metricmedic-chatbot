import streamlit as st

from chatbot import MetricMedic


st.set_page_config(
    page_title="MetricMedic",
    page_icon="🩺",
    layout="centered"
)


@st.cache_resource
def load_chatbot():
    return MetricMedic()


chatbot = load_chatbot()

st.title("🩺 MetricMedic")
st.caption(
    "A lightweight troubleshooting assistant for SQL, dashboards, KPIs, "
    "and data-quality issues."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! Describe a SQL, dashboard, KPI, or data-quality issue "
                "and I'll suggest likely causes and diagnostic checks."
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input(
    "Example: My Power BI revenue is higher than my SQL result"
)

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    response = chatbot.generate_response(user_input)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.markdown(response)