import streamlit as st


def init_session_state():
    """
    Initialize all session state variables.
    Call this once at the top of app.py.
    """
    defaults = {
        "df": None,
        "profile": None,
        "filename": None,
        "anomaly_result": None,
        "auto_summary": None,
        "chat_history": [],
        "pending_question": None,
        "report_text": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_on_new_file():
    """
    Reset all analysis state when a new file is uploaded.
    """
    st.session_state.anomaly_result = None
    st.session_state.chat_history = []
    st.session_state.auto_summary = None
    st.session_state.report_text = None


def get_chat_history():
    return st.session_state.chat_history


def append_chat(role: str, text: str):
    st.session_state.chat_history.append({
        "role": role,
        "parts": [{"text": text}]
    })


def clear_chat():
    st.session_state.chat_history = []