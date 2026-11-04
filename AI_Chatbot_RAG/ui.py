
import streamlit as st
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"


def setup_page():
    st.set_page_config(
        page_title="HCMIU Academic Advising Chatbot",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    /* Main application */
    .stApp {
        background-color: #F5F9FD;
        color: #183B56;
    }

    .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #EAF3FB;
        border-right: 1px solid #D8E7F3;
    }

    /* Titles */
    h1, h2, h3 {
        color: #12568B !important;
    }

    /* Header */
    .iu-header {
        background: white;
        border: 1px solid #DFEAF4;
        border-radius: 16px;
        padding: 20px 25px;
        margin-bottom: 18px;
    }

    .iu-title {
        color: #12568B;
        font-size: 26px;
        font-weight: 750;
        margin: 0;
    }

    .iu-subtitle {
        color: #647D92;
        font-size: 14px;
        margin-bottom: 5px;
    }

    /* Welcome banner */
    .welcome {
        background: linear-gradient(
            110deg, #12568B, #2688CB
        );
        padding: 28px;
        border-radius: 18px;
        margin-bottom: 20px;
    }

    .welcome h2,
    .welcome p {
        color: white !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #FFFFFF;
        border: 1px solid #E1EAF2;
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 12px;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        border-radius: 14px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid #C9DEEE;
        background: #FFFFFF;
        color: #12568B;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #1264A3;
        background: #EAF4FD;
        color: #1264A3;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 10px;
    }

    /* Information boxes */
    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Source expanders */
    [data-testid="stExpander"] {
        background: #FFFFFF;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    logo = ASSETS / "iu_logo.png"

    col1, col2 = st.columns(
        [1, 7],
        gap="medium",
        vertical_alignment="center"
    )

    with col1:
        if logo.exists():
            st.image(str(logo), width=90)
        else:
            st.markdown("## 🎓")

    with col2:
        st.markdown("""
        <div class="iu-header">
            <p class="iu-title">
                INTERNATIONAL UNIVERSITY
            </p>
            <p class="iu-subtitle">
                Vietnam National University - HCMC
            </p>
            <strong>
                AI Academic Advising & Admissions Assistant
            </strong>
        </div>
        """, unsafe_allow_html=True)


def render_welcome():
    st.markdown("""
    <div class="welcome">
        <h2>Welcome to HCMIU AI Assistant</h2>
        <p>
            Your intelligent companion for academic advising,
            admissions, tuition fees, graduation requirements,
            and university information.
        </p>
    </div>
    """, unsafe_allow_html=True)

    campus = ASSETS / "iu_campus.jpg"

    if campus.exists():
        st.image(
            str(campus),
            use_container_width=True
        )

    st.markdown("### How can I help you today?")


def render_sidebar(urls):
    with st.sidebar:
        logo = ASSETS / "iu_logo.png"

        if logo.exists():
            st.image(str(logo), width=95)

        st.title("Student Services")
        st.caption("HCMIU Academic Assistant")

        st.divider()

        st.subheader("Knowledge Base")

        load_websites = st.button(
            "🌐 Load HCMIU Websites",
            use_container_width=True
        )

        st.divider()

        st.subheader("Upload Document")

        uploaded_file = st.file_uploader(
            "Upload an academic PDF",
            type=["pdf"]
        )

        process_pdf = st.button(
            "📄 Process PDF",
            use_container_width=True,
            disabled=uploaded_file is None
        )

        st.divider()

        st.subheader("Official Resources")

        for url in urls:
            st.markdown(f"- [{url}]({url})")

        st.divider()

        clear_chat = st.button(
            "🗑️ Clear Chat History",
            use_container_width=True
        )

        st.caption(
            "Powered by RAG, Gemini and FAISS"
        )

    return {
        "load_websites": load_websites,
        "uploaded_file": uploaded_file,
        "process_pdf": process_pdf,
        "clear_chat": clear_chat
    }


def render_chat_history(messages):
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            sources = message.get("sources", [])

            if sources:
                with st.expander("📚 View Sources"):
                    for source in sources:
                        st.write(source)


def render_suggestions():
    st.markdown("#### Suggested Questions")

    questions = [
        "What are the admission requirements?",
        "What are the graduation requirements?",
        "How can I check my academic results?",
        "Where can I find tuition information?"
    ]

    columns = st.columns(2)

    selected_question = None

    for index, question in enumerate(questions):
        with columns[index % 2]:
            if st.button(
                question,
                key=f"suggestion_{index}",
                use_container_width=True
            ):
                selected_question = question

    return selected_question
