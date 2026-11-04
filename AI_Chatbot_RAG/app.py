import os
import tempfile
import subprocess
import sys
import json
from pathlib import Path
import streamlit as st


from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain



# =========================================================
# 1. STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HCMIU Academic Advising Chatbot",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 HCMIU Academic Advising Chatbot")
st.caption(
    "Powered by Ollama, LangChain, "
    "FAISS and Streamlit"
)



# =========================================================
# 3. OFFICIAL HCMIU WEBSITES
# =========================================================

HCMIU_URLS = [
    "https://hcmiu.edu.vn/gioi-thieu/",
    "https://tuyensinh.hcmiu.edu.vn/",
    "https://tuyensinh.hcmiu.edu.vn/tin-tuc/",
]


# =========================================================
# 4. SESSION STATE
# =========================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================================================
# 5. EMBEDDING MODEL
# =========================================================

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
    )


# =========================================================
# 6. TEXT SPLITTER
# =========================================================

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(documents)


# =========================================================
# 7. WEB DATA COLLECTION
# =========================================================

def load_webpage(url):
    """Run browser scraping in a separate process to avoid Windows event-loop issues."""
    url = url.strip()
    if not url.startswith(("https://", "http://")):
        raise ValueError("URL must start with https:// or http://")
    scraper_path = Path(__file__).resolve().parent / "web_scraper.py"
    if not scraper_path.is_file():
        raise FileNotFoundError("Missing web_scraper.py in the project folder")
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(scraper_path), url],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=75
    )
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(result.stderr.strip() or result.stdout[:400] or "Scraper returned no JSON")
    if result.returncode != 0 or data.get("error"):
        raise RuntimeError(data.get("error") or result.stderr.strip() or "Scraper failed")
    text = data.get("text", "").strip()
    if not text:
        raise ValueError("Website returned no readable text")
    return Document(page_content=text, metadata={"source": data.get("url", url)})


def load_hcmiu_websites():
    documents = []

    for url in HCMIU_URLS:
        try:
            document = load_webpage(url)
            if document is not None:
                documents.append(document)
        except Exception as error:
            st.warning(f"Unable to load {url}: {error}")

    return documents


# =========================================================
# 8. PDF DOCUMENT PROCESSING
# =========================================================

def load_pdf(uploaded_file):
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temporary_file:

            temporary_file.write(
                uploaded_file.getvalue()
            )

            temporary_path = temporary_file.name

        loader = PyPDFLoader(temporary_path)

        documents = loader.load()

        for document in documents:
            document.metadata["source"] = (
                uploaded_file.name
            )

        return documents

    finally:
        if (
            temporary_path
            and os.path.exists(temporary_path)
        ):
            os.remove(temporary_path)


# =========================================================
# 9. FAISS VECTOR DATABASE
# =========================================================

def create_vector_store(documents):
    if not documents:
        return None

    chunks = split_documents(documents)

    if not chunks:
        return None

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


# =========================================================
# 10. OLLAMA LANGUAGE MODEL
# =========================================================

@st.cache_resource
def get_llm():
    return ChatOllama(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0,
        reasoning=False,
        num_ctx=4096,
        num_predict=512,
        keep_alive="30m"
    )


# =========================================================
# 11. RAG PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are an academic advising and admissions
support assistant for International University
(Vietnam National University - Ho Chi Minh City).

Your task is to answer questions using only
the retrieved context provided below.

Instructions:
1. Provide accurate and clear answers.
2. Use the retrieved documents as your source.
3. Do not invent university regulations or policies.
4. If the information is unavailable, clearly say so.
5. Respond in the same language as the user's question.
6. Keep answers short within 512 tokens and helpful and easy to understand.

Retrieved context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}")
])


# =========================================================
# 12. CREATE RAG CHAIN
# =========================================================

def create_rag_chain(vector_store):
    llm = get_llm()

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 6
        }
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain


# =========================================================
# KNOWLEDGE BASE STATE & INITIAL WEBSITE LOAD
# =========================================================
if "websites_initialized" not in st.session_state:
    st.session_state.websites_initialized = False
if "loaded_sources" not in st.session_state:
    st.session_state.loaded_sources = set()

def add_to_knowledge_base(documents):
    if not documents:
        return False
    new_store = create_vector_store(documents)
    if new_store is None:
        return False
    if st.session_state.vector_store is None:
        st.session_state.vector_store = new_store
    else:
        st.session_state.vector_store.merge_from(new_store)
    for doc in documents:
        st.session_state.loaded_sources.add(doc.metadata.get("source", "Unknown"))
    return True

if not st.session_state.websites_initialized:
    with st.spinner("Loading official HCMIU websites..."):
        add_to_knowledge_base(load_hcmiu_websites())
    st.session_state.websites_initialized = True

with st.sidebar:
    st.header("📚 Knowledge Base")
    st.caption(f"{len(st.session_state.loaded_sources)} sources loaded")
    if st.session_state.loaded_sources:
        with st.expander("View sources"):
            for source in sorted(st.session_state.loaded_sources):
                st.write(source)
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    st.caption("Powered by Ollama, LangChain, FAISS and Streamlit")

# =========================================================
# 14. CHAT INTERFACE
# =========================================================

st.subheader("💬 Academic Assistant")

st.info(
    "Ask questions about HCMIU admissions, "
    "academic regulations, tuition fees, "
    "courses and student services."
)

for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        if message.get("sources"):
            with st.expander(
                "📚 View Sources"
            ):
                for source in message["sources"]:
                    st.write(source)


# =========================================================
# 15. USER QUESTION AND RAG RESPONSE
# =========================================================

# Streamlit's built-in attachment button is inside the chat input.
# Paste a URL as a message to add a website to the knowledge base.
chat_submission = st.chat_input(
    "Ask a question, paste a URL, or attach a PDF...",
    accept_file=True,
    file_type=["pdf"],
    key="academic_chat_input",
)

user_question = None
if chat_submission is not None:
    user_question = (chat_submission.text or "").strip()

    for uploaded_file in chat_submission.files:
        try:
            with st.spinner(f"Reading {uploaded_file.name}..."):
                if uploaded_file.name in st.session_state.loaded_sources:
                    st.info(f"Already loaded: {uploaded_file.name}")
                else:
                    pdf_documents = load_pdf(uploaded_file)
                    if add_to_knowledge_base(pdf_documents):
                        st.success(f"Loaded PDF: {uploaded_file.name}")
                    else:
                        st.warning(f"No readable text in {uploaded_file.name}")
        except Exception as error:
            st.error(f"PDF loading failed: {error}")

    if user_question.startswith(("https://", "http://")):
        url = user_question.split()[0]
        try:
            if url in st.session_state.loaded_sources:
                st.info(f"Already loaded: {url}")
            else:
                with st.spinner(f"Reading {url}..."):
                    document = load_webpage(url)
                    if add_to_knowledge_base([document]):
                        st.success(f"Loaded website: {url}")
                    else:
                        st.warning(f"No readable content: {url}")
        except Exception as error:
            st.error(f"Website loading failed: {error}")
        user_question = None

if user_question:

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):

        if st.session_state.vector_store is None:
            answer = (
                "No knowledge base is available. "
                "Attach a PDF using the chat input, or paste a website URL."
            )
            sources = []

        else:
            with st.spinner(
                "Searching academic information..."
            ):
                try:
                    rag_chain = create_rag_chain(
                        st.session_state.vector_store
                    )

                    result = rag_chain.invoke({
                        "input": user_question
                    })

                    answer = result.get(
                        "answer",
                        "No answer was generated."
                    )

                    source_documents = result.get(
                        "context",
                        []
                    )

                    sources = []

                    for document in source_documents:
                        source = document.metadata.get(
                            "source",
                            "Unknown source"
                        )

                        page = document.metadata.get(
                            "page"
                        )

                        if page is not None:
                            source += (
                                f" (Page {page + 1})"
                            )

                        if source not in sources:
                            sources.append(source)

                except Exception as error:
                    answer = f"An error occurred: {error}"
                    sources = []

        st.markdown(answer)

        if sources:
            with st.expander(
                "📚 View Retrieved Sources"
            ):
                for source in sources:
                    st.write(source)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
