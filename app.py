import os
import streamlit as st
from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Optional: used only for handling Groq rate-limit errors
from groq import RateLimitError


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered",
)


# ============================================================
# Step 1 - Build RAG Resources
# Cached so PDFs/embeddings/models are not recreated
# on every Streamlit rerun.
# ============================================================

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_resources():

    # --------------------------------------------------------
    # Embedding Model
    # --------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------------------------------------
    # Retriever Builder
    # --------------------------------------------------------

    def build_retriever(pdf_path: str):

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        # Reduced from k=4 to k=3
        # This reduces the amount of context sent to Groq.
        return vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

    # --------------------------------------------------------
    # PDF Retrievers
    # --------------------------------------------------------

    academic_retriever = build_retriever(
        "data/academics_handbook.pdf"
    )

    fee_retriever = build_retriever(
        "data/fee_structure.pdf"
    )

    # --------------------------------------------------------
    # Groq Models
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # We use a lighter model for classification because
    # classification only requires one word:
    #
    # academic / fee / general
    #
    # The larger model is kept for the final answer.
    # --------------------------------------------------------

    classifier_llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    answer_llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0.4
    )

    return (
        academic_retriever,
        fee_retriever,
        classifier_llm,
        answer_llm
    )


# Load resources once
(
    academic_retriever,
    fee_retriever,
    classifier_llm,
    answer_llm
) = load_resources()


# ============================================================
# Step 2 - State
# ============================================================

class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


# ============================================================
# Step 3 - Classifier Node
# ============================================================

def classifier_node(state: State) -> dict:
    """
    Classifies the latest student query into:
    academic / fee / general
    """

    last_message = state["messages"][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"

        "Use 'academic' for questions about attendance, exams, grading, "
        "credits, promotion, course structure, summer training, "
        "or degree requirements.\n"

        "Use 'fee' for questions about tuition, payment, refund, "
        "late charges, scholarships, or any money-related topic.\n"

        "Use 'general' for greetings, casual talk, or anything "
        "not related to the college rules or fee.\n\n"

        f"Query: {last_message}\n\n"

        "Return only one word: academic, fee, or general."
    )

    try:

        # IMPORTANT:
        # Use lightweight model for classification.
        response = classifier_llm.invoke(prompt)

        category = response.content.strip().lower()

    except RateLimitError:

        # If classifier hits rate limit, don't crash the app.
        # We fallback to a simple keyword-based classifier.

        query_lower = last_message.lower()

        fee_keywords = [
            "fee",
            "fees",
            "tuition",
            "payment",
            "refund",
            "scholarship",
            "late charge",
            "charges",
            "money",
            "payment"
        ]

        academic_keywords = [
            "attendance",
            "exam",
            "exams",
            "grading",
            "grade",
            "credit",
            "credits",
            "promotion",
            "course",
            "semester",
            "degree",
            "training",
            "subject",
            "marks"
        ]

        if any(word in query_lower for word in fee_keywords):
            category = "fee"

        elif any(word in query_lower for word in academic_keywords):
            category = "academic"

        else:
            category = "general"

    except Exception:

        # General fallback if any unexpected error occurs.

        query_lower = last_message.lower()

        if any(
            word in query_lower
            for word in ["fee", "fees", "payment", "refund", "scholarship"]
        ):
            category = "fee"

        elif any(
            word in query_lower
            for word in [
                "exam",
                "attendance",
                "course",
                "semester",
                "credit",
                "degree",
                "marks"
            ]
        ):
            category = "academic"

        else:
            category = "general"

    # --------------------------------------------------------
    # Make sure only valid categories are returned
    # --------------------------------------------------------

    if "academic" in category:
        category = "academic"

    elif "fee" in category:
        category = "fee"

    else:
        category = "general"

    return {
        "query_type": category
    }


# ============================================================
# Step 4 - Academic RAG Node
# ============================================================

def academic_rag_node(state: State) -> dict:
    """
    Retrieves relevant chunks from the academic handbook.
    """

    query = state["messages"][-1].content

    docs = academic_retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return {
        "retrieved_context": context
    }


# ============================================================
# Step 5 - Fee RAG Node
# ============================================================

def fee_rag_node(state: State) -> dict:
    """
    Retrieves relevant chunks from the fee structure PDF.
    """

    query = state["messages"][-1].content

    docs = fee_retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return {
        "retrieved_context": context
    }


# ============================================================
# Step 6 - General Node
# ============================================================

def general_node(state: State) -> dict:
    """
    General questions don't require document retrieval.
    """

    return {
        "retrieved_context": "NO_RETRIEVAL_NEEDED"
    }


# ============================================================
# Step 7 - Response Node
# ============================================================

def response_node(state: State) -> dict:
    """
    Generates the final answer using the larger Groq model.
    """

    query = state["messages"][-1].content

    programme = state.get(
        "programme",
        "Unknown"
    )

    context = state["retrieved_context"]

    # --------------------------------------------------------
    # General Question
    # --------------------------------------------------------

    if context == "NO_RETRIEVAL_NEEDED":

        prompt = (
            f"You are a friendly college assistant talking to a "
            f"{programme} student.\n\n"

            f"Answer this question using your own general knowledge:\n\n"

            f"{query}"
        )

    # --------------------------------------------------------
    # RAG Question
    # --------------------------------------------------------

    else:

        prompt = (
            f"You are a college assistant helping a "
            f"{programme} student.\n\n"

            f"Use the following context from the official college "
            f"documents to answer the question accurately.\n\n"

            f"If the context mentions specific figures for different "
            f"programmes, highlight the one relevant to "
            f"{programme} if possible.\n\n"

            f"Do not invent information that is not present in the "
            f"provided context.\n\n"

            f"Context:\n"
            f"{context}\n\n"

            f"Question:\n"
            f"{query}\n\n"

            f"Give a clear, friendly, and precise answer."
        )

    try:

        response = answer_llm.invoke(prompt)

        answer = response.content.strip()

    except RateLimitError:

        answer = (
            "⚠️ Groq API rate limit has been reached.\n\n"
            "Please wait for a short time and try again."
        )

    except Exception as e:

        answer = (
            "⚠️ Something went wrong while generating the answer.\n\n"
            f"Error: {str(e)}"
        )

    return {
        "messages": [
            ("ai", answer)
        ]
    }


# ============================================================
# Step 8 - Router
# ============================================================

def route_query(state: State):

    if state["query_type"] == "academic":
        return "academic_rag"

    elif state["query_type"] == "fee":
        return "fee_rag"

    else:
        return "general"


# ============================================================
# Step 9 - Build LangGraph
# Cached so graph is created only once.
# ============================================================

@st.cache_resource(show_spinner=False)
def build_graph():

    graph = StateGraph(State)

    # Nodes
    graph.add_node(
        "classifier",
        classifier_node
    )

    graph.add_node(
        "academic_rag",
        academic_rag_node
    )

    graph.add_node(
        "fee_rag",
        fee_rag_node
    )

    graph.add_node(
        "general",
        general_node
    )

    graph.add_node(
        "response",
        response_node
    )

    # --------------------------------------------------------
    # START → CLASSIFIER
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "classifier"
    )

    # --------------------------------------------------------
    # CLASSIFIER → ROUTER
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "classifier",
        route_query
    )

    # --------------------------------------------------------
    # RAG / GENERAL → RESPONSE
    # --------------------------------------------------------

    graph.add_edge(
        "academic_rag",
        "response"
    )

    graph.add_edge(
        "fee_rag",
        "response"
    )

    graph.add_edge(
        "general",
        "response"
    )

    # --------------------------------------------------------
    # RESPONSE → END
    # --------------------------------------------------------

    graph.add_edge(
        "response",
        END
    )

    return graph.compile()


app = build_graph()


# ============================================================
# Step 10 - Custom CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }

    .main-header h1 {
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
    }

    .main-header p {
        color: #888;
        font-size: 0.95rem;
    }

    .stChatMessage {
        border-radius: 12px;
    }

    div[data-testid="stChatInput"] {
        border-radius: 12px;
    }

    .query-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .badge-academic {
        background-color: #1f3a5f;
        color: #93c5fd;
    }

    .badge-fee {
        background-color: #4a3110;
        color: #fcd34d;
    }

    .badge-general {
        background-color: #1f4a2e;
        color: #86efac;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Step 11 - Header
# ============================================================

st.markdown(
    """
    <div class="main-header">

        <h1>🎓 College Assistant</h1>

        <p>
            Ask me about academics, fees,
            or anything else campus-related
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Step 12 - Sidebar
# ============================================================

with st.sidebar:

    st.header("⚙️ Setup")

    programme_map = {
        "BCA": "BCA",
        "BBA": "BBA",
        "B.Com (H)": "B.Com (H)",
    }

    student_programme = st.selectbox(
        "Select your programme",
        options=list(programme_map.keys()),
        index=0
    )

    st.markdown("---")

    st.caption(
        f"📌 Currently set as: "
        f"**{student_programme}** student"
    )

    # --------------------------------------------------------
    # Clear Chat
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.lc_messages = []

        st.rerun()

    st.markdown("---")

    st.caption("Routes queries to:")

    st.caption("📘 Academic Handbook (RAG)")

    st.caption("💰 Fee Structure (RAG)")

    st.caption("💬 General Knowledge")


# ============================================================
# Step 13 - Session State
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "lc_messages" not in st.session_state:

    st.session_state.lc_messages = []


# ============================================================
# Step 14 - Render Chat History
# ============================================================

for msg in st.session_state.messages:

    avatar = (
        "🧑‍🎓"
        if msg["role"] == "user"
        else "🎓"
    )

    with st.chat_message(
        msg["role"],
        avatar=avatar
    ):

        if (
            msg["role"] == "assistant"
            and msg.get("query_type")
        ):

            badge_class = (
                f"badge-{msg['query_type']}"
            )

            st.markdown(
                f"""
                <span class="query-badge {badge_class}">
                    {msg["query_type"].upper()}
                </span>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            msg["content"]
        )


# ============================================================
# Step 15 - Chat Input
# ============================================================

user_query = st.chat_input(
    "Type your question here..."
)


if user_query:

    # --------------------------------------------------------
    # Display User Message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message(
        "user",
        avatar="🧑‍🎓"
    ):

        st.markdown(user_query)

    # --------------------------------------------------------
    # Add Human Message
    # --------------------------------------------------------

    st.session_state.lc_messages.append(
        ("human", user_query)
    )

    # --------------------------------------------------------
    # LIMIT CHAT HISTORY
    #
    # Only send recent messages to the graph.
    # This helps reduce token consumption.
    # --------------------------------------------------------

    recent_messages = (
        st.session_state.lc_messages[-6:]
    )

    # --------------------------------------------------------
    # Invoke LangGraph
    # --------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🎓"
    ):

        with st.spinner(
            "Thinking..."
        ):

            try:

                result = app.invoke(
                    {
                        "programme": student_programme,

                        "messages": recent_messages
                    }
                )

                # ------------------------------------------------
                # Get AI Response
                # ------------------------------------------------

                ai_response = (
                    result["messages"][-1].content
                )

                query_type = (
                    result.get(
                        "query_type",
                        "general"
                    )
                )

                # ------------------------------------------------
                # Badge
                # ------------------------------------------------

                badge_class = (
                    f"badge-{query_type}"
                )

                st.markdown(
                    f"""
                    <span class="query-badge {badge_class}">
                        {query_type.upper()}
                    </span>
                    """,
                    unsafe_allow_html=True
                )

                # ------------------------------------------------
                # Answer
                # ------------------------------------------------

                st.markdown(
                    ai_response
                )

                # ------------------------------------------------
                # Update LangGraph History
                # ------------------------------------------------

                st.session_state.lc_messages = (
                    result["messages"]
                )

                # ------------------------------------------------
                # Save Display History
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",

                        "content": ai_response,

                        "query_type": query_type
                    }
                )

            except RateLimitError:

                st.error(
                    "⚠️ Groq API rate limit reached. "
                    "Please wait and try again."
                )

            except Exception as e:

                st.error(
                    f"⚠️ Something went wrong:\n\n{str(e)}"
                )
                
