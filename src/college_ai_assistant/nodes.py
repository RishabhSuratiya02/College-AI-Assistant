from langchain_groq import ChatGroq

from .state import State
from .config import LLM_MODEL, TEMPERATURE
from .retrievers import build_retriever


# Retrievers

academic_retriever = build_retriever(
    "data/academics_handbook.pdf"
)

fee_retriever = build_retriever(
    "data/fee_structure.pdf"
)


# LLM

llm = ChatGroq(
    model=LLM_MODEL,
    temperature=TEMPERATURE
)


# Classifier Node

def classifier_node(state: State) -> dict:
    """Classifies the user's query."""

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
        "not related to college rules or fees.\n\n"

        f"Query: {last_message}\n\n"

        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)

    category = response.content.strip().lower()

    if category not in {"academic", "fee", "general"}:
        category = "general"

    return {
        "query_type": category
    }


# Academic RAG Node

def academic_rag_node(state: State) -> dict:
    """Retrieves relevant information from academic documents."""

    query = state["messages"][-1].content

    docs = academic_retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "retrieved_context": context
    }


# Fee RAG Node

def fee_rag_node(state: State) -> dict:
    """Retrieves relevant information from fee documents."""

    query = state["messages"][-1].content

    docs = fee_retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "retrieved_context": context
    }


# General Node

def general_node(state: State) -> dict:
    """Handles general questions without retrieval."""

    return {
        "retrieved_context": "NO_RETRIEVAL_NEEDED"
    }


# Response Node

def response_node(state: State) -> dict:
    """Generates the final answer."""

    query = state["messages"][-1].content

    programme = state.get(
        "programme",
        "Unknown"
    )

    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":

        prompt = (
            f"You are a friendly college assistant talking "
            f"to a {programme} student.\n\n"

            f"Answer this question using your own general knowledge:\n\n"
            f"{query}"
        )

    else:

        prompt = (
            f"You are a college assistant helping a "
            f"{programme} student.\n\n"

            "Use the following context from the official "
            "college documents to answer the question accurately.\n\n"

            f"If the context mentions specific figures for "
            f"different programmes, highlight the information "
            f"relevant to {programme} if possible.\n\n"

            f"Context:\n{context}\n\n"

            f"Question: {query}\n\n"

            "Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)

    return {
        "messages": [
            ("ai", response.content.strip())
        ]
    }