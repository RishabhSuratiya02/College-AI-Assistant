from langgraph.graph import StateGraph, START, END

from .state import State

from .nodes import (
    classifier_node,
    academic_rag_node,
    fee_rag_node,
    general_node,
    response_node
)


# Router

def route_query(state: State):

    if state["query_type"] == "academic":
        return "academic_rag"

    elif state["query_type"] == "fee":
        return "fee_rag"

    else:
        return "general"


# Build Graph

graph = StateGraph(State)


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


# Edges

graph.add_edge(
    START,
    "classifier"
)


graph.add_conditional_edges(
    "classifier",
    route_query
)


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

graph.add_edge(
    "response",
    END
)


# Compile

app = graph.compile()