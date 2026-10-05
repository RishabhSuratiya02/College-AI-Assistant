from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str