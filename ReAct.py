from typing import TypedDict, Sequence, Annotated
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages : Annotated[Sequence [BaseMessage], add_messages]


@tool
def add(a:int , b:int):
    """This is an addition function that adds 2 numbers together"""
    return a +b

@tool
def add(a: int, b:int):
    """This is an addition function that adds 2 numbers together"""

    return a + b 

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b


tools =[add, subtract, multiply]

model= ChatOllama(model="qwen3:4b").bind_tools(tools)


def model_call(state:AgentState)->AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] +state["messages"]) #system message and query aka human message
    return {"messages":[response]} #updated state


def should_continue(state:AgentState)-> AgentState:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: #tool_calls is the way the model says: “I want to use a tool.”
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))

# Start at "our_agent"
#     ↓
# Call model → model_call()
#     ↓
# Response (might contain tool_calls)
#     ↓
# Run should_continue():
#      └─ if tool_calls → return "continue" → go to "tools" node
#      └─ else          → return "end"      → stop (END)
