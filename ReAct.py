from typing import Annotated, Sequence, TypedDict 
# Annoated give additional context  without effecting the type its
# sequence - automatically handel the state updates for sequence as by adding new messages to chat history
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage #Basemessage is foundational class for all message types in langgraph
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages #reducer fn - smart ways to update states 
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int,b:int):
    """This is an addition function that adds 2 numbers together""" ##if docstring not included graph doesnt work
    return a+b

@tool
def sub(a:int,b:int):
    """This is an subtraction function that subtracts 2 numbers together""" ##if docstring not included graph doesnt work
    return a-b

@tool
def mul(a:int,b:int):
    """This is an multiplication function that multiplies 2 numbers together""" ##if docstring not included graph doesnt work
    return a*b

tools=[add,sub,mul]

model= ChatOllama(model="qwen3:4b").bind_tools(tools)

def  model_call(state:AgentState) -> AgentState:
    system_prompt =SystemMessage(content = "You are my AI Assistant, please answer my query to the best of your ability")
    response = model.invoke([system_prompt] +state["messages"]) #system message and query aka human message
    return {"messages":[response]} #updated state

def should_continue(state:AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages [-1]
    if not last_message.tool_calls: #tool_calls is the way the model says: “I want to use a tool.”
        return "end"
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools =tools)
graph.add_node("tools",tool_node)

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
graph.add_edge("tools", "our_agent")
graph.set_entry_point("our_agent")
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

while True:
    user_input = input("Enter your query (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    inputs = {"messages": [HumanMessage(content=user_input)]}
    print_stream(app.stream(inputs, stream_mode="values"))