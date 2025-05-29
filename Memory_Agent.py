import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage,  AIMessage
from langchain_ollama import ChatOllama 
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages : List[Union[HumanMessage, AIMessage]]

llm = ChatOllama(model="qwen3:4b")
def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"]) 
    state["messages"].append(AIMessage(content = response.content))
    print(f"\nAI: {response.content}")
    return state

def summarize(state: AgentState) -> AgentState:
    summary = llm.invoke([HumanMessage(content="Summarize the conversation so far")] + state["messages"])
    print(f"\nSummary: {summary.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_node("summarize", summarize)
graph.add_edge(START, "process")
graph.add_edge("process", "summarize") 
graph.add_edge("summarize", END) 
agent = graph.compile()


conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")


with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")