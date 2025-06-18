from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama 
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOllama(model="qwen3:4b")

def process(state: AgentState) -> AgentState:
    """
    This function runs inside the LangGraph.
    It takes the current state (chat history), sends it to the LLM,
    gets the response, and adds it back to the state.
    """
    response = llm.invoke(state['messages'])
    ai_message = AIMessage(content=response.content)
    state['messages'].append(ai_message)
    print(f"AI: {response.content}")
    return state

graph = StateGraph(AgentState)        
graph.add_node("process", process)    
graph.add_edge(START, "process")      
graph.add_edge("process", END)        
agent = graph.compile()


conversation_history = []

print("Chat with AI. Type 'exit' to quit.\n")
user_input = input("You: ") 

while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result['messages']
    user_input = input("You: ")

print("Chat ended.")

with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")