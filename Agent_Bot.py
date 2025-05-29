from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama 
from langgraph.graph import StateGraph, START, END

# from dotenv import load_dotenv  # Optional: loads environment variables from a .env file
# load_dotenv()

# Define the structure of the agent's state
class AgentState(TypedDict):
    messages: List[HumanMessage]  # The agent holds a list of human messages as its state

# Initialize the language model (LLM)
llm = ChatOllama(model="qwen3:4b")  # Using a local model via Ollama

# Define the processing function that handles the agent's state
def process(state: AgentState) -> AgentState:
    """
    Takes the current state (user messages), sends it to the LLM,
    prints the AI's response, and returns the unchanged state.
    """
    response = llm.invoke(state["messages"])  
    print(f" AI : {response.content}")  
    return state 


# Build the LangGraph
graph = StateGraph(AgentState)           
graph.add_node("process", process)         
graph.add_edge(START, "process")           
graph.add_edge("process", END)             
agent = graph.compile()                

# Start an interactive chat loop with the agent
user_input = input("Enter: ")
while user_input != "exit":
    # Wrap user input in a HumanMessage and send it to the agent
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")  # Prompt the user for the next input
