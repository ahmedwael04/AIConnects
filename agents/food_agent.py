import os
from typing import Literal, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from tools.serpapi_tool import food_search_tool
from tools.fallback_tool import fallback_food_tool

load_dotenv()

# Define tools for the food agent
food_tools = [food_search_tool, fallback_food_tool]

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
llm_with_tools = llm.bind_tools(food_tools, parallel_tool_calls=False)

# Define state type
class GraphState(TypedDict):
    messages: list

def tool_calling_llm(state: GraphState):
    sys_msg = SystemMessage(content=(
        "You are a food recommendation assistant for Egypt. Use the provided tools to find restaurants based on the user's location and budget. The input format is 'location|budget' (e.g., 'Cairo|100'). Always use tools to fetch data. Format the final response as a concise list of up to 3 recommendations with their names and brief details. If no results are found, respond with: 'No restaurants found for this location and budget.'"
    ))
    messages = state["messages"]
    # Ensure system message is included
    if not messages:
        messages = [sys_msg]
    elif not hasattr(messages[0], "role") or messages[0].role != "system":
        messages = [sys_msg] + messages
    # Debug: Pretty print messages sent to LLM
    print("Food Agent - Messages sent to LLM (pretty print):")
    for msg in messages:
        msg.pretty_print()
    # Debug: Log messages in dictionary format
    print("Food Agent - Messages sent to LLM (dict):", [
        {"type": type(msg).__name__, "content": msg.content, "tool_calls": getattr(msg, "tool_calls", None)}
        for msg in messages
    ])
    try:
        response = llm_with_tools.invoke(messages)
        # Debug: Pretty print LLM response
        print("Food Agent - LLM response (pretty print):")
        response.pretty_print()
        # Debug: Log response in dictionary format
        print("Food Agent - LLM response (dict):", {
            "type": type(response).__name__,
            "content": response.content,
            "tool_calls": getattr(response, "tool_calls", None)
        })
        # Append new message to the existing list
        return {"messages": messages + [response]}
    except Exception as e:
        print(f"Food Agent - LLM invocation error: {e}")
        return {"messages": messages + [ToolMessage(content=f"Error: {str(e)}", tool_call_id="error")]}

def should_continue(state: GraphState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    # Debug: Pretty print last message
    print("Food Agent - should_continue - Last message (pretty print):")
    last_message.pretty_print()
    # Debug: Log last message in dictionary format
    print("Food Agent - should_continue - Last message (dict):", {
        "type": type(last_message).__name__,
        "content": last_message.content,
        "tool_calls": getattr(last_message, "tool_calls", None)
    })
    return "tools" if getattr(last_message, "tool_calls", None) else END

# Build the graph
builder = StateGraph(GraphState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(food_tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", should_continue)
builder.add_edge("tools", "tool_calling_llm")
food_graph = builder.compile()