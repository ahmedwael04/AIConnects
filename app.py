import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
from agents.food_agent import food_graph
from agents.activity_agent import activity_graph
from agents.router import route_query
from utils.input_validation import validate_input
from utils.error_handling import handle_error
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Egyptian Explorer", page_icon="🇪🇬", layout="wide")

st.title("🇪🇬 Egyptian Explorer")
st.subheader("Discover Food and Activities in Egypt!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history (only user and assistant messages)
for message in st.session_state.messages:
    if isinstance(message, dict) and message.get("role") in ("user", "assistant"):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    else:
        print(f"Invalid message in session state: {message}")

# Input form
with st.form("user_input"):
    location = st.text_input("Enter your city or neighborhood (e.g., Cairo, Zamalek):")
    budget = st.number_input("Enter your budget (EGP):", min_value=0, step=10)
    query_type = st.selectbox("What are you looking for?", ["Food", "Activities"])
    submit = st.form_submit_button("Get Recommendations")

if submit:
    # Validate input
    if not validate_input(location, budget):
        st.error("Please provide a valid location and budget.")
    else:
        try:
            user_input = f"{query_type.lower()} in {location} with budget {budget} EGP"
            # Store user input as a dictionary
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Route query
            agent_type = route_query(user_input)
            graph = food_graph if agent_type == "food" else activity_graph

            # Invoke the LangGraph
            with st.chat_message("assistant"):
                with st.spinner("Searching for recommendations..."):
                    result = graph.invoke({"messages": [HumanMessage(content=f"{location}|{budget}")]})
                    # Debug: Log all messages returned by the graph
                    print("Graph result messages:", [
                        {"type": type(msg).__name__, "content": msg.content, "tool_calls": getattr(msg, "tool_calls", None)}
                        for msg in result["messages"]
                    ])
                    # Extract the final assistant message
                    response = None
                    for msg in reversed(result["messages"]):
                        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                            response = msg.content.strip() if msg.content else "No recommendations found."
                            break
                    if not response:
                        response = "No recommendations found. Please try a different location or budget."
                    st.markdown(response)
                    # Store assistant response as a dictionary
                    st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"Error during graph invocation: {e}")
            handle_error(e)