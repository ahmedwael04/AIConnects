import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
from agents.food_agent import food_executor
from agents.activity_agent import activity_executor
from agents.router import route_query
from utils.input_validation import validate_input
from utils.error_handling import handle_error

st.set_page_config(page_title="Egyptian Explorer", page_icon="🇪🇬", layout="wide")

st.title("🇪🇬 Egyptian Explorer")
st.subheader("Discover Food and Activities in Egypt!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Route query
            agent_type = route_query(user_input)
            executor = food_executor if agent_type == "food" else activity_executor

            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Searching for recommendations..."):
                    response = executor.invoke({"input": f"{location}|{budget}"})["output"]
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            handle_error(e)