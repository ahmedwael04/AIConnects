import streamlit as st

def handle_error(exception):
    st.error(f"An error occurred: {str(exception)}")
    st.info("Using fallback data due to connectivity issues.")
    # Logic to trigger fallback tool can be added in agents