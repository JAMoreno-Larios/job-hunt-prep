"""
main.py

This will be the main entry point for the assitant.
Here we will declare all the UI logic with
Streamlit, and take care of initiating the agent 
within the backend package.

UI is based on
https://github.com/streamlit/demo-ai-assistant/blob/main/streamlit_app.py

J. A. Moreno
2026
"""

import datetime
import time
from langgraph.graph.state import RunnableConfig
import streamlit as st
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler
)
from dotenv import load_dotenv
from htbuilder import div, styles
from htbuilder.units import em
from graph import Agent, LLM, Retriever

# Load environmental variables
load_dotenv()


st.set_page_config(
    page_title="Job Hunt Prep AI Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define a few constants
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=5)

# Initialize the agent
agent = Agent(llm=LLM().llm, retriever=Retriever().retriever)

# Disclaimer
@st.dialog("Disclaimer")
def show_disclaimer_dialog():
    st.caption("""
            This AI chatbot is powered by the configured LLMs
            (in this version, Qwen3.5:9B), and the ingested user information.
            Answers may be inaccurate, inefficient, or biased.
            Any use or decisions based on such answers should include reasonable
            practices including human oversight to ensure they are safe,
            accurate, and suitable for your intended purpose. 
            This chatbot may fail to scrap the provided job post, or 
            to retrieve relevant information from the user database.

            This software is provided as is.
        """)

# Draw the UI

st.html(div(style=styles(font_size=em(5), line_height=1))["❉"])

title_row = st.container(
    horizontal=True,
    vertical_alignment="bottom"
)

with title_row:
    st.title(
        "Job Prep AI assistant",
        anchor=False,
        width="stretch",
    )

# Define what to do when we start
user_just_asked_initial_question = (
    "initial_question" in st.session_state and st.session_state.initial_question
)

has_message_history = (
    "messages" in st.session_state and len(st.session_state.messages) > 0
)

# Show a different UI when the user has not asked anything yet
if not user_just_asked_initial_question and not has_message_history:
    st.session_state.messages = []

    with st.container():
        st.chat_input("Ask a question...", key="initial_question")

    st.button(
        "&nbsp;:small[:gray[:material/balance: Disclaimer]]",
        type="tertiary",
        on_click=show_disclaimer_dialog,
    )

    st.stop()

# Define the rest of the UI
# Show chat input at the bottom when a question has been asked.
user_message = st.chat_input("Ask a follow-up...")

if not user_message:
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question

with title_row:

    def clear_conversation():
        st.session_state.messages = []
        st.session_state.initial_question = None

    st.button(
        "Restart",
        icon=":material/refresh:",
        on_click=clear_conversation,
    )

if "prev_question_timestamp" not in st.session_state:
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)

# Display chat messages from history as speech bubbles.
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.container()  # Fix ghost message bug.

        st.markdown(message["content"])

if user_message:
    # When the user posts a message...

    # Streamlit's Markdown engine interprets "$" as LaTeX code (used to
    # display math). The line below fixes it.
    user_message = user_message.replace("$", r"\$")

    # Display message as a speech bubble.
    with st.chat_message("user"):
        st.text(user_message)

    # Display assistant response as a speech bubble.
    answer_container = st.chat_message("assistant")
    # Define the callback and config, bridging LangChain and
    # Streamlit
    st_callback = StreamlitCallbackHandler(answer_container)
    cfg = RunnableConfig()
    cfg["callbacks"] = [st_callback]
    cfg["configurable"]: {
            "thread_id" : "1"  # Change later
        }

    with st.spinner("Waiting..."):
        # Rate-limit the input if needed.
        question_timestamp = datetime.datetime.now()
        time_diff = question_timestamp - st.session_state.prev_question_timestamp
        st.session_state.prev_question_timestamp = question_timestamp

        if time_diff < MIN_TIME_BETWEEN_REQUESTS:
            time.sleep(time_diff.seconds + time_diff.microseconds * 0.001)

        user_message = user_message.replace("'", "")

    # Send prompt to LLM.
    with st.spinner("Thinking..."):
        answer = agent.run_agent(user_message, cfg)

    # Write output
    answer_container.write(answer['messages'][-1].content)
    # Add messages to chat history.
    st.session_state.messages.append({"role": "user", "content": user_message})
    st.session_state.messages.append({"role": "assistant", "content": answer['messages'][-1].content})
