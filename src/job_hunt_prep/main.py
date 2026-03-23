"""
frontend.py

Here we implement a simple page using Streamlit

J. A. Moreno
2026
"""

from streamlit.runtime.state import session_state
from graph import Graph
import streamlit as st
from dotenv import load_dotenv
from typing import Optional, Set
import hashlib
from io import BytesIO
import requests
from PIL import Image


# Load environmental variables
load_dotenv()


st.set_page_config(
    page_title="Job Hunt Prep AI Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i + 1}. {source}\n"
    return sources_string


# Can fetch a gravatar
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_profile_picture(email: str) -> Optional[Image.Image]:
    """Fetch a small avatar with strict timeouts so UI never blocks."""
    email_norm = (email or "").strip().lower().encode("utf-8")
    email_md5 = hashlib.md5(email_norm).hexdigest()
    gravatar_url = f"https://www.gravatar.com/avatar/{email_md5}?d=identicon&s=200"
    try:
        response = requests.get(
            gravatar_url,
            timeout=(2.0, 4.0),  # (connect, read)
            headers={"User-Agent": "documentation-helper/1.0"},
        )
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception:
        return None


# Custom CSS for dark theme and modern look
st.markdown(
    """
<style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stTextInput > div > div > input {
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: #FFFFFF;
    }
    .stSidebar {
        background-color: #252526;
    }
    .stMessage {
        background-color: #2D2D2D;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Set page config at the very beginning

# Sidebar user information
with st.sidebar:
    st.title("User Profile")

    # You can replace these with actual user data
    user_name = "John Doe"
    user_email = "john.doe@example.com"

    profile_pic = get_profile_picture(user_email)
    if profile_pic is not None:
        st.image(profile_pic, width=150)
    st.write(f"**Name:** {user_name}")
    st.write(f"**Email:** {user_email}")

st.header("Job Hunt Preparation Helper Bot")

# Initialize session state
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

# Use a form to prevent reruns until submission
with st.form(key="chat_form"):
    prompt = st.text_input("Prompt", placeholder="Enter your message here...")
    submit_clicked = st.form_submit_button("Submit")

if submit_clicked and prompt:

    graph = Graph()
    st.session_state["chat_history"].append(("human", prompt))
    st.chat_message("user").write(prompt)
    st.chat_message("ai").write_stream(graph.run_agent_streamlit(prompt))


# Add a footer
st.markdown("---")
st.markdown("Powered by LangChain and Streamlit")
