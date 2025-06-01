# app.py
import streamlit as st
import requests
import sqlite3
import time
from datetime import datetime, timedelta

# Initialize Streamlit page configuration
st.set_page_config(page_title="RAG + Web Search QA", layout="centered")

st.title("📚🌐 Ca' Foscari AI-Based Assistant")

# Setup database connection
def init_db():
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            conversation TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Add a new conversation to the database
def add_conversation(email, conversation):
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute('INSERT INTO conversations (email, conversation) VALUES (?, ?)', (email, conversation))
    conn.commit()
    conn.close()

# Retrieve the last conversation for a specific email
def get_last_conversation(email):
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute('SELECT * FROM conversations WHERE email = ? ORDER BY timestamp DESC LIMIT 1', (email,))
    result = c.fetchone()
    conn.close()
    return result

# Initialize session state for conversation history and timestamp
if "conversation" not in st.session_state:
    st.session_state.conversation = []
    st.session_state.last_interaction_time = datetime.now()

# Input UI: User email and checkbox for web search activation
email = st.text_input("Enter your email address:", key="email_input")
use_web = st.checkbox("Web Search Activate", value=False)

# Check if email is provided before allowing interaction
if email:
    # Check if the last conversation has expired (15-minute timeout)
    if datetime.now() - st.session_state.last_interaction_time > timedelta(minutes=15):
        st.session_state.conversation = []
        st.session_state.last_interaction_time = datetime.now()

    # Input box for the user to ask a question
    query = st.chat_input("Enter your question:")
    
    # Display conversation history
    for i, message in enumerate(st.session_state.conversation):
        if i % 2 == 0:  # User message
            with st.chat_message("user"):
                st.write(message)
        else:  # Assistant message
            with st.chat_message("assistant"):
                st.write(message)

    # On new message
    if query:
        # Add user message to conversation
        st.session_state.conversation.append(query)
        
        # Update the last interaction time
        st.session_state.last_interaction_time = datetime.now()

        with st.spinner("Getting answer..."):
            try:
                # Prepare the conversation history (only previous assistant messages)
                response = requests.post(
                    "http://localhost:8000/ask",
                    json={
                        "query": query,
                        "use_web_search": use_web,
                        "conversation_history": st.session_state.conversation
                    }
                )
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer returned.")
                    # Add assistant response to conversation
                    st.session_state.conversation.append(answer)
                    
                    # Save the current conversation to the database
                    add_conversation(email, " | ".join(st.session_state.conversation))
                    
                    # Display the new messages
                    with st.chat_message("user"):
                        st.write(query)
                    with st.chat_message("assistant"):
                        st.write(answer)
                    
                    # Rerun to update the display
                    st.rerun()
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.warning("Please enter your email to begin the conversation.")

# Button to clear conversation
if st.button("Clear Conversation"):
    st.session_state.conversation = []
    st.session_state.last_interaction_time = datetime.now()
    st.rerun()

# Initialize database if not already done
init_db()
