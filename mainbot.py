import os
import streamlit as st
import sqlite3
import hashlib
from PIL import Image
from datetime import datetime, timedelta

# 🔐 Security Helpers
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# 🗄️ Database Setup
def create_usertable():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Add email to the schema
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT UNIQUE, email TEXT, password TEXT)')
    
    # Try adding the email column if the table already existed without it (for backward compatibility)
    try:
        c.execute('ALTER TABLE userstable ADD COLUMN email TEXT')
    except sqlite3.OperationalError:
        pass # Email column already exists
        
    c.execute('CREATE TABLE IF NOT EXISTS chat_history(username TEXT, date TEXT, role TEXT, content TEXT)')
    conn.commit()
    conn.close()

def save_chat(username, role, content):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d")
    c.execute('INSERT INTO chat_history(username, date, role, content) VALUES (?,?,?,?)', (username, date_str, role, content))
    conn.commit()
    conn.close()

def get_chat_dates(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # SQLite datetime functions can calculate 7 days ago natively, but python works too
    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    c.execute('SELECT DISTINCT date FROM chat_history WHERE username=? AND date >= ? ORDER BY date DESC', (username, seven_days_ago))
    dates = [row[0] for row in c.fetchall()]
    conn.close()
    return dates

def get_chat_for_date(username, date_str):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT role, content FROM chat_history WHERE username=? AND date=? ORDER BY rowid ASC', (username, date_str))
    data = [{'role': row[0], 'content': row[1]} for row in c.fetchall()]
    conn.close()
    return data

def add_userdata(username, email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO userstable(username, email, password) VALUES (?,?,?)', (username, email, password))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    finally:
        conn.close()
    return success

def login_user(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT username FROM userstable WHERE email = ? AND password = ?', (email, password))
    data = c.fetchall()
    conn.close()
    return data

# 👁️ Import your image retrieval pipeline
from main_rag_pipeline import get_best_image_from_query

# 🧠 Imports for Groq API (Cloud LLM)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Global constants
DB_FAISS_PATH = "vectorstore/db_faiss"

# 🗄️ Load vectorstore with caching
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# 🪄 Custom prompt
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# 🚀 Load Groq LLM (Cloud API)
def load_llm():
    # Looks for GROQ_API_KEY in Streamlit Secrets or local environment variables
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    
    if not api_key:
        st.error("Error: GROQ_API_KEY is missing! Add it underneath 'Settings > Secrets' on Streamlit Cloud.")
        st.stop()
        
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant", 
        api_key=api_key,
        temperature=0.5
    )
    return llm

# 🧠 Ollama RAG Chatbot Page
def rag_chatbot_page():
    st.title("🧠 Ask Medical Chatbot (Local RAG with Ollama)")

    if 'view_date' not in st.session_state:
        st.session_state['view_date'] = datetime.now().strftime("%Y-%m-%d")

    # Load from DB to session state if it isn't initialized or if we switched dates
    if 'messages' not in st.session_state or 'needs_reload' in st.session_state:
        st.session_state.messages = get_chat_for_date(st.session_state["username"], st.session_state['view_date'])
        if 'needs_reload' in st.session_state:
            del st.session_state['needs_reload']

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Only allow chatting today to keep history clean
    today_str = datetime.now().strftime("%Y-%m-%d")
    if st.session_state['view_date'] != today_str:
        st.info("You're viewing a past chat date. Switch to 'Today' in the sidebar to send new messages.")
        return

    prompt = st.chat_input("Ask your medical question...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        save_chat(st.session_state["username"], "user", prompt)

        CUSTOM_PROMPT_TEMPLATE = """
You are a helpful and informative medical assistant chatbot.

Use the following context to answer the user's question to the best of your ability. If the context contains relevant information, summarize it even if it's not a complete medical answer. 
You can also use your general knowledge to supplement the answer if it helps explain the context, but prioritize the provided context.

Context: {context}
Question: {question}

Answer:
"""

        try:
            # Clean prompt to remove stray quotes
            clean_prompt = prompt.replace("'", "").replace('"', '').strip()
            
            # Note: We removed the LLM-based spell checker because running a local LLM twice 
            # (once for spelling, once for the answer) causes major delays. 
            # The embedding model (MiniLM) naturally handles minor typos via subword semantics.

            vectorstore = get_vectorstore()
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Use a spinner so the user knows it's thinking
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({'query': clean_prompt})
                
            result_to_show = response["result"]

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
            save_chat(st.session_state["username"], "assistant", result_to_show)

        except Exception as e:
            st.error(f"Error: {str(e)}")

# 👁️ Eye Image Generator Page
def eye_image_generator_page():
    st.title("👁️ Semantic Eye Image Generator")

    user_input = st.text_input("Ask about any eye part or structure to get a matched image:")

    if user_input:
        with st.spinner("Searching for the best matching image..."):
            image_path = get_best_image_from_query(user_input)
        
        if image_path:
            st.image(image_path, caption="Matched Image", use_container_width=True)
        else:
            st.error("No image matched your query.")

# 🚦 Main App Selector
def main():
    st.set_page_config(page_title="Medical RAG + Image Bot", layout="centered")

    # Initialize Authentication state
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = ""

    create_usertable()

    if not st.session_state["logged_in"]:
        st.sidebar.title("🔐 Authentication")
        auth_mode = st.sidebar.selectbox("Choose Mode", ["Login", "Sign Up"])

        if auth_mode == "Login":
            st.title("Login to Medical Assistant")
            email = st.text_input("Email ID")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if email and password:
                    hashed_pswd = make_hashes(password)
                    result = login_user(email, hashed_pswd)
                    if result:
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = result[0][0] # Get mapped username from DB!
                        st.success(f"Logged In Successfully!")
                        st.rerun()
                    else:
                        st.error("Incorrect Email ID or Password")
                else:
                    st.warning("Please enter your credentials.")

        elif auth_mode == "Sign Up":
            st.title("Create New Account")
            new_username = st.text_input("New Username (e.g. JohnDoe)")
            new_email = st.text_input("Email ID")
            new_password = st.text_input("New Password", type="password")
            if st.button("Sign Up"):
                if new_username and new_email and new_password:
                    success = add_userdata(new_username, new_email, make_hashes(new_password))
                    if success:
                        st.success("Account successfully created! Please switch to Login menu.")
                    else:
                        st.error("Username already exists! Please choose another.")
                else:
                    st.warning("Please provide a valid username, email, and password")
    
    # ---------------- If User is Logged In ---------------- #
    else:
        st.sidebar.title(f"Welcome, {st.session_state['username']}! 👋")
        
        # --- Chat History Navigation ---
        st.sidebar.divider()
        st.sidebar.markdown("### 🕒 Chat History (Last 7 Days)")
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        chat_dates = get_chat_dates(st.session_state["username"])

        # Always show a way to get back to today's chat
        if st.sidebar.button("➕ Start / Resume Today's Chat", use_container_width=True):
            st.session_state['view_date'] = today_str
            st.session_state['needs_reload'] = True
            st.rerun()

        st.sidebar.markdown("---")
        
        # List up to the last 7 distinct dates
        for d in chat_dates:
            if d != today_str: # Skip today as we mapped it to the big button
                if st.sidebar.button(f"🗓️ Chat from {d}", use_container_width=True):
                    st.session_state['view_date'] = d
                    st.session_state['needs_reload'] = True
                    st.rerun()

        st.sidebar.divider()
        st.sidebar.title("Navigation")
        choice = st.sidebar.radio(
            "Select Mode:",
            ["🧠 Chat with RAG Bot", "👁️ Generate Eye Image"]
        )

        if choice == "🧠 Chat with RAG Bot":
            rag_chatbot_page()
        elif choice == "👁️ Generate Eye Image":
            eye_image_generator_page()
            
        st.sidebar.divider()
        if st.sidebar.button("Logout", type="primary"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = ""
            st.rerun()

if __name__ == "__main__":
    main()
