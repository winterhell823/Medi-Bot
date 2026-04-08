# Demo Live Link
  https://medical-chatassistant.streamlit.app/

# Medical RAG Chatbot & Eye Image Generator 🏥👁️

An intelligent, locally-hosted medical assistant powered by **Retrieval-Augmented Generation (RAG)** and **Ollama**. This application allows users to ask complex medical questions based on specialized PDF data and accurately retrieves semantic images of eye structures. This project demonstrates backend skills, local AI orchestration, database design, and frontend visualization.

## 🌟 Features
- **Local Large Language Model:** Completely private and secure text generation using Ollama (`qwen3:4b`). No data leaves your machine.
- **Retrieval-Augmented Generation (RAG):** Context-aware intelligence using FAISS vector databases and HuggingFace Embeddings (`all-MiniLM-L6-v2`) to accurately parse and retrieve information from medical documents.
- **Semantic Image Retrieval:** Ask about eye anatomy (e.g., "Cornea" or "Retina") and the system retrieves the most mathematically similar image based on semantic embeddings.
- **Full Authentication System:** Built-in secure user Sign Up and Login, utilizing salted SHA-256 password hashing and SQLite.
- **7-Day Chat History:** Automatically tracks and saves conversation history to the database, allowing users to seamlessly review their past interactions by date.
- **Modern UI:** Clean, responsive, and intuitive web interface built entirely with Streamlit.

## 🛠️ Tech Stack
- **Frontend & App Framework:** Streamlit
- **LLM Engine:** Ollama (Local)
- **RAG & Orchestration:** LangChain, LangChain-Classic
- **Vector Database:** FAISS
- **Embeddings Model:** HuggingFace (`sentence-transformers`), PyTorch
- **Backend & Database:** Python 3.10+, SQLite3

---

## 📸 Demonstration
<!-- Developer Note: Take a screenshot of your app or record a 15-second screen-recording with QuickTime/Loom, convert it to a GIF, drag-and-drop it here inside this README file! This is the #1 most important thing interviewers look at! -->
> **Insert App Screenshot or Demo GIF Here!**

---

## 🚀 Quick Start / Local Setup

Interviewers or developers looking to run this application locally will need to have [Ollama](https://ollama.com/) installed on their machine.

### 1. Install and Start Ollama
Download Ollama from their website and pull the required model:
```bash
ollama run qwen3:4b
```

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/Rag-Bot-main.git
cd Rag-Bot-main
```

### 3. Install Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the Application
Launch the Streamlit server:
```bash
streamlit run mainbot.py
```
Then, open your browser to `http://localhost:8501`.
Creating an Ai-Bot (Medi-Bot) using RAG Model using Langchain , ollama for text generation and image generation
