import os

from langchain_community.llms import Ollama  # Using Ollama LLM for text generation
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Setup LLM (Ollama)
def load_llm():
    llm = Ollama(
        model="qwen3:4b",  # or "mistral", "llama2", etc. based on your pulled models
        temperature=0.8,
    )
    return llm

# Step 2: Create custom prompt
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

CUSTOM_PROMPT_TEMPLATE = """
You are a highly accurate and concise medical assistant chatbot.

Use the following context and your knowledge to answer the user’s medical question truthfully and concisely.

Caution: Only answer if you're confident. If unsure or the question requires a licensed professional, say:
*"I'm not qualified to give a definitive answer. Please consult a licensed healthcare provider."*

Context: {context}
Question: {question}

Answer:
"""

# Step 3: Load FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Run QA with user query
if __name__ == "__main__":
    user_query = input("Write your query here: ")
    response = qa_chain.invoke({'query': user_query})
    print("\nRESULT:")
    print(response["result"])
    print("\nSOURCE DOCUMENTS:")
    for doc in response["source_documents"]:
        print(doc)