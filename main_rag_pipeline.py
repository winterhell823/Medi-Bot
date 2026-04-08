from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from IPython.display import Image
from sentence_transformers import SentenceTransformer
import os



#Load image metadata from folder (folder name = label)
def load_image_docs(image_root="EYE"):
    image_docs = []
    
    for folder in os.listdir(image_root):
        folder_path = os.path.join(image_root, folder)
        if os.path.isdir(folder_path):
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(folder_path, img_file)
                    metadata = {
                        "part": folder,
                        "filename": img_file,
                        "path": full_path
                    }
                    image_docs.append(Document(page_content=folder, metadata=metadata))
    
    return image_docs

#Load sentence transformer embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Create FAISS index
def build_vector_store(docs, embedding_model):
    return FAISS.from_documents(docs, embedding_model)

#Retrieve top-1 matching image based on query
def retrieve_similar_image(vector_store, query):
    result = vector_store.similarity_search(query, k=1)
    return result[0].metadata["path"] if result else None

#Main function to load everything and get best image
def get_best_image_from_query(user_query, image_root="EYE"):
    docs = load_image_docs(image_root)
    embedding_model = get_embedding_model()
    vector_store = build_vector_store(docs, embedding_model)
    return retrieve_similar_image(vector_store, user_query)

#Display image from path
def show_image(image_path):
    if image_path and os.path.isfile(image_path):
        img = Image.open(image_path)
        img.show()
        print(f"✅ Displayed image: {image_path}")
    else:
        print("❌ Image not found.")

#Example CLI usage
if __name__ == "__main__":
    user_query = input("🔍 Enter your question (semantic): ")
    best_image_path = get_best_image_from_query(user_query)
    show_image(best_image_path)