import os
import shutil
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings # Uncomment for FREE local mode

# Import the loader function from your previous script
from ingest import load_and_chunk_pdf

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
CHROMA_PATH = "chroma_db"
PDF_PATH = "sample_medical_guide.pdf"

# 🔑 PASTE YOUR OPENAI API KEY HERE (OR SET AS ENV VARIABLE)
# If you don't have one, get it from platform.openai.com
# It costs less than $0.10 to embed this whole book.

def save_to_chroma(chunks):
    # 1. Clean up old DB if it exists (to start fresh)
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # 2. Initialize the Embedding Model
    # "text-embedding-3-small" is the current standard (cheap & powerful)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # --- FREE LOCAL ALTERNATIVE (If you have no API Key) ---
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # -------------------------------------------------------

    print("🔄 Creating embeddings... (This might take a minute)")
    
    # 3. Create the Vector Store
    # This sends text to OpenAI, gets numbers back, and saves to folder.
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    
    print(f"✅ Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    # Reuse the logic from Phase 1
    chunks = load_and_chunk_pdf(PDF_PATH)
    save_to_chroma(chunks)