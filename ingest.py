import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# PDF_PATH: Put a real medical PDF in the same folder. 
# Download a sample like "WHO Diabetes Guidelines" if you don't have one.
PDF_PATH = "sample_medical_guide.pdf" 

# CHUNK_SIZE: 1000 characters is a sweet spot for RAG. 
# Too small = AI lacks context. Too large = Confuses the AI with too much info.
CHUNK_SIZE = 1000 

# CHUNK_OVERLAP: 200 characters overlap ensures we don't cut a sentence 
# in the middle of an important fact.
CHUNK_OVERLAP = 200

def load_and_chunk_pdf(file_path):
    """
    Loads a PDF and splits it into chunks with metadata (Page Numbers).
    """
    print(f"📄 Loading PDF: {file_path}...")
    
    # 1. Load the PDF
    # PyPDFLoader automatically extracts page numbers into metadata!
    loader = PyPDFLoader(file_path)
    raw_documents = loader.load()
    print(f"✅ Loaded {len(raw_documents)} pages.")

    # 2. Split the Text (The "Deep" Part)
    # We use RecursiveCharacterTextSplitter because it tries to split 
    # by paragraphs first, then sentences, then words. It respects structure.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""] # Try to split by paragraph first
    )

    documents = text_splitter.split_documents(raw_documents)
    print(f"✅ Split into {len(documents)} chunks.")
    
    return documents

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    # Create a dummy PDF if it doesn't exist (just for testing this script)
    if not os.path.exists(PDF_PATH):
        print("⚠️ No PDF found. Please add a 'sample_medical_guide.pdf' to this folder.")
    else:
        chunks = load_and_chunk_pdf(PDF_PATH)

        # 3. VERIFICATION (Crucial for your demo)
        # We print the first 2 chunks to prove we have the "Source" and "Page Number".
        print("\n🔍 --- INSPECTING CHUNK 1 ---")
        first_chunk = chunks[0]
        print(f"CONTENT:\n{first_chunk.page_content[:200]}...") # Show first 200 chars
        print(f"\nMETADATA: {first_chunk.metadata}") 
        # Output should look like: {'source': 'sample_medical_guide.pdf', 'page': 0}
        
        print("\n------------------------------------------------")
        print(f"🚀 Ready for Phase 2! We have {len(chunks)} chunks ready for embedding.")