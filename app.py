import streamlit as st
import os
import tempfile
import uuid
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------
# CONFIGURATION & AUTHENTICATION
# ---------------------------------------------------------
st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")

# 🔐 HYBRID AUTHENTICATION
# 1. Try to get key from Environment (Great for Deploying to Streamlit Cloud)
api_key = os.getenv("OPENAI_API_KEY")

# 2. If no key found in Env, ask the user for it (Great for GitHub/Local users)
if not api_key:
    with st.sidebar:
        st.divider()
        st.markdown("### 🔐 API Key Needed")
        st.info("This app requires an OpenAI API Key to process documents.")
        api_key = st.text_input("Enter OpenAI API Key:", type="password")
        st.caption("Get one [here](https://platform.openai.com/api-keys)")

# 3. Stop if no key is provided
if not api_key:
    st.warning("⚠️ Please enter an API Key in the sidebar to start the engine.")
    st.stop()

# 4. Set the key for LangChain
os.environ["OPENAI_API_KEY"] = api_key

# ---------------------------------------------------------
# PROMPT TEMPLATES
# ---------------------------------------------------------
PROMPT_TEMPLATE = """
You are an expert technical research assistant. Use ONLY the following context to answer the question.

Context:
{context}

Question: 
{question}

Rules:
1. If the answer is not in the context, say "I cannot find this information in the provided document."
2. Do not use outside knowledge.
3. Cite your sources for every claim using the format [Page X].
"""

# ---------------------------------------------------------
# STATE MANAGEMENT
# ---------------------------------------------------------
if "sessions" not in st.session_state:
    default_id = str(uuid.uuid4())
    st.session_state.sessions = {
        default_id: {
            "messages": [],
            "db_path": None,
            "file_name": None,
            "title": "New Investigation"
        }
    }
    st.session_state.active_session_id = default_id

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def get_active_session():
    if st.session_state.active_session_id not in st.session_state.sessions:
        if st.session_state.sessions:
            st.session_state.active_session_id = list(st.session_state.sessions.keys())[0]
        else:
            create_new_session()
            
    return st.session_state.sessions[st.session_state.active_session_id]

def create_new_session():
    new_id = str(uuid.uuid4())
    st.session_state.sessions[new_id] = {
        "messages": [],
        "db_path": None,
        "file_name": None,
        "title": "New Investigation"
    }
    st.session_state.active_session_id = new_id

def delete_session(session_id):
    if len(st.session_state.sessions) <= 1:
        create_new_session()
        del st.session_state.sessions[session_id]
    else:
        del st.session_state.sessions[session_id]
        if st.session_state.active_session_id == session_id:
            st.session_state.active_session_id = list(st.session_state.sessions.keys())[0]

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(raw_documents)

    new_db_path = tempfile.mkdtemp()
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(
        documents=documents, 
        embedding=embedding_function, 
        persist_directory=new_db_path
    )
    
    session = get_active_session()
    session["db_path"] = new_db_path
    session["file_name"] = uploaded_file.name
    session["title"] = f"📄 {uploaded_file.name}"
    
    os.remove(tmp_path)
    return len(documents)

# ---------------------------------------------------------
# SIDEBAR: UNIVERSAL WORKSPACE
# ---------------------------------------------------------
with st.sidebar:
    st.header("🗂️ Workspace")
    
    if st.button("➕ Start New Chat", use_container_width=True):
        create_new_session()
        st.rerun()

    st.markdown("---")
    
    search_query = st.text_input("🔍 Search Chats", placeholder="Type to filter...")
    
    st.markdown("### 🕒 Recent History")
    
    session_ids = list(st.session_state.sessions.keys())
    session_ids.reverse() 
    
    for session_id in session_ids:
        session_data = st.session_state.sessions[session_id]
        
        if search_query.lower() in session_data["title"].lower():
            if session_id == st.session_state.active_session_id:
                label = f"🔵 {session_data['title']}"
            else:
                label = f"⚪ {session_data['title']}"
                
            if st.button(label, key=session_id, use_container_width=True):
                st.session_state.active_session_id = session_id
                st.rerun()

    st.markdown("---")
    
    active_session = get_active_session()
    
    with st.expander("⚙️ Session Settings", expanded=True):
        new_title = st.text_input("Rename Chat", value=active_session["title"])
        if new_title != active_session["title"]:
            active_session["title"] = new_title
            st.rerun()
            
        if st.button("🗑️ Delete Chat", type="primary", use_container_width=True):
            delete_session(st.session_state.active_session_id)
            st.rerun()

    st.markdown("### 📥 Document Upload")
    uploaded_file = st.file_uploader("Upload any PDF (Legal, Tech, Medical)", type="pdf")

    if uploaded_file:
        if active_session["file_name"] != uploaded_file.name:
            with st.spinner("🧠 Analyzing Document..."):
                try:
                    num_chunks = process_pdf(uploaded_file)
                    st.success(f"Processed {num_chunks} knowledge chunks!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# ---------------------------------------------------------
# MAIN INTERFACE: UNIVERSAL
# ---------------------------------------------------------
st.title(f"🧠 {active_session['title']}") 

if not active_session["db_path"]:
    st.info("👈 **Start here:** Upload a PDF document in the sidebar to begin analysis.")
else:
    for message in active_session["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a verifiable question about this document..."):
        
        st.chat_message("user").markdown(prompt)
        active_session["messages"].append({"role": "user", "content": prompt})

        with st.spinner("Searching knowledge base..."):
            try:
                embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
                db = Chroma(persist_directory=active_session["db_path"], embedding_function=embedding_function)
                
                results = db.similarity_search_with_score(prompt, k=5)

                context_text = ""
                sources_list = []
                for doc, _score in results:
                    page_num = doc.metadata.get('page', 'Unknown')
                    context_text += f"\n[Page {page_num}] {doc.page_content}\n"
                    sources_list.append(f"**Page {page_num}:** {doc.page_content[:100]}...")

                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                final_prompt = prompt_template.format(context=context_text, question=prompt)
                
                model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                response = model.invoke(final_prompt)
                answer = response.content

                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("📚 Verified Citations"):
                        for source in sources_list:
                            st.markdown(f"- {source}")
                
                active_session["messages"].append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error: {e}")