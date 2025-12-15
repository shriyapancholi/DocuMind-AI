import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
CHROMA_PATH = "chroma_db"
# 🔑 PASTE YOUR KEY HERE AGAIN (Or use env variable)
# ---------------------------------------------------------
# PROMPT ENGINEERING (The "Deep" Part)
# ---------------------------------------------------------
PROMPT_TEMPLATE = """
You are a strict medical research assistant. Use ONLY the following context to answer the question.

Context:
{context}

Question: 
{question}

Rules:
1. If the answer is not in the context, say "I cannot find this information in the provided guidelines."
2. Do not use outside knowledge.
3. Cite your sources for every claim using the format [Page X].
"""

def query_rag(query_text):
    # 1. Load the Database
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 2. Retrieve Top 5 Matches (Semantic Search)
    # k=5 means "Get the 5 most relevant chunks"
    results = db.similarity_search_with_score(query_text, k=5)

    # 3. Prepare Context with Metadata (The "Citation" Secret)
    context_text = ""
    for doc, _score in results:
        # We append the Page Number to the text so the AI can read it!
        context_text += f"\n[Page {doc.metadata.get('page', 'Unknown')}] {doc.page_content}\n"

    # 4. Formulate the Prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # 5. Generate Answer (Using GPT-4o or GPT-3.5)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) # Temp=0 for maximum factuality
    response = model.invoke(prompt)

    # 6. Output Results
    print("\n🤖 --- AI ANSWER ---")
    print(response.content)
    print("\n📄 --- SOURCES USED ---")
    for doc, _score in results:
        print(f"- Page {doc.metadata.get('page', '?')}: {doc.page_content[:60]}...")

if __name__ == "__main__":
    # Loop to let you ask multiple questions in the terminal
    print("✅ System Ready! Type 'exit' to stop.")
    while True:
        user_query = input("\n🔎 Ask a medical question: ")
        if user_query.lower() == "exit":
            break
        query_rag(user_query)