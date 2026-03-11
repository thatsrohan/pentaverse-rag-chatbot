# %%
import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
RELEVANT_CHUNKS=5

# -------- LOADING ENV VARIABLES --------

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-2.5-flash-lite")


# -------- EMBEDDING MODEL --------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------- LOADING OR CREATING VECTOR DATABASE --------

if os.path.exists("faiss_index"):

    print("Loading existing FAISS index...")

    vectorstore = FAISS.load_local(
        "faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

else:

    print("Creating FAISS index from PDF...")

    # -------- LOADING PDF --------

    loader = PyPDFLoader("Pentaverse-India (1).pdf")
    documents = loader.load()

    # -------- SPLITTING TEXT --------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    # -------- CREATING VECTOR DATABASE --------

    vectorstore = FAISS.from_documents(
        docs,
        embedding_model
    )

    # -------- SAVING FAISS INDEX --------

    vectorstore.save_local("faiss_index")

    print("FAISS index saved!")


# -------- CHAT FUNCTION --------

def ask(question):

    results = vectorstore.max_marginal_relevance_search(question, k=RELEVANT_CHUNKS) #List of LangChain Document objects

# Extracting the text (page_content) from each retrieved Document 
#and combine them into a single context string separated by new lines to send to the LLM

    context = "\n".join([doc.page_content for doc in results]) 

    prompt =  f"""
You are an AI assistant for Pentaverse.

Answer the user's question using ONLY the information provided in the context below.

Rules:
- If the answer is not present in the context, say: "I don't have that information in the provided document."
- Do NOT use outside knowledge.
- Keep the answer concise and clear.

Context:
{context}

Question:
{question}
"""          #--------HELPS MINIMIZE HALLUCINATION BY PROVIDING A CLEAR PROMPT WITH RULES--------



    response = model.generate_content(prompt)

    return {
        "question": question,
        "answer": response.text,
    }


# -------- CHAT LOOP --------

while True:

    q = input("User: ")

    if q.lower() == "exit":
        break

    answer = ask(q)

    print("Bot:", answer)

# %%