# %%
import os
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings

RELEVANT_CHUNKS = 5

# -------- LOADING ENV VARIABLES --------

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-2.5-flash-lite")

# -------- GEMINI EMBEDDING WRAPPER --------

class GeminiEmbeddings(Embeddings):

    def embed_documents(self, texts):

        vectors = []

        for text in texts:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text
            )
            vectors.append(result["embedding"])

        return vectors

    def embed_query(self, text):

        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text
        )

        return result["embedding"]


embedding_model = GeminiEmbeddings()

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

    context = ""

    try:

        results = vectorstore.max_marginal_relevance_search(question, k=RELEVANT_CHUNKS)

        context = "\n".join([doc.page_content for doc in results])

        prompt = f"""
You are the official AI assistant for Pentaverse.

Your role is to help users understand Pentaverse, including information about the hackathon, participation, phases, rules, schedules, and related details.

Identity and conversational behavior:
- If a user asks questions such as "Who are you?", "What are you?", or "What can you do?", respond that you are the Pentaverse AI assistant designed to help users with information about Pentaverse.
- If a user greets you (e.g., "hello", "hi", "hey"), respond politely and ask how you can assist.

Knowledge usage rules:
- Answer the user's question using ONLY the information provided in the context below.
- Do NOT use outside knowledge.
- Do NOT make assumptions or fabricate details.
- If the information needed to answer the question is not present in the context, respond with:
"I don't have that information at the moment based on the available knowledge."

Context reasoning rules:
- Carefully read all provided context before answering.
- If multiple pieces of context are relevant, combine them to form a clear answer.
- Prefer the most relevant and specific information from the context.

Response style guidelines:
- Keep answers concise, clear, and informative.
- Use complete sentences.
- Avoid mentioning the context, documents, or retrieval process.
- Avoid speculation or guesses.

Context:
{context}

Question:
{question}
"""

        # -------- RETRY GEMINI CALL --------

        for attempt in range(3):

            try:

                response = model.generate_content(prompt)

                return {
                    "question": question,
                    "answer": response.text,
                }

            except Exception:

                if attempt < 2:
                    time.sleep(2)
                else:
                    raise

    except Exception:

        # -------- GRACEFUL DEGRADATION --------

        return {
            "question": question,
            "answer": f"""
The AI model is temporarily unavailable.

Here is relevant information retrieved from the Pentaverse knowledge base:

{context}
"""
        }

# -------- CHAT LOOP FOR LOCAL TESTING --------

if __name__ == "__main__":

    while True:

        q = input("User: ")

        if q.lower() == "exit":
            break

        answer = ask(q)

        print("Bot:", answer)

# %%