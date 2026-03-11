# Pentaverse Document Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on the Pentaverse document.

The chatbot retrieves relevant sections of the document using vector search (FAISS) and generates answers using Google's Gemini model.

The system combines document retrieval and large language models to provide accurate answers grounded in the provided document.

---

## Features

- Document-based question answering
- Semantic search using embeddings
- FAISS vector database for fast retrieval
- Gemini model for answer generation
- JSON formatted responses
- Precomputed vector database for faster startup

---

## Project Structure

pentaverse-chatbot/
│
├── inference.py  
├── requirements.txt  
├── .env  
├── Pentaverse-India (1).pdf  
├── faiss_index/  
│   ├── index.faiss  
│   └── index.pkl  
└── README.md  

---

## Requirements

- Python 3.9 or higher
- Internet connection (for Gemini API requests)

---

## Installation

### 1. Install Python

Ensure Python 3.9 or later is installed.

Check installation:

python --version

---

### 2. Install Dependencies

Navigate to the project directory and run:

pip install -r requirements.txt

This installs all required libraries.

---

## API Key Setup

This chatbot requires a Gemini API key.

A `.env` file is already included in the project directory.

Before running the chatbot, you must update the API key inside this file.

### Step 1

Open the `.env` file located in the root directory.

### Step 2

Locate the following variable:

GEMINI_API_KEY=YOUR_API_KEY_HERE

Replace `YOUR_API_KEY_HERE` with your own Gemini API key.

You can generate a Gemini API key here:

https://aistudio.google.com/app/apikey

After updating the key, save the `.env` file.

### Important

- The API key included in the project is a placeholder.
- Each deployment must use its own Gemini API key.
- Do not share or expose API keys publicly.

---

## Running the Chatbot

Start the chatbot using:

python inference.py

You will see a prompt:

User:

Type a question related to the document.

Example:

User: What does Pentaverse do?

The chatbot will respond in JSON format.

Example response:

{
  "question": "What does Pentaverse do?",
  "answer": "Pentaverse provides ..."
}

---

## How the System Works

The chatbot follows a Retrieval-Augmented Generation pipeline:

User Question  
↓  
Embedding Generation  
↓  
FAISS Vector Search  
↓  
Retrieve Relevant Document Chunks  
↓  
Gemini Model  
↓  
Generated Answer  

The retrieved document chunks are used as context so that the model answers based only on information contained in the document.

---

## Vector Database

The FAISS vector database is stored in the `faiss_index/` folder.

This allows the chatbot to start instantly without recomputing embeddings.

If the vector database is missing, the system will automatically regenerate it from the provided PDF.

---

## Security

API keys are loaded from environment variables using the `.env` file.

This prevents sensitive credentials from being stored directly in the code.

Do not commit `.env` files to version control.

---

## Exit

To stop the chatbot, type:

exit

in the terminal.

---

## Dependencies

Key libraries used in this project:

- LangChain
- FAISS
- Sentence Transformers
- Google Gemini API
- python-dotenv
- PyPDF

All dependencies are listed in `requirements.txt`.

---

## Notes

- The chatbot answers questions only using information found in the document.
- If the requested information is not present in the document context, the chatbot will indicate that the information is unavailable.