import numpy as np
import os
from dotenv import load_dotenv
from google import genai
from langchain_ollama import ChatOllama
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import glob

# Load environment variables from .env file
load_dotenv()

# Initialize Ollama LLM
llm = ChatOllama(model="llama3.2")

# Initialize Google Generative AI client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Setup ChromaDB persistent storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="college_docs")

# Read PDF and extract full text
full_text = ""
pdf_files = glob.glob("documents/*.pdf")
for pdf_path in pdf_files:
    doc = fitz.open(pdf_path)
    for page in doc:
        full_text += page.get_text()

# Split full text into overlapping chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_text(full_text)

# Convert text to vector using Google embedding model
def get_embedding(text, model="gemini-embedding-2"):
    result = client.models.embed_content(model=model, contents=[text])
    return result.embeddings[0].values

# Ingest chunks into ChromaDB only if collection is empty
if collection.count() == 0:
    print("Ingesting documents...")
    for i, chunk in enumerate(documents):
        vector = get_embedding(chunk)
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[vector],
            documents=[chunk]
        )
    print(f"Total chunks stored: {collection.count()}")
else:
    print(f"Collection already has {collection.count()} chunks, skipping ingestion.")

# Rewrite query 3 ways to improve semantic matching
def rewrite_query(original_query):
    prompt = f"""Generate 3 different ways to ask the following question. 
Return ONLY the 3 questions, numbered 1-3, nothing else.
Original question: {original_query}
Rewritten questions:"""
    response = llm.invoke(prompt)
    lines = response.content.splitlines()
    return [line.split(". ", 1)[1] for line in lines if line.strip() and line[0].isdigit()]

# Generate final answer using retrieved context
def generate_answer(query, context):
    prompt = f"""
You are a strict and helpful assistant. Read the context carefully and answer only from it.
RULES:
- Find the relevant sentence in the context and base your answer on it.
- Quote or closely paraphrase the context. Do not add anything else.
- If the answer is truly not present, say: "I don't have information about this in the provided documents."
- Do NOT guess, infer, or use outside knowledge under any circumstances.
CONTEXT:
{context}
QUESTION:
{query}
ANSWER:
"""
    response = llm.invoke(prompt)
    return response.content

# Main RAG pipeline — takes a query and returns an answer
def ask(query):
    # Step 1: rewrite query for better retrieval
    rewritten_queries = rewrite_query(query)

    # Step 2: search ChromaDB with original + rewritten queries
    all_chunks = []
    seen = set()
    for q in [query] + rewritten_queries:
        results = collection.query(
            query_embeddings=[get_embedding(q)],
            n_results=5
        )
        # Step 3: deduplicate retrieved chunks
        for chunk in results["documents"][0]:
            if chunk not in seen:
                seen.add(chunk)
                all_chunks.append(chunk)

    # Step 4: build context from top chunks
    context = "\n\n".join(all_chunks[:5])

    # Step 5: generate and return final answer
    return generate_answer(query, context)
