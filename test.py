import numpy as np
import os
from dotenv import load_dotenv
from google import genai
from langchain_ollama import ChatOllama
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="college_docs")

load_dotenv()
llm = ChatOllama(model="llama3.2")

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Pdf to pages to mini chunka
doc = fitz.open("notices.pdf")
full_text = ""
for page in doc:
    full_text += page.get_text()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_text(full_text)


def get_embedding(text, model="gemini-embedding-2"):
    result = client.models.embed_content(model=model, contents=[text])
    return result.embeddings[0].values

# Store chunks in ChromaDB
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

# For Semantic Matching with the original query

query = "What is the function of the tailstock?"

def rewrite_query(original_query):
    prompt = f"""Generate 3 different ways to ask the following question. 
Return ONLY the 3 questions, numbered 1-3, nothing else.

Original question: {original_query}

Rewritten questions:"""
    
    response = llm.invoke(prompt)
    lines = response.content.splitlines()
    parsed_list = [line.split(". ", 1)[1] for line in lines if line.strip() and line[0].isdigit()]
    return parsed_list


rewritten_queries = rewrite_query(query)
print("Rewritten queries:")
for q in rewritten_queries:
    print(q)

#Query ChromaDB

all_chunks = []
seen = set()

queries_to_search = [query] + rewritten_queries

for q in queries_to_search:
    results = collection.query(
        query_embeddings=[get_embedding(q)],
        n_results=5
    )
    for chunk in results["documents"][0]:
        if chunk not in seen:
            seen.add(chunk)
            all_chunks.append(chunk)

top_docs = all_chunks[:5]

print(f"\nTotal unique chunks retrieved: {len(all_chunks)}")
print("\n--- RETRIEVED CHUNKS ---")
for i, chunk in enumerate(all_chunks):
    print(f"\nChunk {i+1}:\n{chunk}")

#9. CONTEXT BUILDING
context = "\n\n".join(top_docs)

#10. Providing context to the llm(ollama)
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


#11. OUTPUT
final_answer = generate_answer(query, context)

print("\nFINAL ANSWER:\n")
print(final_answer)
