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

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
documents = splitter.split_text(full_text)

query = "What happens to the job if thread cutting goes wrong?"

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

# Query ChromaDB
results = collection.query(
    query_embeddings=[get_embedding(query)],
    n_results=3
)

top_docs = results["documents"][0]


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
