import numpy as np
import os
from dotenv import load_dotenv
from google import genai
from langchain_ollama import ChatOllama
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

#Chunk function to convert a text into smaller chunks
"""def chunk_text(text, chunk_size=20, overlap=8):
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks"""

# 4. Embed all documents
embedded_docs = []

for doc in documents:
    vector = get_embedding(doc)
    embedded_docs.append({"text": doc, "vector": vector})

# 5. Embed the query
query_vector = get_embedding(query)

# 6. Loop and compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#7. TOP-K RETRIEVAL (IMPORTANT CHANGE)

scores = []

for doc in embedded_docs:
    score = cosine_similarity(query_vector, doc["vector"])
    scores.append((score, doc["text"]))

#8. sort highest similarity first
scores.sort(reverse=True)

top_k = 3
top_docs = [text for score, text in scores[:top_k]]

print("TOP RETRIEVED CHUNKS:")
for i, text in enumerate(top_docs):
    print(f"\n--- Chunk {i+1} ---")
    print(text)


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
