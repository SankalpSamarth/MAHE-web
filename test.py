import numpy as np
import os
from dotenv import load_dotenv
from google import genai
from langchain_ollama import ChatOllama

load_dotenv()
llm = ChatOllama(model="llama3.2")

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

documents = [
        
        "The University Student Services Office released updated operational guidelines for the Autumn Semester.",

"LIBRARY SERVICES: The Central Library is open from 8:00 AM to 10:00 PM on weekdays and from 10:00 AM to 6:00 PM on weekends. However, during examination periods, the library extends operations until midnight only for students carrying a valid university ID card. Students are not permitted to borrow reference books after 7:00 PM, although digital access remains available throughout operational hours. The Innovation Lab, located beside the library, remains open from 9:00 AM to 8:00 PM on weekdays. Mechanical prototyping equipment such as 3D printers may only be used between 11:00 AM and 5:00 PM due to maintenance schedules.",

"LAUNDRY SERVICES: Laundry collection takes place every Monday, Wednesday, and Friday between 7:30 AM and 9:30 AM near Hostel Block B. Delivery of cleaned clothes occurs after 6:00 PM on the following day. Students who miss collection hours may drop clothes manually at the Laundry Service Room near Gate 2, but express delivery is unavailable for manual submissions.",

"CAFETERIA POLICY:The Main Cafeteria serves breakfast between 7:00 AM and 10:00 AM, lunch from 12:00 PM to 3:00 PM, and dinner from 7:00 PM to 10:00 PM. However, after 9:00 PM, only vegetarian food is served due to reduced kitchen operations. Students with a premium meal subscription may access the Night Counter until 12:30 AM.",

"HOSTEL RULES:Hostel entry closes at 10:30 PM for first-year students and 11:30 PM for second-year students. Final-year students participating in officially approved project work may return until 1:00 AM with digital approval from the department coordinator. Students entering after approved hours must sign an incident register.",

"MEDICAL CENTER:The Campus Medical Center operates from 8:00 AM to 8:00 PM daily. Emergency care remains available 24/7, but dental consultation is restricted to Tuesdays and Thursdays between 2:00 PM and 5:00 PM.",

"SPECIAL NOTICE:During university festivals, all service timings may be revised except emergency medical services. The student handbook should be checked for updated schedules." ]

query = "What time can I use the 3D printer?"

def get_embedding(text, model="gemini-embedding-2"):
    result = client.models.embed_content(model=model, contents=[text])
    return result.embeddings[0].values

#Chunk function to convert a text into smaller chunks
def chunk_text(text, chunk_size=20, overlap=8):
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks

# 4. Embed all documents
embedded_docs = []

for doc in documents:
    chunks = chunk_text(doc)

    for chunk in chunks:
        vector = get_embedding(chunk)
        embedded_docs.append({
            "text": chunk,
            "vector": vector
        })

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


#9. CONTEXT BUILDING
context = "\n\n".join(top_docs)

#10. Providing context to the llm(ollama)
def generate_answer(query, context):
    prompt = f"""
You are a strict and helpful assistant.

RULES:
- Use ONLY the given context.
- If answer is not in context, say "I don't know based on the provided information."
- Do NOT use outside knowledge.

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
