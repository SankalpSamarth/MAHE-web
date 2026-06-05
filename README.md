# SemanticIQ: Semantic RAG Document Intelligence Platform

AI-powered Retrieval-Augmented Generation (RAG) system designed to help students instantly access information from university documents such as notices, circulars, fee structures, hostel policies, academic schedules, and administrative communications.

---

## Overview

Students often receive hundreds of emails, notices, and PDF documents throughout an academic year. Locating a specific piece of information can become time-consuming and inefficient.

SemanticIQ addresses this problem by transforming institutional documents into a searchable knowledge base. Using semantic search, OCR, vector embeddings, and a local language model, the system provides accurate answers grounded in university documents.

---

## Problem Statement

| Challenge | Impact |
|------------|----------|
| Large volume of university emails | Important information gets buried |
| Multiple PDF circulars and notices | Difficult to search manually |
| Scanned documents | Traditional text search fails |
| Repetitive student queries | Increased administrative burden |
| Information scattered across sources | Reduced accessibility |

---

## Solution

The system performs the following operations:

1. Extracts text from both digital and scanned PDF documents.
2. Converts document content into vector embeddings.
3. Stores embeddings in a persistent ChromaDB vector database.
4. Expands user queries using a local language model.
5. Retrieves semantically relevant document chunks.
6. Generates context-aware responses based solely on retrieved information.

---

## System Architecture

```text
                           ┌─────────────────┐
                           │ PDF Documents   │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Text Extraction │
                           │ + OCR Pipeline  │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Document        │
                           │ Chunking        │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Gemini          │
                           │ Embeddings      │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ ChromaDB        │
                           │ Vector Store    │
                           └────────┬────────┘
                                    │
                                    ▼
User Query ──► Query Rewriting ──► Semantic Retrieval
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Retrieved       │
                           │ Context         │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Local LLM       │
                           │ (Qwen 2.5)      │
                           └────────┬────────┘
                                    │
                                    ▼
                               Response
```

---

## Core Features

| Feature | Description |
|----------|------------|
| OCR Support | Extracts text from scanned PDF documents |
| Semantic Search | Retrieves information based on meaning rather than exact keywords |
| Query Expansion | Generates multiple query variations for improved retrieval |
| Persistent Vector Store | ChromaDB-based local storage |
| Local LLM Integration | Uses Qwen 2.5 via Ollama |
| Context-Based Generation | Responses generated using retrieved document context |
| Streamlit Interface | Interactive conversational UI |

---

## Technology Stack

| Category | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Programming Language | Python |
| Embeddings | Google Gemini Embeddings |
| Vector Database | ChromaDB |
| LLM | Qwen 2.5 (Ollama) |
| OCR Engine | Tesseract OCR |
| PDF Processing | PyMuPDF |
| PDF-to-Image Conversion | pdf2image |
| Frameworks | LangChain |

---

## Project Structure

```text
MAHE-web/
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
│
├── documents/
│   ├── circulars/
│   ├── notices/
│   └── policies/
│
├── chroma_db/
│
└── README.md
```

---

## Workflow

| Stage | Description |
|---------|------------|
| Document Ingestion | Reads PDF documents from local storage |
| OCR Processing | Converts scanned pages into text |
| Chunk Creation | Splits documents into retrievable segments |
| Embedding Generation | Creates vector representations |
| Vector Storage | Stores embeddings in ChromaDB |
| Query Expansion | Rewrites user query into multiple variants |
| Retrieval | Retrieves relevant chunks |
| Response Generation | Produces final answer using retrieved context |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/MAHE-web.git
cd MAHE-web
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

macOS/Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

---

## Running the Application

Start Ollama:

```bash
ollama serve
```

Ensure the model is available:

```bash
ollama pull qwen2.5:7b
```

Launch Streamlit:

```bash
streamlit run app.py
```

---

## Example Queries

| Category | Example Query |
|------------|--------------|
| Hostel | What is the hostel fee refund policy? |
| Academic | When are the semester examinations scheduled? |
| Library | What are the library timings? |
| Administration | How can I apply for a room category change? |
| Fees | What are the hostel charges for the current academic year? |

---

## Future Enhancements

| Planned Feature | Status |
|-----------------|---------|
| Citation-Based Responses | Planned |
| WhatsApp Integration | Planned |
| Multi-User Authentication | Planned |
| Department-Specific Knowledge Bases | Planned |
| Mobile Application | Planned |
| Real-Time Email Synchronization | Planned |

---

## Author

**Sankalp Samarth**

First-Year Software Engineering Student

This project was developed to address a practical challenge faced by students: quickly locating accurate information from large volumes of institutional documents and communications.

---
