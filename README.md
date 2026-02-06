# Intelligent Enterprise Retrieval System  
## RAG Pipeline + Async FastAPI Backend

This project implements a **Retrieval-Augmented Generation (RAG) system** that combines **LlamaIndex** with a **Vector Database (ChromaDB)** and an **Asynchronous FastAPI backend** to query unstructured enterprise data.

The goal of this project is to build a **low-latency, semantic search engine** that allows users to query internal documents (PDFs) using Natural Language, overcoming the limitations of traditional keyword search and LLM hallucinations.

---

## Problem Statement

Traditional enterprise search systems often suffer from:
- Inability to understand context (Keyword Search limitations)
- Hallucinations when using standalone LLMs for internal data
- High latency in processing large document sets

While **LLMs (like GPT-3.5)** have strong reasoning capabilities, they lack access to private, real-time data.
On the other hand, **Keyword Search** is fast but fails to capture semantic meaning.

This project combines both approaches using a **RAG framework** to build a **context-aware and hallucination-free** answering engine.

---

## Project Objective

- Design a scalable RAG pipeline using LlamaIndex and ChromaDB
- Optimize API latency using Asynchronous programming in FastAPI
- Implement advanced chunking strategies to preserve context
- Deploy the solution using Docker for reproducibility

---

## Key Contributions

- **Hybrid Search Architecture:** Combines semantic embedding search with LLM generation.
- **Async API Design:** Utilized `async/await` in FastAPI to handle concurrent user requests efficiently.
- **Advanced Data Ingestion:** Implemented `SentenceSplitter` for optimal text chunking (1024 tokens) with overlap.
- **Containerized Deployment:** Fully dockerized application for easy deployment and scaling.

---

## System Workflow

Unstructured Data (PDFs)
↓
Data Ingestion & Chunking
↓
Vector Embedding (OpenAI)
↓
ChromaDB Storage
↓
User Query (Async API)
↓
Semantic Retrieval & RAG
↓
Context-Aware Response

---

## Methodology

### 1. Data Ingestion & Chunking
- Loaded unstructured PDF documents using `SimpleDirectoryReader`
- Applied **Sentence Splitting** (chunk size: 1024, overlap: 200) to maintain context across boundaries
- Ensured semantic integrity of long-form documents

### 2. Vector Database Integration
- Generated embeddings using **OpenAI's text-embedding-3-small**
- Stored vectors in **ChromaDB** (Persistent Client)
- Enabled efficient similarity search for high-speed retrieval

### 3. Retrieval-Augmented Generation (RAG)
- Implemented **LlamaIndex Query Engine** with custom prompt templates
- Retrieved top-k similar context nodes for every user query
- Injected context into the LLM prompt to ground the answer in facts

### 4. High-Performance API Layer
- Developed using **FastAPI** with Asynchronous endpoints (`async def`)
- Implemented **Lazy Loading** for the RAG engine to reduce startup overhead
- Added **Logging & Error Handling** for enterprise-grade robustness

---

## Environment & Simulation

- **Framework:** LlamaIndex & FastAPI
- **Model:** GPT-3.5-Turbo (Generation) & Text-Embedding-3 (Embeddings)
- **Containerization:** Docker & Docker Compose
- **API Testing:** Swagger UI & Postman

---

## Results & Performance

- **Latency Optimization:** Reduced query overhead using async handling
- **High Precision:** Achieved accurate context retrieval using overlapping chunks
- **Zero Hallucination:** System explicitly states when information is missing from documents
- **Scalability:** Dockerized architecture allows for horizontal scaling

> **Observation:** The use of overlapping window chunking significantly improved the retrieval accuracy for queries spanning multiple paragraphs.

---

## Tech Stack

**Programming Language**
- Python 3.10+

**AI & Frameworks**
- LlamaIndex
- OpenAI API
- FastAPI (Async)

**Database & DevOps**
- ChromaDB (Vector Store)
- Docker
- Uvicorn

---
