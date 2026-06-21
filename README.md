# Local PDF-QA RAG System

An end-to-end, privacy-first Retrieval-Augmented Generation (RAG) system engineered to transform unstructured PDF documents into an intelligent, interactive knowledge base. 

This system bridges the gap between semantic vector space and exact document grounding by combining dense embeddings, neural reranking, and a local Large Language Model (LLM) to deliver hallucination-free answers backed by strict, page-level source attribution.

---

## Repository Structure & Core Workflow

The project is modularly structured to separate the core engineering logic from the user interface:

* **`utility.py`**: The engine room of the project. Contains the base implementation for layout-aware PDF parsing, chunking logic, metadata extraction, ChromaDB vector indexing, and Cross-Encoder reranking wrappers.
* **`app.py` & `utility_app.py`**: The frontend implementation layer. These files leverage Streamlit to build a seamless, web-based interface for interactive PDF uploads, ingestion status monitoring, and real-time chat with visible source attributions.

---

## Tech Stack

* **LLM Engine:** [Ollama](https://ollama.com/) running **Llama 3** (Fully local inference)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Metadata-filtered vector storage)
* **Dense Embeddings:** **BGE Embeddings** (`BAAI/bge-large-en-v1.5`)
* **Neural Reranking:** **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`)
* **Web Framework:** [Streamlit](https://streamlit.io/)

---

## Key Features & Pipeline

### 1. Ingestion & Core Processing (`utility.py`)
* Layout-aware PDF parsing to preserve tables, headers, and reading flow.
* Automated text-splitting with granular metadata mapping (Source File, Page Number, Chunk Index) for absolute traceability.

### 2. Intelligent Two-Stage Retrieval
* **Stage 1 (Coarse Retrieval):** Fast semantic lookup via BGE Embeddings in ChromaDB to isolate the top 25 candidate chunks.
* **Stage 2 (Fine Re-ranking):** Cross-Encoder evaluation scores true context relevance, fixing the "lost-in-the-middle" LLM window problem.

### 3. Streamlit Interface (`app.py`)
* Drag-and-drop PDF ingestion.
* Conversational chat window pulling grounded answers and displaying exact citations (e.g., `[Document_Name.pdf, Page 14]`).

---

## Active Development Roadmap

As the system moves toward production readiness, the following enhancements are currently being integrated:

- [ ] **Hybrid Retrieval Engine:** Blending dense vector search with BM25 sparse keyword matching for better technical terms/ID handling.
- [ ] **Context Compression:** Integrating LLMLingua to compress retrieved tokens, reducing local inference latency without sacrificing accuracy.

---

