# Semantic Search with Qdrant & Sentence Transformers

## Overview
This project demonstrates how to build a **semantic search engine** using:
- **Sentence Transformers** for text embeddings.
- **Qdrant** as a vector database for fast similarity searches.
- **FastAPI** for exposing search functionality as a web service.

## Features
- Convert text-based customer support tickets into **vector embeddings**.
- Store embeddings in **Qdrant** for fast retrieval.
- Perform **semantic similarity search** on customer issues.

## Installing

docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
pip install -r requirements.txt