# AI Chatbot With RAG

## What is the RAG?

RAG (Retrieval-Augmented Generation) is an AI framework that combines the strengths of traditional information retrieval systems (such as search and databases) with the capabilities of generative large language models (LLMs).

## How it work vector databases?

A vector database works by representing data as high-dimensional vectors (arrays of numbers) and using mathematical similarity metrics to find the most relevant information. Unlike traditional databases that look for exact keyword matches, vector databases find items that are "conceptually" or "semantically" similar.

## What is the problem we want fix?

## What we did?

We developed AI Chatbot with RAG technologies. The ChromaDB ensures vector database need and we used Google Deepminds open source model gemma3:4b.

## Features

- Ollama
- ChromaDB
- FastAPI

### Used Model (LLM)

**Name:** gemma3:4b
**Parameter:** 4B parameter
**Context Window:** 128k
**Visit website:** [Ollama gemma3 website page](https://ollama.com/library/gemma3:4b).

---

## Deployment

### Download Ollama

#### For Windows

Click to [Ollama Windows Download](https://ollama.com/download/windows).

Click to

```bash
windows + r

ollama run gemma3:1b
```

### Clone the repo

```bash
git clone <repo>

cd <repo>
```

### Create Virtual Environment

```bash
python -m venv venv

venv\Scripts\activate

pip install --no-cache-dir -r requirements.txt
```

### Start the app

```bash
python -m app.main.main
```

### Gets Swagger UI

Click to [Swagger UI](http://localhost:2222/docs#/).
