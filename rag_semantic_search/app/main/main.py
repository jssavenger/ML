from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import uvicorn

# Vector Database
from ..classes.collections import VectorDB

# Helpers
from ..utils.helper_functions import read_json, preprocess_dataset

# Routers
from ..api.llm_api import router as LLMRouters

# Services
from ..classes.llm import LLM

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"\nApp starting now...\n")
        
    dataset = read_json()
    ids, documents = preprocess_dataset(dataset)

    vector_db = VectorDB()
    vector_db.create_collection()
    vector_db.add_collection(ids, documents)
    
    print(f"Collection created.")

    app.state.vector_db = vector_db
    app.state.llm_model = LLM()
    
    yield
    
    print(f"\nApp is down.\n")

app = FastAPI(
    title="Rag Semantic Search",
    description="Semantic search with local LLM and ChromaDB.",
    version="0.01",
    lifespan=lifespan
    )

app.include_router(LLMRouters)

@app.get("/healthy", tags=['Healthy'])
async def healthy_check():
    try:
        return {
            "status": True,
            "message": "App is healthy."
        }
    except Exception as e:
        return {
            "status": False,
            "message": f"Error: {str(e).strip()}"
        }
        

if __name__ == "__main__":
    uvicorn.run("app.main.main:app", host="0.0.0.0", port=2222, reload=True)