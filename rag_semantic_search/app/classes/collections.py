from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from pydantic import BaseModel
from pathlib import Path
import chromadb
import os

_CURRENT_PATH= Path(__file__).parent.parent
_ENV_FILE_PATH= _CURRENT_PATH / ".env"

load_dotenv(dotenv_path=_ENV_FILE_PATH)

_OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

class VectorDB:
    def __init__(self, vector_db_settings: BaseModel):
        self.client = chroma_client = chromadb.Client()
        self.openai_api_key = _OPENAI_API_KEY
        self.settings = vector_db_settings.vector_db_settings
        self.model_name = self.settings.model_name
        self.collection_name = self.settings.collection_name
        
    def create_collection(self):
        """Creates ChromaDB vector database collection.
        """
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=OpenAIEmbeddingFunction(
                    api_key=self.openai_api_key,
                    model_name=self.model_name
                )
            )
            return True
        except Exception as e:
            print(f"Error from create_collection: {str(e).strip()}")
            return False
    
    def add_collection(self, ids: list, documents: list):
        """Adds data to collections. First turn data to embedding with OpenAI and adds to vector database.
                Args:
                    ids (list): The ID list. Every text has an ID.
                    documents (list): The text data list.
        """
        
        try:
            self.collection.add(
                ids=ids,
                documents=documents
            )
            return True
        except Exception as e:
            print(f"Error: {str(e).strip()}")
            return False