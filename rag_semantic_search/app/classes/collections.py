from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from pathlib import Path
import chromadb
import os

_CURRENT_PATH= Path(__file__).parent.parent
_ENV_FILE_PATH= _CURRENT_PATH / ".env"

load_dotenv(dotenv_path=_ENV_FILE_PATH)

_OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

class VectorDB:
    def __init__(self):
        self.client = chroma_client = chromadb.Client()
        self.openai_api_key = _OPENAI_API_KEY
        
    def create_collection(self):
        """Creates ChromaDB vector database collection.
        """
        try:
            self.collection = self.client.create_collection(
                name="my_collection",
                embedding_function=OpenAIEmbeddingFunction(
                    api_key=self.openai_api_key,
                    model_name="text-embedding-3-small"
                )
            )
            return True
        except Exception as e:
            print(f"Error: {str(e).strip()}")
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