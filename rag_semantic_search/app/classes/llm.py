from ollama import chat
from pydantic import BaseModel

class LLM:
    def __init__(self):
        self.model: str = "gemma3:1b"
        self.temperature: float = 0.0
        self.system_prompt: str = ""
    
    async def client(self, data: BaseModel):
        """LLM Client function
        """
        
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': data.role, 'content': data.content}
        ]
        
        print(messages)