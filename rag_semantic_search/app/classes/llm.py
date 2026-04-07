from ollama import chat
from pydantic import BaseModel

class LLM:
    def __init__(self, llm_settings, prompt_settings):
        
        # Settings
        self.settings = llm_settings.llm_settings
        self.prompt_settings = prompt_settings.system_prompts
        
        # Variables
        self.model = self.settings.model_name
        self.temperature = self.settings.temperature
        self.system_prompt_one = self.prompt_settings.prompt_one
    
    async def create_messages_schema(self, data: BaseModel):
        """Creates Messages Schema for LLM
                Args:
                    data (BaseModel): role and content
        """
        messages = [
            {'role': 'system', 'content': self.system_prompt_one},
            {'role': data.role, 'content': data.content}
        ]
        
        return messages
    
    async def client(self, messages: list):
        """LLM Client function
                Args:
                    messages (list): Messages list for LLM.
        """
        response = chat(
            model=self.model,
            messages=messages,
            options={
                'temperature': self.temperature
            }
        )

        r = response['message']['content']
        return r