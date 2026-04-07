from pydantic import BaseModel
from pathlib import Path
import yaml

_CURRENT_PATH = Path(__file__).parent.parent
_CONFIG_FILE_PATH = _CURRENT_PATH / "config"
_LLM_SETTINGS_YAML = _CONFIG_FILE_PATH / "llm_settings.yaml"
_PROMPTS_YAML = _CONFIG_FILE_PATH / "prompts.yaml"
_VECTOR_DATABASE_YAML = _CONFIG_FILE_PATH / "vector_database.yaml"

### LLM Settings
class ModelSettings(BaseModel):
    model_name: str
    temperature: float

class LLMSettings(BaseModel):
    llm_settings: ModelSettings
###    
    
### System Prompts
class PromptOptions(BaseModel):
    prompt_one: str

class SystemPrompts(BaseModel):
    system_prompts: PromptOptions
###    
    
### Vector Database Settings
class CollectionSettings(BaseModel):
  model_name: str
  collection_name: str
  max_results: int
  
class VectorDatabaseSettings(BaseModel):
    vector_db_settings: CollectionSettings
###

def read_yaml(file_path: str, model: BaseModel):
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)  
    data = model.model_validate(data)      
    return data

llm_settings = read_yaml(_LLM_SETTINGS_YAML, LLMSettings)
system_prompts = read_yaml(_PROMPTS_YAML, SystemPrompts)
collection_settings = read_yaml(_VECTOR_DATABASE_YAML, VectorDatabaseSettings)