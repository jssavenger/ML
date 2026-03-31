from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import Literal

class LLMMessagesSchema(BaseModel):
    role: Literal["user"]
    content: str = Field(..., min_length=10, max_length=100)

router = APIRouter(
    prefix="/api/v1",
    tags=['Large Language Model']
)

@router.get("/healthy")
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
        

@router.post("/client")
async def llm_client(data: LLMMessagesSchema, request: Request):
    """LLM Client API
            Args:
                data (BaseModel): LLMMessagesSchema
    """
    
    llm_model = request.app.state.llm_model
    await llm_model.client(data)
    return True
    