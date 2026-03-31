from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import pandas as pd
import uvicorn
import pickle

# schema
from schemas.inference_schema import Keys

# Paths
_BASE_PATH= Path(__file__).parent.parent

# Models folder path
_MODEL_FOLDER_PATH= _BASE_PATH / "models"

_TEMPLATES_PATH= _BASE_PATH / "src" / "templates"

# Model paths
scaler_path = _MODEL_FOLDER_PATH / "scaler.pkl"
model_path  =   _MODEL_FOLDER_PATH / "model.pkl"

# load models function
def load_models(path: Path):
    """Reads models
            Args:
                path (Path): The Model where will be load.
    """
    with open(path, "rb") as f:
        name = pickle.load(f)
    
    return name

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App is starting...")
    
    # Upload Logistic Regression model and StandardScalar
    app.state.scaler = load_models(scaler_path)
    app.state.model  = load_models(model_path)
    
    yield
    
    
app = FastAPI(
    title="IBM Telco Customer Churn Prediction",
    description="",
    version="0.01",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=_TEMPLATES_PATH)

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse(
        request=request, name='index.html'
    )

@app.get("/healthy")
async def healthy_check():
    """Helthy Check API
    """
    return {
        "status" : True,
        "message": "App is healthy!"
    }

@app.post("/predict")
async def model_predict(data: Keys):
    """The Model predicts
            Args:
                data (dict): Dict for predict.
    """    
    try:
        # Call Logistic Regression model and StandardScaler
        model  = app.state.model
        scaler = app.state.scaler
        
        data   = data.model_dump(by_alias=True)
        new_df = pd.DataFrame([data])
        
        scaled_data = scaler.transform(new_df)
        result = model.predict(scaled_data).tolist()[0]
        result = "Yes" if result == 1 else "No"
        
        return {
            "status"  : True,
            "message" : "Model predicted.",
            "response": result
        }
    except Exception as e:
        print(f"Error from model_predict: {str(e).strip()}")
        raise HTTPException(status_code=500, detail="Model couldn't predict.")

if __name__ == "__main__":
    print("Uvicorn starting now...")
    uvicorn.run("inference:app", host="127.0.0.1", port=342, reload=True)