import os
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
from typing import Dict, Any, Optional

from src.inference.predictor import SignalScopePredictor

app = FastAPI(
    title="SignalScope API",
    description="AI Media Authenticity Analysis API",
    version="0.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor singleton
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        # Load config
        with open('configs/inference.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        model_path = os.getenv('SIGNALSCOPE_MODEL_PATH', 'model/weights/best_model.pt')
        predictor = SignalScopePredictor(config, model_path)
    return predictor

@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}
    
@app.post("/api/v1/predict")
async def predict_image(
    file: UploadFile = File(...), 
    caption: Optional[str] = None,
    pred: SignalScopePredictor = Depends(get_predictor)
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")
        
    temp_path = f"/tmp/{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = pred.predict(temp_path, caption)
        return result
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
