import os
import yaml
import shutil
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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
        
    import tempfile
    
    # Use a secure temp directory
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = pred.predict(temp_path, caption)
        return result
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Serve the frontend UI
os.makedirs("app/ui", exist_ok=True)

@app.get("/")
async def serve_ui():
    ui_path = "app/ui/index.html"
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return {"message": "UI not found. Please create app/ui/index.html"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
