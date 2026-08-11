from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas import PredictionResponse, ModelInfo
import app.inference as inference

router = APIRouter()

@router.get("/")
def read_root():
    return {"message": "Plant Disease Detection API is running. Go to /docs for the interactive API documentation."}

@router.get("/health")
def health_check():
    model_loaded = inference.model is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded
    }

@router.get("/model/info", response_model=ModelInfo)
def model_info():
    try:
        info = inference.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
        
    if inference.model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
        
    try:
        contents = await file.read()
        result = inference.predict(contents)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
