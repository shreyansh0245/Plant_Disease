from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import router
import app.inference as inference

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load metadata and model
    try:
        inference.load_metadata()
        inference.load_model()
        print("Model and metadata loaded successfully.")
    except Exception as e:
        print(f"Failed to load model on startup: {e}")
    
    yield
    
    # Shutdown
    inference.model = None

app = FastAPI(
    title="Plant Disease Detection API",
    description="API for detecting plant diseases using a trained MobileNetV2 model.",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)
