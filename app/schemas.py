from pydantic import BaseModel
from typing import List

class TopPrediction(BaseModel):
    class_name: str
    confidence: float

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_predictions: List[TopPrediction]

class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    input_size: List[int]
    num_classes: int
    classes: List[str]
