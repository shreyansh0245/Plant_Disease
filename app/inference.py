import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from io import BytesIO
import os

from app.config import MODEL_PATH, LABELS_PATH, CONFIG_PATH

# Global variables for caching model and metadata
model = None
labels = {}
model_config = {}

def load_metadata():
    global labels, model_config
    try:
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r") as f:
                labels_raw = json.load(f)
                labels = {int(k): v for k, v in labels_raw.items()}
        
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                model_config = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Could not load metadata files: {e}")

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Could not load model: {e}")

def get_model_info():
    if not labels or not model_config:
        load_metadata()
        
    num_classes = model_config.get("num_classes", len(labels))
    image_size = model_config.get("image_size", 224)
    
    classes_list = [labels.get(i, f"Class {i}") for i in range(num_classes)]
    
    return {
        "model_name": model_config.get("model_name", "MobileNetV2"),
        "model_version": model_config.get("model_version", "1.0"),
        "input_size": [image_size, image_size],
        "num_classes": num_classes,
        "classes": classes_list
    }

def preprocess_image(image_bytes: bytes):
    image_size = model_config.get("image_size", 224)
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize((image_size, image_size))
        img_array = np.array(image, dtype=np.float32)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

def predict(image_bytes: bytes):
    if model is None:
        raise RuntimeError("Model is not loaded.")
        
    img_array = preprocess_image(image_bytes)
    predictions = model.predict(img_array)[0]
    
    # Get top 3 predictions
    top_3_indices = np.argsort(predictions)[-3:][::-1]
    
    predicted_class_idx = top_3_indices[0]
    predicted_class = labels.get(predicted_class_idx, f"Class {predicted_class_idx}")
    confidence = float(predictions[predicted_class_idx])
    
    top_predictions = []
    for idx in top_3_indices:
        top_predictions.append({
            "class_name": labels.get(idx, f"Class {idx}"),
            "confidence": float(predictions[idx])
        })
        
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_predictions": top_predictions
    }
