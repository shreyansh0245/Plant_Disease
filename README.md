# Plant Disease Detection — FastAPI Backend

A simple, interview-ready FastAPI backend for serving a trained **Plant Disease Detection** model built with TensorFlow/Keras and MobileNetV2 transfer learning.

## Project Overview

This project exposes a trained deep learning model as a REST API. The model classifies tomato leaf images into 8 categories (7 diseases + healthy) using MobileNetV2 pretrained on ImageNet, fine-tuned on the PlantVillage dataset.

**Model Architecture:**
- Base: MobileNetV2 (ImageNet weights, frozen in Phase 1)
- Head: GlobalAveragePooling → Dropout(0.3) → Dense(8, softmax)
- Training: 2-phase transfer learning (Phase 1: head only, Phase 2: top 20 layers fine-tuned)
- Input: 224×224×3 images, preprocessed with MobileNetV2's `preprocess_input`
- Output: 8-class softmax probabilities

**Classes:**
1. Tomato Bacterial Spot
2. Tomato Early Blight
3. Tomato Late Blight
4. Tomato Leaf Mold
5. Tomato Septoria Leaf Spot
6. Tomato Spider Mites
7. Tomato Target Spot
8. Tomato Healthy

## Inference Architecture

```
Client
   ↓
POST /predict (multipart image upload)
   ↓
FastAPI
   ↓
Image Validation (JPEG/PNG, ≤10MB)
   ↓
Resize to 224×224 + MobileNetV2 preprocess_input
   ↓
Trained MobileNetV2 Model
   ↓
Softmax Probabilities → argmax → class label
   ↓
JSON Response (predicted_class, confidence, all_probabilities)
```

## Setup & Running Locally

1. **Clone the repository and create a virtual environment:**
```bash
git clone https://github.com/shreyansh0245/Plant_Disease.git
cd Plant_Disease
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
> Note: TensorFlow is pinned to `>=2.17.0,<2.19.0` to match the serialization format of the trained model. Using a different version may cause Keras model-deserialization compatibility issues.

3. **Run the application:**
```bash
python -m uvicorn app.main:app --reload
```

4. **Open Swagger UI to test the endpoints:**
```
http://localhost:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Welcome message and link to docs |
| `GET` | `/health` | Check if API is running and model is loaded |
| `GET` | `/model/info` | Get model metadata and class names |
| `POST` | `/predict` | Upload a leaf image to get a disease prediction |

### Example `/predict` Response
```json
{
  "predicted_class": "Tomato_Early_blight",
  "confidence": 0.9732,
  "all_probabilities": {
    "Tomato_Bacterial_spot": 0.0021,
    "Tomato_Early_blight": 0.9732,
    "Tomato_Late_blight": 0.0083,
    "Tomato_Leaf_Mold": 0.0012,
    "Tomato_Septoria_leaf_spot": 0.0061,
    "Tomato_Spider_mites_Two_spotted_spider_mite": 0.0045,
    "Tomato__Target_Spot": 0.0031,
    "Tomato_healthy": 0.0015
  }
}
```

## Project Structure

```
Plant_Disease/
├── app/
│   ├── main.py          # FastAPI app setup and startup events
│   ├── routes.py        # API endpoint definitions
│   ├── inference.py     # Model loading and prediction logic
│   ├── config.py        # Path and settings configuration
│   └── tests/           # pytest tests for all endpoints
├── artifacts/
│   └── model/
│       ├── best_model.keras   # Trained MobileNetV2 model
│       ├── labels.json        # Index → class name mapping
│       └── config.json        # Model metadata
├── requirements.txt     # Pinned dependencies
├── Dockerfile           # Container definition for deployment
└── README.md
```

## Deployment

The app is deployed on **Render** at:
**https://plant-disease-xz2k.onrender.com**

A `Dockerfile` is included for deployment to Render, Hugging Face Spaces, or any container platform. It bundles the FastAPI app with the trained model artifacts.

To test the live deployment:
- **Health check:** https://plant-disease-xz2k.onrender.com/health
- **Interactive docs:** https://plant-disease-xz2k.onrender.com/docs
- **Prediction:** `POST` an image to https://plant-disease-xz2k.onrender.com/predict
