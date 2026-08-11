# Plant Disease Detection Backend

This is a simple FastAPI backend for serving a trained Plant Disease Detection model based on MobileNetV2.

## Inference Architecture

```text
Client
   ↓
POST /predict
   ↓
FastAPI
   ↓
Image Validation
   ↓
Image Preprocessing
   ↓
Trained MobileNetV2
   ↓
Prediction
   ↓
JSON Response
```

## Setup & Running Locally

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

2. Run the application:
```bash
python -m uvicorn app.main:app --reload
```

3. Open Swagger UI to test the endpoints:
http://localhost:8000/docs

## API Endpoints

- `GET /health` : Check if the API is running and the model is loaded.
- `GET /model/info` : Get information about the loaded model and its classes.
- `POST /predict` : Upload an image to get a disease prediction. You can test this via the Swagger UI at `/docs`.

## Deployment

A `Dockerfile` is included for easy deployment to services like Render or Hugging Face Spaces. It bundles the FastAPI app along with the trained model artifacts.
