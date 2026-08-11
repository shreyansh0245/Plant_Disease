from fastapi.testclient import TestClient
from app.main import app
import io
from PIL import Image

def test_predict_invalid_file():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400
        assert "not an image" in response.json()["detail"].lower()

def test_predict_valid_image():
    with TestClient(app) as client:
        health = client.get("/health").json()
        if not health.get("model_loaded"):
            # Model not available, test should expect 503
            img_byte_arr = io.BytesIO()
            image = Image.new('RGB', (224, 224), color = 'green')
            image.save(img_byte_arr, format='JPEG')
            
            response = client.post(
                "/predict",
                files={"file": ("test.jpg", img_byte_arr.getvalue(), "image/jpeg")}
            )
            assert response.status_code == 503
            return

        # Model is loaded, proceed with a real inference
        img_byte_arr = io.BytesIO()
        image = Image.new('RGB', (224, 224), color = 'green')
        image.save(img_byte_arr, format='JPEG')
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_byte_arr.getvalue(), "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "top_predictions" in data
