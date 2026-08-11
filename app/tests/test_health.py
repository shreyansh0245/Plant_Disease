from fastapi.testclient import TestClient
from app.main import app

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data

def test_model_info():
    with TestClient(app) as client:
        response = client.get("/model/info")
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "classes" in data
