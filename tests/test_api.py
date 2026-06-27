# Tests automatiques de l'API
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# vérifier que l'API tourne
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

# vérifier le message d'accueil
def test_accueil():
    response = client.get("/")
    assert response.status_code == 200
    assert "bienvenue" in response.json()

# prédiction maligne
def test_predict_maligne():
    response = client.post("/predict", json={
        "features": [17.99, 10.38, 122.8, 1001.0, 0.1184,
                     0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                     1.095, 0.9053, 8.589, 153.4, 0.006399,
                     0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                     25.38, 17.33, 184.6, 2019.0, 0.1622,
                     0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
    })
    assert response.status_code == 200
    assert response.json()["diagnostic"] == "Maligne"

# prédiction bénigne
def test_predict_benigne():
    response = client.post("/predict", json={
        "features": [13.54, 14.36, 87.46, 566.3, 0.09779,
                     0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
                     0.2699, 0.7886, 2.058, 23.56, 0.008462,
                     0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
                     15.11, 19.26, 99.7, 711.2, 0.144,
                     0.1773, 0.239, 0.1288, 0.2977, 0.07259]
    })
    assert response.status_code == 200
    assert response.json()["diagnostic"] == "Bénigne"

# mauvais nombre de features
def test_predict_mauvais_input():
    response = client.post("/predict", json={
        "features": [1.0, 2.0, 3.0]
    })
    assert response.status_code == 200
    assert "erreur" in response.json()