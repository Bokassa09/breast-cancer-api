from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(
    title="Breast Cancer Classifier",
    description="Outil d'aide à la détection de tumeurs — Ne remplace pas un avis médical",
    version="1.0"
)

class TumeurData(BaseModel):
    features: List[float]

# Route accueil
@app.get("/")
def accueil():
    return {
        "bienvenue": "Breast Cancer Classifier API",
        "description": "Entrez les 30 mesures d'une tumeur pour obtenir une analyse",
        "routes": {
            "/predict": "Faire une prédiction",
            "/health": "Vérifier que l'API tourne",
            "/docs": "Documentation interactive"
        }
    }

# Route health 
@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "API en ligne et opérationnelle"
    }

# Route predict
@app.post("/predict")
def predict(data: TumeurData):

    if len(data.features) != 30:
        return {
            "erreur": f"30 mesures attendues, {len(data.features)} reçues",
            "conseil": "Vérifiez que toutes les mesures sont bien renseignées"
        }

    X = np.array(data.features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probabilite = model.predict_proba(X_scaled)[0].max()
    pourcentage = round(float(probabilite) * 100, 1)

    # Résultat selon la prédiction 
    if prediction == 1:  # bénigne
        if pourcentage >= 90:
            niveau_danger = "Faible"
            message = "Les mesures indiquent une tumeur probablement bénigne."
            conseil = "Un suivi médical régulier reste recommandé."
            emoji = "🟢"
        else:
            niveau_danger = "Modéré"
            message = "Les mesures semblent indiquer une tumeur bénigne mais avec une certaine incertitude."
            conseil = "Consultez un médecin pour confirmer ce résultat."
            emoji = "🟡"
    else:  # maligne
        if pourcentage >= 90:
            niveau_danger = "Élevé"
            message = "Les mesures indiquent une tumeur probablement maligne."
            conseil = "Consultez un médecin spécialiste en urgence."
            emoji = "🔴"
        else:
            niveau_danger = "Modéré à élevé"
            message = "Les mesures suggèrent une possible tumeur maligne."
            conseil = "Consultez un médecin rapidement pour des examens complémentaires."
            emoji = "🟠"

    return {
        "resultat": f"{emoji} {message}",
        "diagnostic": "Bénigne" if prediction == 1 else "Maligne",
        "confiance": f"{pourcentage}%",
        "niveau_danger": niveau_danger,
        "conseil_medical": conseil,
        "avertissement": "⚠️ Cet outil est une aide à la décision — il ne remplace pas un avis médical professionnel"
    }