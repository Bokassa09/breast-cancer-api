from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List
import time
from database import init_db, enregistrer_prediction, get_stats, get_historique

# Charger le modèle et le scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialiser la base de données 
init_db()

app = FastAPI(
    title="Breast Cancer Classifier",
    description="Pipeline Semi-Supervisé avec Self-Training — FastAPI + Docker + SQLite",
    version="2.0"
)

class TumeurData(BaseModel):
    features: List[float]


@app.get("/")
def accueil():
    return {
        "bienvenue": "Breast Cancer Classifier API v2.0",
        "description": "Entrez les 30 mesures d'une tumeur pour obtenir une analyse",
        "routes": {
            "/predict":    "Faire une prédiction",
            "/stats":      "Statistiques globales",
            "/historique": "Les 10 dernières prédictions",
            "/health":     "Vérifier que l'API tourne",
            "/docs":       "Documentation interactive"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "API en ligne et opérationnelle",
        "version": "2.0"
    }


@app.post("/predict")
def predict(data: TumeurData):

    if len(data.features) != 30:
        return {
            "erreur": f"30 mesures attendues, {len(data.features)} reçues",
            "conseil": "Vérifiez que toutes les mesures sont bien renseignées"
        }

    
    debut = time.time()

    X = np.array(data.features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probabilite = model.predict_proba(X_scaled)[0].max()
    pourcentage = round(float(probabilite) * 100, 1)

    temps_ms = round((time.time() - debut) * 1000, 2)

    # Niveau de danger
    if prediction == 1:
        if pourcentage >= 90:
            niveau_danger = "Faible"
            message = "Les mesures indiquent une tumeur probablement bénigne."
            conseil = "Un suivi médical régulier reste recommandé."
            emoji = "🟢"
            diagnostic = "Bénigne"
        else:
            niveau_danger = "Modéré"
            message = "Les mesures semblent indiquer une tumeur bénigne mais avec une certaine incertitude."
            conseil = "Consultez un médecin pour confirmer ce résultat."
            emoji = "🟡"
            diagnostic = "Bénigne"
    else:
        if pourcentage >= 90:
            niveau_danger = "Élevé"
            message = "Les mesures indiquent une tumeur probablement maligne."
            conseil = "Consultez un médecin spécialiste en urgence."
            emoji = "🔴"
            diagnostic = "Maligne"
        else:
            niveau_danger = "Modéré à élevé"
            message = "Les mesures suggèrent une possible tumeur maligne."
            conseil = "Consultez un médecin rapidement pour des examens complémentaires."
            emoji = "🟠"
            diagnostic = "Maligne"

    confiance = "haute" if pourcentage >= 90 else "moyenne" if pourcentage >= 70 else "faible"

    # Enregistrer dans la DB
    enregistrer_prediction(diagnostic, probabilite, confiance, temps_ms)

    return {
        "resultat": f"{emoji} {message}",
        "diagnostic": diagnostic,
        "confiance": f"{pourcentage}%",
        "niveau_danger": niveau_danger,
        "conseil_medical": conseil,
        "temps_reponse_ms": temps_ms,
        "avertissement": "⚠️ Cet outil est une aide à la décision — il ne remplace pas un avis médical professionnel"
    }

# Route stats 
@app.get("/stats")
def stats():
    return get_stats()

# Route hist.
@app.get("/historique")
def historique():
    return get_historique()