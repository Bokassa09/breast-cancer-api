# Breast Cancer Classifier — Semi-Supervised Learning

API de classification de tumeurs mammaires basée sur une approche de Semi-Supervised Learning avec Self-Training.

## Objectif du projet
Ce projet montre comment déployer un modèle ML performant avec très peu de données annotées.
Le modèle utilise seulement 2% de données labellisées et exploite les données non annotées 
grâce à une stratégie de Self-Training. Toutes les prédictions sont enregistrées en base 
de données pour un suivi en temps réel.

## Fonctionnalités
- Classification de tumeurs bénignes / malignes
- Pipeline Semi-Supervised Learning (Self-Training + ElasticNet)
- Stockage des prédictions en base de données (SQLite)
- Statistiques en temps réel (/stats, /historique)
- API REST avec FastAPI
- Conteneurisation avec Docker
- Déploiement sur Hugging Face Spaces

## Stack technique
- Python | Scikit-learn | FastAPI | SQLite | Docker | Hugging Face Spaces

## Résultats
| Modèle | Accuracy |
|---|---|
| Supervisé classique (2% labels) | 89.5% |
| Semi-supervisé (Self-Training) | 95.6% |
| Gain obtenu | +6.1 points |

## Routes API
| Route | Méthode | Description |
|---|---|---|
| / | GET | Message d'accueil |
| /predict | POST | Faire une prédiction |
| /stats | GET | Statistiques globales |
| /historique | GET | 10 dernières prédictions |
| /health | GET | Vérifier que l'API tourne |
| /docs | GET | Documentation interactive |

## API en ligne
🔗 https://bokassa09-breast-cancer-api.hf.space/docs

## Lancer le projet en local
```bash
docker build -t breast-cancer-api .
docker run -p 8000:8000 breast-cancer-api
```

## Structure du projet
```
├── main.py              # API FastAPI
├── database.py          # Gestion SQLite
├── train_and_save.py    # Entraînement Self-Training
├── model.pkl            # Modèle sauvegardé
├── scaler.pkl           # Scaler sauvegardé
├── Dockerfile           # Conteneurisation
└── requirements.txt     # Dépendances
```