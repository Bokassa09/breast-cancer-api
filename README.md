# Breast Cancer Classifier — Semi-Supervised Learning

API de classification de tumeurs mammaires basée sur une approche de Semi-Supervised Learning avec Self-Training.

## Objectif du projet
Ce projet montre comment entraîner un modèle performant avec très peu de données annotées.

Le modèle utilise seulement 2% de données labellisées et exploite les données non annotées grâce à une stratégie de Self-Training.

## Fonctionnalités
- Classification de tumeurs bénignes / malignes
- Pipeline de Semi-Supervised Learning
- API REST avec FastAPI
- Conteneurisation avec Docker
- Déploiement sur Hugging Face Spaces

## Stack technique
- Python
- Scikit-learn
- FastAPI
- Docker
- Hugging Face Spaces

## Résultats

| Modèle | Accuracy |
|---|---|
| Supervisé classique (2% labels) | 89.5% |
| Semi-supervisé (Self-Training) | 95.6% |
| Gain obtenu | +6.1 points |

## API en ligne
🔗 https://bokassa09-breast-cancer-api.hf.space/docs

## Lancer le projet en local

```bash
docker build -t breast-cancer-api .
docker run -p 8000:8000 breast-cancer-api