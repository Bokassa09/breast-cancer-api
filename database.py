import sqlite3
from datetime import datetime

DB_PATH = "predictions.db"

def init_db():
    """Créer les tables si elles n'existent pas"""
    with sqlite3.connect(DB_PATH) as connexion:
        curseur = connexion.cursor()

        # toutes les prédictions
        curseur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                date         TEXT NOT NULL,
                diagnostic   TEXT NOT NULL,
                probabilite  REAL NOT NULL,
                confiance    TEXT NOT NULL,
                temps_ms     REAL NOT NULL
            )
        """)

        # statistiques journalières
        curseur.execute("""
            CREATE TABLE IF NOT EXISTS stats_journalieres (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                date             TEXT NOT NULL,
                total            INTEGER NOT NULL,
                total_malignes   INTEGER NOT NULL,
                total_benignes   INTEGER NOT NULL,
                prob_moyenne     REAL NOT NULL
            )
        """)

        connexion.commit()
        print("Base de données initialisée !")

def enregistrer_prediction(diagnostic, probabilite, confiance, temps_ms):
    """Enregistrer une prédiction dans la base"""
    with sqlite3.connect(DB_PATH) as connexion:
        curseur = connexion.cursor()
        curseur.execute("""
            INSERT INTO predictions (date, diagnostic, probabilite, confiance, temps_ms)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            diagnostic,
            probabilite,
            confiance,
            temps_ms
        ))
        connexion.commit()

def get_stats():
    """Récupérer les statistiques globales"""
    with sqlite3.connect(DB_PATH) as connexion:
        curseur = connexion.cursor()

        # T des prédictions
        curseur.execute("SELECT COUNT(*) FROM predictions")
        total = curseur.fetchone()[0]

        # T malignes
        curseur.execute("""
            SELECT COUNT(*) FROM predictions
            WHERE diagnostic = 'Maligne'
        """)
        total_malignes = curseur.fetchone()[0]

        # T bénignes
        curseur.execute("""
            SELECT COUNT(*) FROM predictions
            WHERE diagnostic = 'Bénigne'
        """)
        total_benignes = curseur.fetchone()[0]

        # Proba moyenne
        curseur.execute("SELECT ROUND(AVG(probabilite), 4) FROM predictions")
        prob_moyenne = curseur.fetchone()[0] or 0

        # Time de réponse moyen
        curseur.execute("SELECT ROUND(AVG(temps_ms), 2) FROM predictions")
        temps_moyen = curseur.fetchone()[0] or 0

        return {
            "total_predictions": total,
            "total_malignes": total_malignes,
            "total_benignes": total_benignes,
            "probabilite_moyenne": prob_moyenne,
            "temps_reponse_moyen_ms": temps_moyen
        }

def get_historique(limite=10):
    """Récupérer les dernières prédictions"""
    with sqlite3.connect(DB_PATH) as connexion:
        curseur = connexion.cursor()
        curseur.execute("""
            SELECT date, diagnostic, probabilite, confiance, temps_ms
            FROM predictions
            ORDER BY date DESC
            LIMIT ?
        """, (limite,))
        lignes = curseur.fetchall()
        return [
            {
                "date": l[0],
                "diagnostic": l[1],
                "probabilite": l[2],
                "confiance": l[3],
                "temps_ms": l[4]
            }
            for l in lignes
        ]