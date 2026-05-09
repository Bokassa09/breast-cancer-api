# Les packages necessaire 
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib



# Charger les données
data = load_breast_cancer()
X = data.data   # les 30 mesures médicales
y = data.target # 0 = maligne, 1 = bénigne

print(f"Nombre d'exemples : {X.shape[0]}")
print(f"Nombre de features : {X.shape[1]}")
print(f"Labels possibles : {data.target_names}")
print(f"Malignes : {sum(y == 0)} | Bénignes : {sum(y == 1)}")


# Simulation du manque de labels 

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rng = np.random.RandomState(42)
mask_unlabeled = rng.rand(len(X_train)) < 0.98  # 2% étiquetés

X_labeled   = X_train[~mask_unlabeled]
y_labeled   = y_train[~mask_unlabeled]
X_unlabeled = X_train[mask_unlabeled]

print(f"Train total     : {len(X_train)}")
print(f"Étiquetés (2%) : {len(X_labeled)}")
print(f"Non étiquetés   : {len(X_unlabeled)}")

# Modèle supervisé sur 2% seulement
clf_supervised = LogisticRegression(max_iter=1000, random_state=42)
clf_supervised.fit(X_labeled, y_labeled)

acc_supervised = accuracy_score(y_test, clf_supervised.predict(X_test))

print(f"\n[SUPERVISÉ — 2% labels]")
print(f"Accuracy : {acc_supervised:.4f} ({acc_supervised*100:.1f}%)")
print(f"\nComparaison :")
print(f"  Avec 100% des labels : 98.2%")
print(f"  Avec 2% des labels  : {acc_supervised*100:.1f}%")
print(f"  Perte                : {(0.982 - acc_supervised)*100:.1f} points")


# Self-Training
SEUIL = 0.90    # on accepte uniquement les prédictions > 90% de confiance
MAX_ITER = 20

# On repart des données étiquetées initiales
X_lab = X_labeled.copy()
y_lab = y_labeled.copy()
X_unlab = X_unlabeled.copy()

print(f"Départ : {len(X_lab)} exemples étiquetés, {len(X_unlab)} non étiquetés")
print(f"{'─'*55}")

for i in range(MAX_ITER):

    # Entraîner sur ce qu'on a
    clf = LogisticRegression(max_iter=1000, random_state=42, penalty='elasticnet',
    solver='saga',
    l1_ratio=0.5)
    
    clf.fit(X_lab, y_lab)

    # S'arrêter s'il ne reste rien
    if len(X_unlab) == 0:
        print(f"  Itération {i+1} : plus rien à étiqueter, arrêt.")
        break

    # Prédire les probabilités sur les non étiquetés
    probas = clf.predict_proba(X_unlab)
    confiance = probas.max(axis=1)      # meilleure proba pour chaque exemple
    pseudo_labels = probas.argmax(axis=1)  # label choisi

    # Garder seulement les exemples assez confiants
    idx_confiants = np.where(confiance >= SEUIL)[0]

    if len(idx_confiants) == 0:
        print(f"  Itération {i+1} : aucun exemple assez confiant, arrêt.")
        break

    # Les ajouter au dataset étiqueté
    X_lab = np.vstack([X_lab, X_unlab[idx_confiants]])
    y_lab = np.concatenate([y_lab, pseudo_labels[idx_confiants]])

    # Les retirer du pool non étiqueté
    X_unlab = np.delete(X_unlab, idx_confiants, axis=0)

    # Mesurer l'accuracy à cette itération
    acc_iter = accuracy_score(y_test, clf.predict(X_test))
    print(f"  Itération {i+1:2d} | "
          f"Ajoutés : {len(idx_confiants):3d} | "
          f"Étiquetés : {len(X_lab):3d} | "
          f"Accuracy : {acc_iter:.4f}")

# Résultat final
acc_semi = accuracy_score(y_test, clf.predict(X_test))
gain = (acc_semi - acc_supervised) * 100

print(f"\n{'='*55}")
print(f"  Supervisé  (2% labels)     : {acc_supervised*100:.1f}%")
print(f"  Semi-supervisé (self-train): {acc_semi*100:.1f}%")
print(f"  Gain                       : {gain:+.1f} points")
print(f"{'='*55}")


# Sauvegarder le modèle final (
joblib.dump(clf, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModèle self-training sauvegardé : model.pkl")
print("Scaler sauvegardé : scaler.pkl")