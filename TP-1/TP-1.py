"""TP-1.py

Exemple simple de régression linéaire avec scikit-learn.

But du script :
- générer un petit jeu de données d'exemple,
- entraîner un modèle de régression linéaire,
- afficher les paramètres du modèle et faire quelques prédictions,
- tracer les données, la droite de régression et les prédictions.

Entrées / sorties :
- Entrées : données codées en dur dans le script (X : features, y : targets)
- Sorties : affichage des coefficients, prédictions et une fenêtre graphique (matplotlib)

Remarques : ce script est volontairement minimal pour un TP d'initiation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1) Générer des données d'exemple
# - X doit être une matrice 2D de shape (n_samples, n_features) pour scikit-learn
# - on crée donc un tableau 1D puis on reshape en colonne (n_samples, 1)
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
# y est un vecteur 1D des valeurs cibles correspondantes
y = np.array([2, 4, 5, 4, 5, 7])

# 2) Créer le modèle et l'entraîner
# LinearRegression() ajuste une droite y = a * x + b par moindres carrés
model = LinearRegression()
# fit attend X (2D) et y (1D)
model.fit(X, y)

# 3) Afficher les paramètres appris
# - model.coef_ est un tableau (pour chaque feature) ; ici une seule feature
# - model.intercept_ est l'ordonnée à l'origine (b)
print("Coefficient (a):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# 4) Faire des prédictions sur de nouvelles valeurs
# - X_new doit aussi être 2D
X_new = np.array([[7], [8], [9]])
y_pred = model.predict(X_new)
print("\nPrédictions pour X = 7, 8, 9 :", y_pred)

# 5) Visualisation
# - points bleus : données d'entraînement
# - droite rouge : droite de régression ajustée sur les données d'entraînement
# - points verts : prédictions faites pour X_new
plt.scatter(X, y, color='blue', label="Données d'entraînement")
plt.plot(X, model.predict(X), color='red', label="Droite de régression")
plt.scatter(X_new, y_pred, color='green', label="Prédictions")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Exemple de régression linéaire")
plt.legend()
plt.show()