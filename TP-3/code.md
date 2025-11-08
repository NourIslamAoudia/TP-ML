# Détection d'Intrusions avec Réseaux de Neurones Artificiels (ANN)

## 📋 Vue d'ensemble

Ce projet implémente un système de détection d'intrusions utilisant des réseaux de neurones artificiels (ANN) avec TensorFlow/Keras. Il compare deux architectures contrastées (shallow vs deep) sur le dataset NSL-KDD.

---

## 🎯 Objectifs

- Détecter les attaques réseau (DoS, Probing, R2L, U2R)
- Comparer les performances entre architectures simples et profondes
- Analyser l'impact des hyperparamètres sur l'apprentissage

---

## 📦 Dépendances

```bash
pip install tensorflow pandas scikit-learn matplotlib numpy
```

---

## 🔍 Explication du Code Ligne par Ligne

### **Section 0 : Imports et Reproductibilité**

```python
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
```
- **Imports des bibliothèques** : pandas pour la manipulation des données, scikit-learn pour le preprocessing, TensorFlow pour les réseaux de neurones, matplotlib pour la visualisation.

```python
try:
    keras = tf.keras
    layers = tf.keras.layers
except Exception:
    import keras as _keras
    keras = _keras
    layers = _keras.layers
```
- **Gestion de compatibilité** : Utilise `tf.keras` en priorité, sinon bascule vers Keras standalone pour éviter les erreurs d'import.

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```
- **Reproductibilité** : Fixe les graines aléatoires pour que les résultats soient identiques à chaque exécution.

---

### **Section 1 : Chargement des Données**

```python
data_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
df = pd.read_csv(data_url, header=None)
```
- **Téléchargement** : Charge le dataset NSL-KDD (données d'entraînement) depuis GitHub.
- `header=None` : Le fichier n'a pas de ligne d'en-tête.

```python
cols = ['duration', 'protocol_type', 'service', ...]
df.columns = cols
```
- **Attribution des noms de colonnes** : Les 43 colonnes représentent les caractéristiques des connexions réseau (durée, protocole, service, etc.).

```python
print("Shape:", df.shape)
print(df.head())
```
- **Inspection** : Affiche la forme du dataset (nombre de lignes × colonnes) et les 5 premières lignes.

---

### **Section 2 : Exploration des Données**

```python
print(df.info())
print("\nLabel distribution:\n", df['label'].value_counts())
```
- `df.info()` : Affiche le type de chaque colonne et les valeurs manquantes.
- `value_counts()` : Compte le nombre d'échantillons "normal" vs "attaques".

---

### **Section 3 : Prétraitement**

```python
df = df.drop(columns=['difficulty'])
```
- **Suppression** : La colonne `difficulty` n'est pas utilisée pour l'entraînement.

```python
X = df.drop(columns=['label'])
y = df['label'].apply(lambda s: 0 if s == 'normal' else 1)
```
- **Séparation X/y** : 
  - `X` contient toutes les features (caractéristiques).
  - `y` contient les labels encodés en binaire (0 = normal, 1 = attaque).

```python
X = pd.get_dummies(X, columns=['protocol_type', 'service', 'flag'])
```
- **Encodage One-Hot** : Transforme les variables catégorielles en colonnes binaires.
  - Exemple : `protocol_type = 'tcp'` devient `protocol_type_tcp = 1`.
  - Fait passer le nombre de features de 41 à ~122.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
- **Normalisation** : Centre les données (moyenne = 0) et réduit leur échelle (écart-type = 1).
- Essentiel pour les réseaux de neurones (convergence plus rapide).

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.values, test_size=0.2, random_state=SEED, stratify=y.values
)
```
- **Division train/test** : 80% entraînement, 20% test.
- `stratify` : Conserve la proportion de classes dans chaque ensemble.

---

### **Section 4 : Construction du Modèle**

```python
def build_model(input_dim, n_hidden_layers=1, n_neurons=32, learning_rate=0.001, dropout_rate=0.0):
```
- **Fonction générique** : Crée des architectures personnalisables.

```python
model = keras.Sequential()
model.add(layers.Input(shape=(input_dim,)))
```
- **Modèle séquentiel** : Les couches sont empilées linéairement.
- `Input` : Spécifie la taille d'entrée (122 features).

```python
for i in range(n_hidden_layers):
    model.add(layers.Dense(n_neurons, activation='relu'))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
```
- **Couches cachées** : Boucle pour ajouter `n_hidden_layers` couches.
  - `Dense` : Couche fully-connected avec `n_neurons` neurones.
  - `relu` : Fonction d'activation (Rectified Linear Unit).
  - `Dropout` : Désactive aléatoirement `dropout_rate` % des neurones (régularisation).

```python
model.add(layers.Dense(1, activation='sigmoid'))
```
- **Couche de sortie** : 1 neurone avec activation `sigmoid` (sortie entre 0 et 1).
- Interprété comme la probabilité d'attaque.

```python
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
```
- **Compilation** :
  - `Adam` : Optimiseur adaptatif (ajuste automatiquement le taux d'apprentissage).
  - `binary_crossentropy` : Fonction de perte pour classification binaire.
  - `metrics=['accuracy']` : Suit la précision pendant l'entraînement.

---

### **Section 5 : Définition des Architectures**

```python
m1_params = {
    'n_hidden_layers': 1,
    'n_neurons': 4,
    'learning_rate': 0.05,
    'dropout_rate': 0.0,
    'batch_size': 512,
    'epochs': 15
}
```
- **Modèle Shallow (simple)** :
  - 1 seule couche cachée avec seulement 4 neurones.
  - Taux d'apprentissage élevé (0.05).
  - Batch size large (512).
  - **Problème** : Goulot d'étranglement (122 features → 4 neurones).

```python
m2_params = {
    'n_hidden_layers': 3,
    'n_neurons': 32,
    'learning_rate': 0.001,
    'dropout_rate': 0.2,
    'batch_size': 64,
    'epochs': 15
}
```
- **Modèle Deep (profond)** :
  - 3 couches cachées avec 32 neurones chacune.
  - Taux d'apprentissage faible (0.001).
  - Dropout de 20% (évite le surapprentissage).
  - Batch size modéré (64).

---

### **Section 6 : Construction et Résumés**

```python
model_shallow = build_model(input_dim, **{k:v for k,v in m1_params.items() if k in [...]})
model_deep = build_model(input_dim, **{k:v for k,v in m2_params.items() if k in [...]})
```
- **Filtrage des paramètres** : Extrait uniquement les paramètres nécessaires pour `build_model`.

```python
model_shallow.summary()
model_deep.summary()
```
- **Affichage de l'architecture** : Montre le nombre de paramètres entraînables.

---

### **Section 7 : Entraînement**

```python
history_shallow = model_shallow.fit(X_train, y_train,
                                   validation_split=0.2,
                                   epochs=m1_params['epochs'],
                                   batch_size=m1_params['batch_size'],
                                   verbose=1)
```
- `fit()` : Lance l'entraînement.
- `validation_split=0.2` : Utilise 20% des données d'entraînement pour la validation.
- `epochs` : Nombre de passages complets sur les données.
- `batch_size` : Nombre d'échantillons traités avant la mise à jour des poids.
- `verbose=1` : Affiche la progression.

---

### **Section 8 : Évaluation**

```python
test_loss_sh, test_acc_sh = model_shallow.evaluate(X_test, y_test, verbose=0)
test_loss_de, test_acc_de = model_deep.evaluate(X_test, y_test, verbose=0)
```
- **Test** : Calcule la perte et la précision sur les données de test (non vues pendant l'entraînement).

```python
print(f"Shallow test acc: {test_acc_sh:.4f} | Deep test acc: {test_acc_de:.4f} | Diff: {test_acc_de - test_acc_sh:.4f}")
```
- **Comparaison** : Affiche la différence de performance entre les deux modèles.

---

### **Section 9 : Visualisation**

```python
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history_shallow.history['accuracy'], label='Shallow train acc')
plt.plot(history_deep.history['accuracy'], label='Deep train acc')
```
- **Courbes d'entraînement** : Visualise l'évolution de la précision au fil des epochs.

```python
plt.subplot(1,2,2)
plt.plot(history_shallow.history['val_accuracy'], label='Shallow val acc')
plt.plot(history_deep.history['val_accuracy'], label='Deep val acc')
```
- **Courbes de validation** : Détecte le surapprentissage (si train >> val).

```python
plt.figure(figsize=(6,4))
accs = [test_acc_sh, test_acc_de]
names = ['Shallow', 'Deep']
bars = plt.bar(names, accs)
for bar,acc in zip(bars,accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{acc:.4f}", ha='center', va='bottom')
```
- **Graphique à barres** : Compare visuellement les précisions de test avec annotations.

---

## 🎓 Concepts Clés

### **Pourquoi le modèle shallow performe mal ?**
1. **Goulot d'étranglement** : 122 features → 4 neurones = perte d'information massive.
2. **Capacité d'apprentissage limitée** : Trop peu de paramètres pour capturer la complexité.
3. **Hyperparamètres inadaptés** : Learning rate trop élevé, batch size trop grand.

### **Pourquoi le modèle deep performe mieux ?**
1. **Plus de capacité** : 32 neurones par couche.
2. **Apprentissage hiérarchique** : 3 couches permettent d'extraire des features à différents niveaux d'abstraction.
3. **Régularisation** : Dropout évite le surapprentissage.
4. **Meilleurs hyperparamètres** : Learning rate faible + batch size modéré = convergence stable.

---

## 📊 Résultats Attendus

- **Shallow Network** : ~75-85% de précision
- **Deep Network** : ~90-95% de précision
- **Amélioration** : +10-15 points de pourcentage

---

## 🚀 Utilisation

1. **Exécuter le code complet** :
```bash
python script.py
```

2. **Analyser les sorties** :
   - Formes des datasets
   - Distribution des labels
   - Précisions de test
   - Graphiques de performance

---

## 📝 Notes Importantes

- **Reproductibilité** : Le SEED=42 garantit des résultats identiques.
- **Normalisation** : Essentielle pour la convergence des réseaux de neurones.
- **Encodage One-Hot** : Transforme les catégories en features numériques exploitables.
- **Validation Split** : Permet de surveiller le surapprentissage pendant l'entraînement.

---

## 🔬 Expérimentations Suggérées

1. **Dégrader le shallow** : 2 neurones au lieu de 4 → performance encore pire.
2. **Améliorer le shallow** : 64 neurones + meilleurs hyperparamètres → se rapproche du deep.
3. **Retirer le dropout** : Observer l'impact sur le surapprentissage.

---

## 📚 Références

- **Dataset** : NSL-KDD (Network Security Laboratory - Knowledge Discovery in Databases)
- **Framework** : TensorFlow/Keras
- **Préprocessing** : Scikit-learn

---

## 👤 Auteur

Projet académique - TP3 Machine Learning, Deep Learning et Sécurité  
4ème Année Ingénierie Sécurité, USTHB