# TP-1 — Instructions d'installation

Ce document explique comment créer et activer un environnement virtuel Python (venv) sous Windows (PowerShell) et comment installer les dépendances listées dans `requirements.txt`.

## Prérequis

- Python 3.x installé et disponible via la commande `python`.
- Un accès en lecture au dossier du projet contenant `TP-1.py` et `requirements.txt`.

## 1) Créer un environnement virtuel

Dans le dossier du projet (`TP-1`), exécutez l'une des commandes suivantes pour créer un venv.

PowerShell (recommandé) :

```powershell
# crée un venv nommé .venv
python -m venv .venv

# (alternative) crée un venv nommé venv
python -m venv venv
```

## 2) Activer l'environnement virtuel (PowerShell)

Après la création, activez l'environnement. Selon le nom choisi pour le dossier du venv utilisez :

```powershell
# si venv s'appelle .venv
.\.venv\Scripts\Activate.ps1

# si venv s'appelle venv
.\venv\Scripts\Activate.ps1
```

Si PowerShell empêche l'exécution du script (erreur d'exécution), vous pouvez autoriser les scripts pour l'utilisateur courant :

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
# puis réactivez :
.\.venv\Scripts\Activate.ps1
```

Note : l'utilisation de `Set-ExecutionPolicy` modifie la stratégie d'exécution pour l'utilisateur courant uniquement. Si vous n'êtes pas à l'aise avec ce changement, vous pouvez exécuter les commandes Python directement sans activer le venv en préfixant la commande par le chemin vers l'exécutable du venv.

## 3) Mettre à jour pip et installer les dépendances

Une fois le venv activé, mettez à jour pip puis installez les paquets depuis `requirements.txt` :

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si `requirements.txt` est vide ou absent :

- vérifiez que le fichier se trouve bien dans le dossier `TP-1` ;
- sinon installez manuellement les paquets nécessaires (par exemple `scikit-learn`, `numpy`, `matplotlib`) :

```powershell
pip install numpy matplotlib scikit-learn
```

## 4) Lancer le script

Depuis le dossier `TP-1` (avec le venv activé) lancez :

```powershell
python TP-1.py
```

Cela affichera les coefficients calculés, les prédictions et ouvrira une fenêtre Matplotlib avec le graphique.

## 5) Nettoyage

Pour désactiver l'environnement virtuel :

```powershell
deactivate
```

---

Si vous voulez que j'ajoute des instructions pour d'autres shells (cmd.exe, Git Bash) ou pour macOS/Linux, dites-le et je les ajoute.
