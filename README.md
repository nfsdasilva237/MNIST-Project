# 🧠 Projet Deep Learning : Classification MNIST avec API Flask

Projet de Travaux Pratiques - Deep Learning Engineering  
**Département Génie Informatique, ENSPY**  
**Date**: Septembre 2025

---

## 📋 Table des matières

- [Description](#description)
- [Architecture du projet](#architecture-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [API Endpoints](#api-endpoints)
- [Docker](#docker)
- [MLflow](#mlflow)
- [Résultats](#résultats)
- [Auteur](#auteur)

---

## 📖 Description

Ce projet implémente un pipeline complet de Deep Learning, de la conception au déploiement, pour la classification des chiffres manuscrits du dataset MNIST.

**Objectifs pédagogiques** :
- Construction et entraînement d'un réseau de neurones avec TensorFlow/Keras
- Suivi des expérimentations avec MLflow
- Création d'une API REST avec Flask
- Conteneurisation avec Docker
- Versionnement avec Git/GitHub

---

## 🏗️ Architecture du projet

```
DEEP/
├── venv/                      # Environnement virtuel Python
├── train_model.py             # Script d'entraînement du modèle
├── mnist_model.h5             # Modèle entraîné (sauvegardé)
├── app.py                     # API Flask pour les prédictions
├── test_api.py                # Script de test de l'API
├── requirements.txt           # Dépendances Python
├── Dockerfile                 # Configuration Docker
├── .dockerignore              # Fichiers à ignorer par Docker
├── mlruns/                    # Logs et artifacts MLflow
└── README.md                  # Documentation (ce fichier)
```

---

## ✅ Prérequis

- Python 3.9 ou supérieur
- pip (gestionnaire de paquets Python)
- Git
- Docker (pour la conteneurisation)
- 4 GB RAM minimum
- 2 GB d'espace disque

---

## 🚀 Installation

### 1. Cloner le repository

```bash
git clone <URL_de_votre_repo>
cd DEEP
```

### 2. Créer et activer l'environnement virtuel

**Windows (PowerShell)** :
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac** :
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 💻 Utilisation

### Étape 1 : Entraîner le modèle

```bash
python train_model.py
```

**Sortie attendue** :
- Un fichier `mnist_model.h5` contenant le modèle entraîné
- Précision sur les données de test : ~97-98%
- Logs MLflow dans le dossier `mlruns/`

### Étape 2 : Lancer l'API Flask

```bash
python app.py
```

L'API sera accessible à : `http://localhost:5000`

### Étape 3 : Tester l'API

Dans un autre terminal :

```bash
python test_api.py
```

---

## 🌐 API Endpoints

### 1. Page d'accueil
```http
GET /
```

**Réponse** :
```json
{
  "message": "API de prédiction MNIST",
  "status": "running",
  "endpoints": { ... }
}
```

### 2. Health Check
```http
GET /health
```

**Réponse** :
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3. Prédiction
```http
POST /predict
Content-Type: application/json
```

**Corps de la requête** :
```json
{
  "image": [0.0, 0.1, ..., 0.0]  // 784 valeurs entre 0 et 1
}
```

**Réponse** :
```json
{
  "prediction": 7,
  "confidence": 0.9823,
  "probabilities": [0.001, 0.002, ..., 0.982, ...]
}
```

### Exemple avec curl

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": [0.0, 0.0, ...]}'
```

---

## 🐋 Docker

### Construction de l'image

```bash
docker build -t mnist-api:latest .
```

### Lancement du conteneur

```bash
docker run -d -p 5000:5000 --name mnist-container mnist-api:latest
```

### Vérification

```bash
# Voir les logs
docker logs mnist-container

# Tester l'API
curl http://localhost:5000/health
```

### Arrêt et nettoyage

```bash
# Arrêter le conteneur
docker stop mnist-container

# Supprimer le conteneur
docker rm mnist-container

# Supprimer l'image
docker rmi mnist-api:latest
```

---

## 📊 MLflow

### Lancer l'interface MLflow

```bash
mlflow ui
```

Accédez à : `http://localhost:5000` (ou le port affiché)

### Informations trackées

- **Paramètres** : epochs, batch_size, dropout_rate
- **Métriques** : test_accuracy
- **Artifacts** : modèle complet

---

## 📈 Résultats

### Performance du modèle

| Métrique | Valeur |
|----------|--------|
| Précision (test) | ~97-98% |
| Temps d'entraînement | ~2-3 min (CPU) |
| Taille du modèle | ~5 MB |
| Nombre de paramètres | ~400K |

### Architecture du réseau

```
Input Layer:  784 neurones (28x28 pixels)
Hidden Layer: 512 neurones (ReLU + Dropout 0.2)
Output Layer: 10 neurones (Softmax)
```

### Hyperparamètres

- **Optimiseur** : Adam
- **Loss** : Sparse Categorical Crossentropy
- **Epochs** : 5
- **Batch Size** : 128
- **Dropout Rate** : 0.2

---

## 🛠️ Technologies utilisées

- **TensorFlow/Keras** : Construction et entraînement du modèle
- **Flask** : API REST
- **MLflow** : Suivi des expérimentations
- **Docker** : Conteneurisation
- **NumPy** : Manipulation de données
- **Git/GitHub** : Versionnement

---

## 📝 Réponses aux questions du TP

### Partie 1 - Question 1 : Couches Dense et Dropout

**Dense** : Couche fully-connected où chaque neurone est connecté à tous les neurones de la couche précédente. Apprend les relations complexes entre features.

**Dropout** : Technique de régularisation qui désactive aléatoirement des neurones pendant l'entraînement (ici 20%). Prévient le surapprentissage (overfitting).

**Softmax** : Fonction d'activation utilisée en sortie pour la classification multi-classes. Convertit les scores en probabilités qui somment à 1. Permet d'interpréter la sortie comme une distribution de probabilité sur les classes.

### Partie 1 - Question 2 : Optimiseur Adam

**Adam** (Adaptive Moment Estimation) améliore SGD en :
1. **Momentum adaptatif** : Accumule les gradients passés pour accélérer la convergence
2. **Learning rate adaptatif** : Ajuste automatiquement le learning rate pour chaque paramètre
3. **Combinaison** : Combine les avantages de RMSprop et Momentum

**Avantages** :
- Convergence plus rapide
- Moins sensible au choix du learning rate initial
- Meilleure performance sur de nombreux problèmes

### Partie 1 - Question 3 : Vectorisation et calculs par lots

**Vectorisation** : 
- Les images 28×28 sont transformées en vecteurs de 784 valeurs (`reshape(60000, 784)`)
- Permet d'utiliser des opérations matricielles optimisées

**Calculs par lots (batching)** :
- `batch_size=128` : Traite 128 images simultanément
- Exploite le parallélisme des GPU/CPU
- Équilibre entre vitesse et précision du gradient

---

## 🚀 Améliorations futures

- [ ] Ajout de data augmentation
- [ ] Utilisation d'un réseau convolutionnel (CNN)
- [ ] Déploiement sur le cloud (AWS, GCP, Azure)
- [ ] Interface web interactive
- [ ] Tests unitaires complets
- [ ] CI/CD avec GitHub Actions

---

## 📚 Ressources

- [Documentation TensorFlow](https://www.tensorflow.org/)
- [Documentation Flask](https://flask.palletsprojects.com/)
- [Documentation MLflow](https://mlflow.org/)
- [Documentation Docker](https://docs.docker.com/)
- [Dataset MNIST](http://yann.lecun.com/exdb/mnist/)

---

## 👨‍💻 Auteur

**[Votre Nom]**  
Étudiant en Génie Informatique  
École Nationale Supérieure Polytechnique de Yaoundé (ENSPY)  

**Contact** : [votre.email@example.com]  
**GitHub** : [github.com/votre-username](https://github.com/votre-username)

---

## 📄 Licence

Ce projet est réalisé dans un cadre pédagogique.

---

## 🙏 Remerciements

- Prof. Louis Fippo Fitime, Claude Tinku, Kerolle Sonfack
- Département Génie Informatique, ENSPY
- Communauté TensorFlow et Keras

---

**Date de création** : Septembre 2025  
**Dernière mise à jour** : [Date actuelle]