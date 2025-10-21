import requests
import numpy as np
from tensorflow import keras

# Charger une image de test depuis MNIST
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

# Prendre la première image de test
test_image = x_test[0]
true_label = y_test[0]

print(f"Label réel de l'image: {true_label}")

# Normaliser et aplatir l'image
test_image_normalized = test_image.astype("float32") / 255.0
test_image_flat = test_image_normalized.flatten().tolist()

# Préparer la requête
url = "http://localhost:5000/predict"
payload = {
    "image": test_image_flat
}

# Envoyer la requête POST
try:
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Prédiction réussie!")
        print(f"Classe prédite: {result['prediction']}")
        print(f"\nProbabilités pour chaque classe (0-9):")
        for i, prob in enumerate(result['probabilities'][0]):
            print(f"  Classe {i}: {prob:.4f}")
    else:
        print(f"❌ Erreur: {response.status_code}")
        print(response.json())
        
except requests.exceptions.ConnectionError:
    print("❌ Impossible de se connecter à l'API. Assurez-vous que le conteneur Docker est en cours d'exécution.")
except Exception as e:
    print(f"❌ Erreur: {str(e)}")