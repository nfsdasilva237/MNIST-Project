import   tensorflow   as   tf
from   tensorflow   import   keras
import   numpy   as   np


#chargement du jeu de données MINST
(x_train, y_train), (x_test, y_test) = keras.dataset.mnist
load_data()

#Normalisation des données
x_train = x_train.astype("float32")/ 255.0
x_test = x_test.astype("float32")/ 255.0

#redimensionnement des images pour les reseaux fully-connected
x_train = x_train.reshape("60000 , 784")
x_test = x_test.reshape("10000 , 784")

#construction du model
model = keras.sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.DEnse(10, activation='softmax')
])

#Compilation du model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=[accuracy]
)

#entrainement du model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

#validation du model
test_loss , test_acc = model.evaluate (x_test, y_test)
print(f"Precision sur les données de test : { test_acc:.4f}")
#sauvegarde du model
model.save("mnist_model.h5")
print("Model sauvegardé sous mnist.h5")