import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Cargar los datos desde los archivos .npy
X = np.load('TP\mi_dataset_X.npy')
Y = np.load('TP\mi_dataset_Y.npy')

# Normalización de datos (opcional si los datos ya están normalizados)
X = np.array(X)

# Crear un modelo simple con coordenadas 3D
model = models.Sequential([
    layers.Input(shape=(2, 21, 3)),  # Entrada 2 manos, 21 puntos, 3 coordenadas
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')  
])

# Compilar el modelo
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Dividir los datos en entrenamiento y prueba
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Evaluar el modelo
y_pred = np.argmax(model.predict(X_val), axis=1)
print("Reporte de clasificación:")
print(classification_report(y_val, y_pred))
print("Matriz de confusión:")
print(confusion_matrix(y_val, y_pred))

# Guardar el modelo en un archivo .h5
model.save('TP\mi_modelo.h5')
print("Modelo guardado en 'mi_modelo.h5'.")
