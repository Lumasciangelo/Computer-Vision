# Proyecto de Clasificación de Gestos y Control por Gestos

Este proyecto consta de tres partes principales que permiten capturar, entrenar y utilizar un modelo de clasificación de gestos para interactuar con la computadora a través de acciones como abrir el navegador, bloquear la pantalla o cambiar ventanas.

## Archivos Principales

1. **record-dataset.py**
   - Captura datos de gestos utilizando la cámara en vivo y MediaPipe para detectar manos.
   - Permite etiquetar manualmente los gestos.
   - Guarda las coordenadas procesadas de las manos en un dataset para su entrenamiento.

2. **train-gesture-classifier.py**
   - Entrena un modelo de clasificación basado en las coordenadas de los gestos capturados.
   - Utiliza TensorFlow para crear un modelo denso simple.
   - Guarda el modelo entrenado en formato .h5.

3. **final-prediction.py**
   - Carga el modelo entrenado para predecir gestos en tiempo real.
   - Realiza acciones en la computadora según el gesto detectado (por ejemplo, subir volumen o tomar una selfie).

## Requisitos

- Python 3.7+
- Librerías:
  - OpenCV
  - MediaPipe
  - TensorFlow
  - NumPy
  - Scikit-learn
  - PyAutoGUI

Instala todas las dependencias ejecutando:
```bash
pip install opencv-python mediapipe tensorflow numpy scikit-learn pyautogui
```

## Flujo de Trabajo

### 1. Capturar y Crear el Dataset
Ejecuta `record-dataset.py` para:
- Capturar gestos en tiempo real con la cámara.
- Etiquetar manualmente los gestos con las teclas del 0 al 9.
- Guardar las coordenadas de los gestos en los archivos `mi_dataset_X.npy` y `mi_dataset_Y.npy` dentro de la carpeta `TP`.

### 2. Entrenar el Modelo
Ejecuta `train-gesture-classifier.py` para:
- Entrenar un modelo utilizando el dataset capturado.
- Evaluar el modelo con métricas como reporte de clasificación y matriz de confusión.
- Guardar el modelo entrenado en `TP\mi_modelo.h5`.

### 3. Predicciones y Acciones en Tiempo Real
Ejecuta `final-prediction.py` para:
- Detectar gestos en tiempo real utilizando la cámara.
- Realizar acciones automáticas en la computadora según el gesto detectado.

Cualquier usuario puede probar este modelo en su computadora con tan solo descargar el repositorio, asegurándose de que contenga los archivos `mi_dataset_X.npy`, `mi_dataset_Y.npy` y el modelo `mi_modelo.h5` en el mismo directorio. Luego, ejecutando el script `final-prediction.py`, la aplicación estará funcionando.

#### Mapa de Gestos
Cada gesto está asociado a una acción específica:
| Gesto | Descripción      | Acción             |
|-------|------------------|--------------------|
| 0     | Puño            | Abrir navegador    |
| 1     | Ok               | Minimizar todo     |
| 2     | Paz              | Captura de pantalla |
| 3     | Like             | Subir volumen      |
| 4     | Dislike          | Bajar volumen      |
| 5     | Cuerno           | Abrir explorador   |
| 6     | Suerte           | Bloquear pantalla  |
| 7     | Mano abierta     | Cambiar ventana    |
| 8     | Pistola          | Cerrar ventana     |
| 9     | Tres dedos arriba| Tomar selfie       |

## Notas Adicionales
- **Guardar capturas de pantalla y selfies:** Las capturas de pantalla se guardan como `captura_pantalla.png` y las selfies como `selfie.png` en el directorio actual.
- **Gestos personalizados:** Puedes agregar nuevos gestos editando los mapas en `record-dataset.py` y `final-prediction.py`.

## Contribución
Si tienes sugerencias o mejoras, siéntete libre de realizar un fork de este repositorio y enviar un pull request.
