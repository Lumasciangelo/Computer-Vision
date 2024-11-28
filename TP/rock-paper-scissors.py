import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np

# Cargar el modelo entrenado
model = tf.keras.models.load_model('mi_modelo.h5')

# Inicializar MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inicializar captura de video (cámara en vivo)
captura = cv2.VideoCapture(0)  # 0 es la cámara por defecto

if not captura.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

# Bucle para capturar video en vivo
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        # Leer un frame
        ret, frame = captura.read()

        # Si no se puede leer el frame, salir del bucle
        if not ret:
            print("No se puede obtener el frame.")
            break

        # Convertir BGR a RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar el frame con MediaPipe
        results = hands.process(image_rgb)

        # Lista para almacenar las coordenadas de los 21 puntos
        puntos_mano = []

        # Si se detectan manos, extraer puntos clave
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    puntos_mano.append((lm.x, lm.y, lm.z))  # Incluye coordenadas 3D
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Realizar la predicción si hay puntos clave
        if puntos_mano:
            input_data = np.array([puntos_mano])  # Agregar dimensión
            predictions = model.predict(input_data)
            predicted_label = np.argmax(predictions)

            # Mapeo de etiquetas a gestos
            if predicted_label == 0:
                texto = 'Piedra'
            elif predicted_label == 1:
                texto = 'Papel'
            elif predicted_label == 2:
                texto = 'Tijera'
        else:
            texto = 'No se detectaron manos'

        # Mostrar la predicción en pantalla
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Video en Vivo", frame)

        # Salir con 'q'
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break

cv2.destroyAllWindows()
