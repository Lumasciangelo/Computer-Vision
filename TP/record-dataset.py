import cv2
import os
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inicializar captura de video (cámara en vivo)
captura = cv2.VideoCapture(0)  # 0 es la cámara por defecto

if not captura.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

X = []
Y = []

# Bucle para capturar video en vivo
with mp_hands.Hands(static_image_mode=False, max_num_hands=4, min_detection_confidence=0.7) as hands:
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

        # Mostrar el frame con los puntos clave dibujados
        cv2.imshow("Video en Vivo", frame)

        # Etiquetar manualmente las imágenes según el gesto
        key = cv2.waitKey(5) & 0xFF
        if key == ord('r') and puntos_mano:
            Y.append(0)
            X.append(puntos_mano)
            print("Etiqueta: Piedra (0)")
        elif key == ord('p') and puntos_mano:
            Y.append(1)
            X.append(puntos_mano)
            print("Etiqueta: Papel (1)")
        elif key == ord('s') and puntos_mano:
            Y.append(2)
            X.append(puntos_mano)
            print("Etiqueta: Tijera (2)")
        elif key == ord('q'):  # Salir con 'q'
            break

# Guardar el dataset en archivos .npy
np.save('mi_dataset_X.npy', np.array(X))
np.save('mi_dataset_Y.npy', np.array(Y))

print("Datos guardados en 'mi_dataset_X.npy' y 'mi_dataset_Y.npy'.")
cv2.destroyAllWindows()
