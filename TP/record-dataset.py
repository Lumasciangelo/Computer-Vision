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
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:
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

        manos_detectadas = []
        
        # Procesar las manos detectadas
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                puntos_mano = []  # Reseteamos para cada mano
                for lm in hand_landmarks.landmark:
                    puntos_mano.append((lm.x, lm.y, lm.z))  # Coordenadas 3D
                manos_detectadas.append(puntos_mano)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Normalizar para asegurarnos de tener ambas manos
        if len(manos_detectadas) == 1:  # Solo una mano detectada
            manos_detectadas.append([(0, 0, 0)] * 21)  # Añadir mano faltante con ceros
        elif len(manos_detectadas) == 0:  # Ninguna mano detectada
            manos_detectadas = [[(0, 0, 0)] * 21, [(0, 0, 0)] * 21]

        # Mostrar el frame con los puntos clave dibujados
        cv2.imshow("Video en Vivo", frame)

        # Etiquetar manualmente las imágenes según el gesto
        key = cv2.waitKey(5) & 0xFF
        if key == ord('0') and len(manos_detectadas) > 0:
            Y.append(0)
            X.append(manos_detectadas)
            print("Etiqueta: Puño (0)")
        elif key == ord('1') and len(manos_detectadas) > 0:
            Y.append(1)
            X.append(manos_detectadas)
            print("Etiqueta: Ok (1)")
        elif key == ord('2') and len(manos_detectadas) > 0:
            Y.append(2)
            X.append(manos_detectadas)
            print("Etiqueta: Paz (2)")
        elif key == ord('3') and len(manos_detectadas) > 0:
            Y.append(3)
            X.append(manos_detectadas)
            print("Etiqueta: Like (3)")
        elif key == ord('4') and len(manos_detectadas) > 0:
            Y.append(4)
            X.append(manos_detectadas)
            print("Etiqueta: Dislike (4)")
        elif key == ord('5') and len(manos_detectadas) > 0:
            Y.append(5)
            X.append(manos_detectadas)
            print("Etiqueta: Cuerno (5)")
        elif key == ord('6') and len(manos_detectadas) > 0:
            Y.append(6)
            X.append(manos_detectadas)
            print("Etiqueta: Suerte (6)")
        elif key == ord('7') and len(manos_detectadas) > 0:
            Y.append(7)
            X.append(manos_detectadas)
            print("Etiqueta: Mano abierta (7)")
        elif key == ord('8') and len(manos_detectadas) > 0:
            Y.append(8)
            X.append(manos_detectadas)
            print("Etiqueta: Pistola (8)")
        elif key == ord('9') and len(manos_detectadas) > 0:
            Y.append(9)
            X.append(manos_detectadas)
            print("Etiqueta: Tres dedos arriba (9)")
        elif key == ord('q'):  # Salir con 'q'
            break

# Guardar el dataset en archivos .npy
np.save('mi_dataset_X.npy', np.array(X))
np.save('mi_dataset_Y.npy', np.array(Y))

print("Datos guardados en 'mi_dataset_X.npy' y 'mi_dataset_Y.npy'.")
cv2.destroyAllWindows()
