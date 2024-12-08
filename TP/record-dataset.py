import cv2
import os
import mediapipe as mp
import numpy as np

# Crear carpetas para guardar las imágenes
output_dir_original = "imagenes_originales"
output_dir_procesadas = "imagenes_procesadas"
os.makedirs(output_dir_original, exist_ok=True)
os.makedirs(output_dir_procesadas, exist_ok=True)

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

        if not ret:
            print("No se puede obtener el frame.")
            break

        # Convertir BGR a RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar el frame con MediaPipe
        results = hands.process(image_rgb)

        # Crear lista para las landmarks detectadas
        manos_detectadas = []

        # Procesar las manos detectadas
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                puntos_mano = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                manos_detectadas.append(puntos_mano)

        # Normalizar datos para el dataset
        while len(manos_detectadas) < 2:
            manos_detectadas.append([(0, 0, 0)] * 21)

        # Mostrar el frame con los puntos clave dibujados
        frame_with_landmarks = frame.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_with_landmarks, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Video en Vivo", frame_with_landmarks)

        # Etiquetar manualmente las imágenes según el gesto
        key = cv2.waitKey(5) & 0xFF
        if key in [ord(str(i)) for i in range(10)] and len(manos_detectadas) > 0:
            etiqueta = int(chr(key))  # Obtener la etiqueta como número
            Y.append(etiqueta)
            X.append(manos_detectadas)

            # Generar nombres únicos para las imágenes
            original_filename = os.path.join(output_dir_original, f"gesto_{etiqueta}_original_{len(X)}.jpg")
            processed_filename = os.path.join(output_dir_procesadas, f"gesto_{etiqueta}_procesada_{len(X)}.jpg")

            # Guardar la imagen original
            cv2.imwrite(original_filename, frame)

            # Guardar la imagen procesada
            cv2.imwrite(processed_filename, frame_with_landmarks)

            print(f"Etiqueta: {etiqueta} - Imagen original: {original_filename} - Imagen procesada: {processed_filename}")

        elif key == ord('q'):  # Salir con 'q'
            break

# Guardar el dataset en archivos .npy
np.save('mi_dataset_X.npy', np.array(X))
np.save('mi_dataset_Y.npy', np.array(Y))

print("Datos guardados en 'mi_dataset_X.npy' y 'mi_dataset_Y.npy'.")
cv2.destroyAllWindows()
