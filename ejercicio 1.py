# Detectar a una persona u objeto en un video y desenfocar el fondo (blurred)

import cv2
import numpy as np
from ultralytics import YOLO

# Cargar el modelo YOLOv8 con segmentación
model = YOLO('yolov8n-seg')  # YOLOv8 con segmentación

# Inicializo Nombres GUI
win_frame = 'Frame'

# Inicializo la captura de video
cap = cv2.VideoCapture(0)

# Creo ventana
cv2.namedWindow(win_frame)

while True:
    # Obtengo frame
    ret, frame = cap.read()
    if frame is None:
        break

    # Realizo las detecciones con YOLO (incluye segmentación)
    results = model(frame)[0]

    # Crear una máscara vacía con el mismo tamaño que el frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Obtener las máscaras de los objetos detectados (personas, etc.)
    if results.masks is not None:
        # Acceder a las máscaras usando results.masks.data
        for seg_mask in results.masks.data:
            # Cada máscara es un tensor de PyTorch; convertirlo a un numpy array
            seg_mask = seg_mask.cpu().numpy().astype(np.uint8)  # Convertimos a uint8 (0 o 1)

            # Redimensionamos la máscara al tamaño del frame si es necesario
            seg_mask = cv2.resize(seg_mask, (frame.shape[1], frame.shape[0]))

            # Combinar las máscaras en una sola máscara general
            mask = cv2.bitwise_or(mask, seg_mask * 255)

    # Aplico un desenfoque gaussiano al frame completo
    blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)

    # Combino el fondo desenfocado con las áreas de los objetos sin desenfoque
    frame_result = np.where(mask[:, :, None] == 255, frame, blurred_frame)

    # Muestro el video con el fondo desenfocado y la forma precisa de la persona
    cv2.imshow(win_frame, frame_result)

    # Termino al presionar 'q' o la tecla ESC
    key = cv2.waitKey(30)
    if key == ord('q') or key == 27:
        break

# Libero recursos
cap.release()
cv2.destroyAllWindows()
