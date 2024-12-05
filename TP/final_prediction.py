import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import os
import pyautogui

# Cargar el modelo entrenado
model = tf.keras.models.load_model('mi_modelo.h5')

# Inicializar MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inicializar captura de video (cámara en vivo)
captura = cv2.VideoCapture(0)

if not captura.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

# Mapa de gestos
mapa_gestos = {
    0: 'Puño',          # Abrir navegador
    1: 'Ok',            # Minimizar todas las ventanas
    2: 'Paz',           # Tomar una captura de pantalla
    3: 'Like',          # Subir el volumen
    4: 'Dislike',       # Bloquear pantalla
    5: 'Cuerno',        # Abrir explorador de archivos
    6: 'Suerte',        # Bajar volumen
    7: 'Mano abierta',  # Cambiar ventana activa
    8: 'Pistola',       # Cerrar la ventana activa
    9: 'Tres dedos arriba'  # Tomar una selfie
}

# Funciones de acciones
def bloquear_pantalla():
    os.system('rundll32.exe user32.dll,LockWorkStation')

def minimizar_todo():
    pyautogui.hotkey('win', 'd')

def captura_pantalla():
    screenshot = pyautogui.screenshot()
    screenshot.save('captura_pantalla.png')

def subir_volumen():
    pyautogui.press('volumeup')

def abrir_navegador():
    os.system('start chrome')

def abrir_explorador():
    os.system('explorer')

def bajar_volumen():
    pyautogui.press('volumedown')

def cambiar_ventana():
    pyautogui.hotkey('alt', 'tab')

def cerrar_ventana():
    pyautogui.hotkey('alt', 'f4')

def tomar_selfie(frame):
    cv2.imwrite('selfie.png', frame)

# Bucle para capturar video en vivo
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = captura.read()
        if not ret:
            print("No se puede obtener el frame.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        manos_detectadas = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                puntos_mano = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                manos_detectadas.append(puntos_mano)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(manos_detectadas) == 0:
            texto = 'No se detectaron manos'
            cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow("Video en Vivo", frame)
            continue

        if len(manos_detectadas) == 1:
            manos_detectadas.append([(0, 0, 0)] * 21)

        input_data = np.array([manos_detectadas])
        predictions = model.predict(input_data)
        predicted_label = np.argmax(predictions)

        texto = mapa_gestos.get(predicted_label, 'Gesto desconocido')
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Video en Vivo", frame)

        # Realizar la acción correspondiente
        if predicted_label == 0:
            abrir_navegador()
        elif predicted_label == 1:
            minimizar_todo()
        elif predicted_label == 2:
            captura_pantalla()
        elif predicted_label == 3:
            subir_volumen()
        elif predicted_label == 4:
            bloquear_pantalla()
        elif predicted_label == 5:
            abrir_explorador()
        elif predicted_label == 6:
            bajar_volumen()
        elif predicted_label == 7:
            cambiar_ventana()
        elif predicted_label == 8:
            cerrar_ventana()
        elif predicted_label == 9:
            tomar_selfie(frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break

cv2.destroyAllWindows()
