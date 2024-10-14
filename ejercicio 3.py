# Importar las librerías necesarias
import supervision as sv  # Librería para el seguimiento de objetos
from ultralytics import YOLO  # Librería para la detección de objetos con YOLO
import numpy as np  # Librería para operaciones matemáticas y manejo de arrays

# Definir rutas de los videos de entrada y salida, y el nombre del modelo
SOURCE_VIDEO_PATH = "highway_600.mp4"  # Ruta al video de entrada
TARGET_VIDEO_PATH = "highway_600_tracking.mp4"  # Ruta donde se guardará el video con seguimiento de objetos
MODEL_NAME = "yolov8x.pt"  # Nombre del modelo preentrenado de YOLO a utilizar

# Inicializar el modelo de detección y el rastreador de objetos
model = YOLO(MODEL_NAME)  # Cargar el modelo de YOLO
tracker = sv.ByteTrack()  # Inicializar ByteTrack para el seguimiento de objetos

# Inicializar los anotadores para las cajas delimitadoras y las etiquetas
bounding_box_annotator = sv.BoundingBoxAnnotator()  # Para dibujar cajas delimitadoras
label_annotator = sv.LabelAnnotator()  # Para dibujar etiquetas (IDs de seguimiento)

# Definir la función de callback que procesa cada frame
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame)[0]  # Detectar objetos en el frame actual con YOLO
    detections = sv.Detections.from_ultralytics(results)  # Convertir resultados a formato de supervision
    detections = tracker.update_with_detections(detections)  # Actualizar el estado del rastreador con las detecciones

    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]  # Crear etiquetas con los IDs de seguimiento

    # Anotar el frame con bounding boxes y etiquetas
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame  # Devolver el frame anotado

# Procesar el video: leer, aplicar callback a cada frame y guardar el resultado
sv.process_video(
    source_path=SOURCE_VIDEO_PATH,  # Ruta del video original
    target_path=TARGET_VIDEO_PATH,  # Ruta del video resultante
    callback=callback  # Función de callback para procesar cada frame
)

########################################################################################
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

video_path = 'c:\Users\Usuario\Downloads\Video.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if ret:
        results = model.track(frame, persist = True)
        image = results[0].plot()
        cv2.imshow()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break