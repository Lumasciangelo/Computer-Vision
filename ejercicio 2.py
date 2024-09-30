# # Implementar una balanza por cv que pueda pesar bananas, manzanas, naranjas, peras (imagen)

# # PointRend --> ResNet-50 con FPN

# from ultralytics import YOLO
# import cv2
# import numpy as np
# import requests
# from io import BytesIO

# def load_model():
#     # Cargar el modelo YOLOv8
#     model = YOLO('yolov8n.pt')  # Puedes usar 'n', 's', 'm', 'l', o 'x' según el tamaño deseado
#     return model

# def download_image(url):
#     response = requests.get(url)
#     image = np.array(bytearray(response.content), dtype=np.uint8)
#     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#     return image

# def detect_fruits(image_url, model):
#     # Descargar la imagen desde la URL
#     img = download_image(image_url)
    
#     # Realizar la detección
#     results = model(img)
    
#     # Obtener la primera imagen procesada (asumiendo que solo hay una imagen)
#     result = results[0]
    
#     # Dibujar las detecciones
#     for box in result.boxes:
#         x1, y1, x2, y2 = box.xyxy[0]
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         conf = float(box.conf)
#         cls = int(box.cls)
        
#         # Dibujar el bounding box
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
#         # Añadir etiqueta
#         label = f'{result.names[cls]} {conf:.2f}'
#         cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
#     return img, result


# # Uso del detector
# model = load_model()
# image_url = "https://elegifruta.com.ar/wp-content/uploads/2023/06/imagenpublinota-1A.png"
# img_with_detections, result = detect_fruits(image_url, model)

# # Mostrar la imagen con las detecciones
# cv2.imshow("Detección de frutas", img_with_detections)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Imprimir información sobre las detecciones
# for box in result.boxes:
#     cls = int(box.cls)
#     conf = float(box.conf)
#     bbox = box.xyxy[0].tolist()
#     print(f"Clase: {result.names[cls]}, Confianza: {conf:.2f}, Bounding Box: {bbox}")

# # Guardar la imagen con las detecciones
# cv2.imwrite("frutas_detectadas_yolov8.jpg", img_with_detections)

# # Imprimir información sobre las detecciones
# for box in result.boxes:
#     cls = int(box.cls)
#     conf = float(box.conf)
#     bbox = box.xyxy[0].tolist()
#     print(f"Clase: {result.names[cls]}, Confianza: {conf:.2f}, Bounding Box: {bbox}")

# # Guardar la imagen con las detecciones
# cv2.imwrite("frutas_detectadas_yolov8.jpg", img_with_detections)

from ultralytics import YOLO
import cv2
import numpy as np
import requests
from io import BytesIO

def load_model():
    # Cargar el modelo YOLOv8
    model = YOLO('yolov8n.pt')  # Puedes usar 'n', 's', 'm', 'l', o 'x' según el tamaño deseado
    return model

def download_image(url):
    response = requests.get(url)
    image = np.array(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def calcular_volumen(cls, bbox):
    # Aproximar el volumen según el tipo de fruta
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    if cls == "manzana" or cls == "naranja":  # Aproximamos a una esfera
        radio = (width + height) / 4  # Promedio de las dimensiones para el radio
        volumen = (4/3) * np.pi * (radio ** 3)
    
    # elif cls == "pera":  # Aproximamos a un elipsoide
    #     a = width / 2
    #     b = height / 2
    #     c = (a + b) / 2  # Suponemos que el eje c es el promedio
    #     volumen = (4/3) * np.pi * a * b * c
    
    elif cls == "banana":  # Aproximamos a un cilindro
        radio = width / 2  # Tomamos la mitad del ancho como radio
        altura = height  # El largo del bounding box es la altura del cilindro
        volumen = np.pi * (radio ** 2) * altura
    
    else:
        volumen = None  # Fruta no reconocida para el cálculo
    
    return volumen

def calcular_peso(cls, volumen):
    # Densidades aproximadas en gramos por cm^3
    densidades = {
        "manzana": 0.8,
        "naranja": 0.9,
        "pera": 0.6,
        "banana": 0.95  # Densidad aproximada para la banana
    }
    
    if cls in densidades and volumen:
        return volumen * densidades[cls]
    else:
        return None

def detect_fruits(image_url, model):
    # Descargar la imagen desde la URL
    img = download_image(image_url)
    
    # Realizar la detección
    results = model(img)
    
    # Obtener la primera imagen procesada (asumiendo que solo hay una imagen)
    result = results[0]
    
    # Dibujar las detecciones y calcular volumen/peso
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = float(box.conf)
        cls = int(box.cls)
        nombre_fruta = result.names[cls]
        
        # Dibujar el bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Añadir etiqueta
        label = f'{nombre_fruta} {conf:.2f}'
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Calcular volumen y peso
        bbox = [x1, y1, x2, y2]
        volumen = calcular_volumen(nombre_fruta, bbox)
        peso = calcular_peso(nombre_fruta, volumen)
        
        # Verificar que el volumen y el peso no sean None antes de imprimir
        if volumen is not None and peso is not None:
            print(f"Fruta: {nombre_fruta}, Volumen: {volumen:.2f} cm^3, Peso estimado: {peso:.2f} g")
        else:
            print(f"Fruta: {nombre_fruta}, no se pudo calcular el volumen o el peso.")

    return img, result


# Uso del detector
model = load_model()
image_url = "https://elegifruta.com.ar/wp-content/uploads/2023/06/imagenpublinota-1A.png"
img_with_detections, result = detect_fruits(image_url, model)

# Mostrar la imagen con las detecciones
cv2.imshow("Detección de frutas", img_with_detections)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar la imagen con las detecciones
cv2.imwrite("frutas_detectadas_yolov8.jpg", img_with_detections)
