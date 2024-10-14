# En un video, detectar bananas, manzanas, naranjas y peras y contar cuantas hay

from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

video = 'C:\Users\Usuario\Documents\GitHub\Computer-Vision\Video.mp4'
cap = cv2.VideoCapture(video)

while True:
    ret, frame = cap.read()
    if ret:
        results = model.track(frame, persist = True)
        image = results[0].plot()
        cv2.imshow()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break