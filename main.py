"""
Proiect: Recunoașterea elementelor într-o imagine/video (YOLOv8)
Echipa: 12-E1 (MARTIN DRAGOŞ-GABRIEL, VERMEŞAN BOGDAN-ANDREI)
Sursă documentație: https://docs.ultralytics.com/
"""

import cv2
import time
from ultralytics import YOLO

# Încărcăm un model pre-antrenat (YOLOv8n este varianta "nano", cea mai rapidă)
model = YOLO('yolov8n.pt')


def detect_objects_in_image(frame):
    """
    Cerința 1: Funcție care detectează obiecte și returnează datele brute.
    Returnează: rezultate (clase, cutii, scoruri)
    """
    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        detections.append({
            "coords": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
            "confidence": float(box.conf[0]),
            "class": int(box.cls[0]),
            "name": model.names[int(box.cls[0])]
        })
    return detections, results


def process_video_stream(source=0):
    """
    Cerința 2 & 3: Procesare stream video + Măsurare FPS.
    source=0 pentru webcam, sau 'cale/catre/video.mp4'
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Eroare: Nu s-a putut accesa sursa video.")
        return

    print("Apasă 'q' pentru a închide fereastra.")

    while True:
        start_time = time.time()  # Start cronometru pentru FPS

        ret, frame = cap.read()
        if not ret:
            break

        # Detectăm obiectele
        detections, results = detect_objects_in_image(frame)

        # Desenăm rezultatele folosind metoda built-in din Ultralytics pentru viteză
        # Alternativ, se poate desena manual cu cv2.rectangle folosind lista 'detections'
        annotated_frame = results.plot()

        # Calculăm FPS (Cerința 3)
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # Afișăm FPS pe imagine
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Afișăm în consolă FPS-ul
        print(f"Frame Processing Time: {end_time - start_time:.4f}s | FPS: {fps:.2f}")

        # Afișăm rezultatul vizual
        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

        # Ieșire la apăsarea tastei 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Rulăm pe webcam (sursa 0)
    process_video_stream(source=0)