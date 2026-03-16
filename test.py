from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

# Test camera 1
cap = cv2.VideoCapture('data/videos/cam1.mp4')
ret, frame = cap.read()

if ret:
    results = model(frame, classes=[0], verbose=False, conf=0.25)[0]
    print(f"Camera 1 - Frame 1: {len(results.boxes)} people detected")
    
    # Save annotated frame
    annotated = results.plot()
    cv2.imwrite('test_cam1_detection.jpg', annotated)
    print("Saved: test_cam1_detection.jpg")

cap.release()

# Test camera 2
cap = cv2.VideoCapture('data/videos/cam2.mp4')
ret, frame = cap.read()

if ret:
    results = model(frame, classes=[0], verbose=False, conf=0.25)[0]
    print(f"Camera 2 - Frame 1: {len(results.boxes)} people detected")
    
    annotated = results.plot()
    cv2.imwrite('test_cam2_detection.jpg', annotated)
    print("Saved: test_cam2_detection.jpg")

cap.release()