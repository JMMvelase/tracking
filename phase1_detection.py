from ultralytics import YOLO
import cv2

def main():
    print("=== Multi-Camera Tracking - Phase 1: Detection ===")
    
    # Load YOLO model
    print("[1] Loading YOLO model...")
    model = YOLO('yolov8n.pt')  # Auto-downloads if not present
    
    # Open video
    print("[2] Opening video...")
    cap = cv2.VideoCapture('data/videos/test1.mp4')
    
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps}fps")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/detection_result.mp4', fourcc, fps, (width, height))
    
    print("[3] Processing frames...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO detection
        results = model(frame, classes=[0], verbose=False)  # class 0 = person
        people_count = len(results[0].boxes)
        print(f"Frame {frame_count}: Detected {people_count} people")
        # Draw detections
        annotated_frame = results[0].plot()
        
        # Write frame
        out.write(annotated_frame)
        
        # Progress
        if frame_count % 30 == 0:
            print(f"  Processed frame {frame_count}")
    
    print(f"[4] Done! Processed {frame_count} frames")
    print("Output saved to: output/detection_result.mp4")
    
    cap.release()
    out.release()

if __name__ == "__main__":
    main()