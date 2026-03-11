from ultralytics import YOLO
import cv2
import supervision as sv

def main():
    print("=== Multi-Camera Tracking - Phase 2: Tracking ===")
    
    # Load YOLO model
    print("[1] Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    
    # Initialize tracker (ByteTrack algorithm)
    print("[2] Initializing tracker...")
    tracker = sv.ByteTrack()
    
    # Create annotators (for drawing)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # Open video
    print("[3] Opening video...")
    cap = cv2.VideoCapture('data/videos/test1.mp4')
    
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps}fps")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/tracking_result.mp4', fourcc, fps, (width, height))
    
    print("[4] Processing frames with tracking...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO detection
        results = model(frame, classes=[0], verbose=False)[0]
        
        # Convert YOLO results to supervision Detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Update tracker with detections
        detections = tracker.update_with_detections(detections)
        
        # Create labels with tracker IDs
        labels = [
            f"ID:{tracker_id} {confidence:.2f}"
            for tracker_id, confidence in zip(detections.tracker_id, detections.confidence)
        ]
        
        # Draw boxes and labels
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        # Write frame
        out.write(annotated_frame)
        
        # Progress
        if frame_count % 30 == 0:
            active_tracks = len(detections.tracker_id) if detections.tracker_id is not None else 0
            print(f"  Frame {frame_count}: {active_tracks} active tracks")
    
    print(f"[5] Done! Processed {frame_count} frames")
    print("Output saved to: output/tracking_result.mp4")
    
    cap.release()
    out.release()

if __name__ == "__main__":
    main()