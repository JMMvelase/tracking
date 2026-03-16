from ultralytics import YOLO
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

class GlobalIDManager:
    """Manages global IDs across cameras using appearance features"""
    
    def __init__(self, similarity_threshold=0.5):
        self.similarity_threshold = similarity_threshold
        self.global_id_counter = 0
        self.global_tracks = {}  # {global_id: {'feature': ..., 'cameras': set()}}
        self.camera_to_global = defaultdict(dict)  # {camera_id: {local_id: global_id}}
        
    def cosine_similarity(self, feat1, feat2):
        """Calculate cosine similarity between two feature vectors"""
        feat1 = np.array(feat1).flatten()
        feat2 = np.array(feat2).flatten()
        
        dot_product = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def match_to_global_id(self, feature, camera_id, local_id):
        """Match a local track to a global ID using appearance similarity"""
        
        if feature is None:
            return self._create_new_global_id(feature, camera_id, local_id)
        
        # Check if this local ID already has a global ID
        if local_id in self.camera_to_global[camera_id]:
            global_id = self.camera_to_global[camera_id][local_id]
            # Update feature
            self.global_tracks[global_id]['feature'] = feature
            self.global_tracks[global_id]['cameras'].add(camera_id)
            return global_id
        
        # Compare with all existing global tracks
        best_match_id = None
        best_similarity = 0.0
        
        for global_id, track_info in self.global_tracks.items():
            if track_info['feature'] is None:
                continue
            
            similarity = self.cosine_similarity(feature, track_info['feature'])
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match_id = global_id
        
        if best_match_id is not None:
            # Matched to existing global ID
            print(f"  [MATCH] Cam{camera_id} Local:{local_id} → Global:{best_match_id} (similarity: {best_similarity:.3f})")
            self.camera_to_global[camera_id][local_id] = best_match_id
            self.global_tracks[best_match_id]['feature'] = feature
            self.global_tracks[best_match_id]['cameras'].add(camera_id)
            return best_match_id
        else:
            # Create new global ID
            return self._create_new_global_id(feature, camera_id, local_id)
    
    def _create_new_global_id(self, feature, camera_id, local_id):
        """Create a new global ID for a track"""
        global_id = self.global_id_counter
        self.global_id_counter += 1
        self.global_tracks[global_id] = {
            'feature': feature,
            'cameras': {camera_id}
        }
        self.camera_to_global[camera_id][local_id] = global_id
        return global_id
    
    def get_stats(self):
        """Get statistics about cross-camera tracking"""
        cross_camera_tracks = [
            gid for gid, info in self.global_tracks.items()
            if len(info['cameras']) > 1
        ]
        return {
            'total_global_ids': self.global_id_counter,
            'cross_camera_matches': len(cross_camera_tracks),
            'cross_camera_ids': cross_camera_tracks
        }


def process_camera(video_path, camera_id, model, global_id_manager):
    """Process a single camera stream"""
    
    print(f"\n{'='*60}")
    print(f"Processing Camera {camera_id}: {video_path}")
    print(f"{'='*60}")
    
    # Initialize DeepSORT tracker
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=0.7,
        embedder="mobilenet",
        embedder_wts=None,
        polygon=False,
        today=None
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps ({total_frames} frames)")
    
    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = f'output/camera{camera_id}_global.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    local_tracks_seen = set()
    detection_count = 0
    
    print("Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO detection with lower confidence threshold
        results = model(frame, classes=[0], verbose=False, conf=0.25)[0]
        
        # Prepare detections for DeepSORT
        raw_detections = []
        
        if len(results.boxes) > 0:
            detection_count += len(results.boxes)
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                w = x2 - x1
                h = y2 - y1
                
                # Skip very small detections
                if w < 20 or h < 20:
                    continue
                
                bbox = [float(x1), float(y1), float(w), float(h)]
                raw_detections.append((bbox, conf, 'person'))
        
        # Update DeepSORT tracker
        try:
            if len(raw_detections) > 0:
                tracks = tracker.update_tracks(raw_detections, frame=frame)
            else:
                # No detections this frame
                tracks = []
        except Exception as e:
            print(f"  Warning at frame {frame_count}: {e}")
            tracks = []
        
        # Process each track
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            local_id = track.track_id
            local_tracks_seen.add(local_id)
            
            # Get ReID feature
            try:
                feature = track.get_feature()
            except:
                feature = None
            
            # Match to global ID
            global_id = global_id_manager.match_to_global_id(
                feature, camera_id, local_id
            )
            
            # Draw bounding box
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Create label
            label = f"Cam{camera_id} L:{local_id} G:{global_id}"
            
            # Draw label with background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Write frame
        out.write(frame)
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"  Frame {frame_count}/{total_frames} ({progress:.1f}%) - Detections: {len(raw_detections)}, Tracks: {len([t for t in tracks if t.is_confirmed()])}")
    
    cap.release()
    out.release()
    
    print(f"\nCamera {camera_id} complete:")
    print(f"  - Frames processed: {frame_count}")
    print(f"  - Total detections: {detection_count}")
    print(f"  - Unique local tracks: {len(local_tracks_seen)}")
    print(f"  - Output saved: {output_path}")


def main():
    print("="*60)
    print("  Multi-Camera Tracking System - Phase 3")
    print("  Using DeepSORT + Global ID Matching")
    print("="*60)
    
    # Load YOLO model
    print("\n[1] Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    print("    ✓ YOLO loaded")
    
    # Initialize global ID manager
    print("\n[2] Initializing Global ID Manager...")
    global_id_manager = GlobalIDManager(similarity_threshold=0.5)
    print("    ✓ Global ID Manager ready")
    print(f"    - Similarity threshold: 0.5")
    
    # Define camera sources
    cameras = [
        ('data/videos/cam1.mp4', 1),
        ('data/videos/cam2.mp4', 2),
    ]
    
    # Process each camera
    print("\n[3] Processing cameras...")
    for video_path, camera_id in cameras:
        process_camera(video_path, camera_id, model, global_id_manager)
    
    # Print final statistics
    print("\n" + "="*60)
    print("  FINAL STATISTICS")
    print("="*60)
    
    stats = global_id_manager.get_stats()
    print(f"Total unique global IDs: {stats['total_global_ids']}")
    print(f"Cross-camera matches: {stats['cross_camera_matches']}")
    
    if stats['cross_camera_ids']:
        print(f"\nGlobal IDs seen in multiple cameras:")
        for gid in stats['cross_camera_ids']:
            cameras = global_id_manager.global_tracks[gid]['cameras']
            print(f"  - Global ID {gid}: Cameras {sorted(cameras)}")
    
    print("\n✓ Processing complete!")
    print("Output videos saved to: output/camera*_global.mp4")


if __name__ == "__main__":
    main()