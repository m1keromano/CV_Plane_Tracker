import cv2
import argparse
from ultralytics import YOLO
import config
from tracker import PlaneTracker
import sys

def main():
    parser = argparse.ArgumentParser(description="CV Plane Tracker")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--yolo_model", default="yolov8n.pt", help="Path to YOLO model")
    args = parser.parse_args()

    # Load YOLO
    print(f"Loading YOLO model: {args.yolo_model}...")
    try:
        model = YOLO(args.yolo_model)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        sys.exit(1)

    # Open Video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video_path}")
        sys.exit(1)

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps} FPS")

    # Initialize Tracker
    tracker = PlaneTracker(fps)

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize if needed (optional, for speed)
        frame = cv2.resize(frame, (config.RESIZE_WIDTH, int(config.RESIZE_WIDTH * height / width)))

        # Process every Nth frame
        if frame_count % config.MEASUREMENT_SKIP == 0:
            # Run Detection
            results = model(frame, verbose=False)
            
            # Extract measurement (assume airplane class is 4 for COCO, or just take the highest confidence object if we assume only plane is present)
            # Better: Look for class 'airplane' (id 4 in COCO)
            measurement = None
            best_box = None
            best_conf = 0
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    # COCO class 4 is airplane. If using a custom model, this might differ.
                    # We'll assume COCO or just take the most confident detection if it's the only thing.
                    # Let's stick to 'airplane' class if possible, or just the best detection.
                    # For robustness, let's check if the model has class names.
                    cls_name = model.names[cls_id]
                    
                    if cls_name == 'airplane' or 'plane' in cls_name:
                        if conf > best_conf:
                            best_conf = conf
                            x1, y1, x2, y2 = box.xyxy[0]
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            measurement = (cx, cy)
                            best_box = (int(x1), int(y1), int(x2), int(y2))
            
            # Update Tracker
            tracker.update(measurement)

            # Draw Bounding Box
            if best_box is not None:
                cv2.rectangle(frame, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), 2)

        # Visualization
        # Draw Actual Path (History) - Blue
        if len(tracker.history) > 1:
            for i in range(len(tracker.history) - 1):
                pt1 = tracker.history[i]
                pt2 = tracker.history[i+1]
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        # Draw Predicted Path - Red
        pred_path = tracker.get_prediction_path()
        if len(pred_path) > 1:
            # Start from current state pos
            start_pt = (int(tracker.kf.x[0,0]), int(tracker.kf.x[1,0]))
            cv2.line(frame, start_pt, pred_path[0], (0, 0, 255), 2)
            for i in range(len(pred_path) - 1):
                pt1 = pred_path[i]
                pt2 = pred_path[i+1]
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

        # Draw Current Measurement (if any this frame)
        # Since we only measure every N frames, we might want to persist the last measurement visualization?
        # Or just draw it when it happens. Let's just draw a circle at the current estimated position.
        if tracker.is_initialized:
            est_pos = (int(tracker.kf.x[0,0]), int(tracker.kf.x[1,0]))
            cv2.circle(frame, est_pos, 5, (0, 255, 0), -1) # Green dot for estimate

        # Display info
        cv2.putText(frame, f"Mode: {config.STATE_VECTOR_MODE}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if tracker.is_initialized:
            vx, vy = tracker.kf.x[2, 0], tracker.kf.x[3, 0]
            if tracker.kf.mode == '6D':
                ax, ay = tracker.kf.x[4, 0], tracker.kf.x[5, 0]
                cv2.putText(frame, f"Vel: ({vx:.1f}, {vy:.1f})", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Acc: ({ax:.2f}, {ay:.2f})", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Vel: ({vx:.1f}, {vy:.1f})", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Plane Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
