import cv2
import time
import numpy as np
from ultralytics import YOLO
import queue

def init_model(model_path):
    """Initialize the YOLO model"""
    try:
        return YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def run_detection(camera_id, model_path, confidence_threshold, frame_queue, stop_event, detection_count):
    """Run object detection on camera frames"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        stop_event.set()
        return

    try:
        model = init_model(model_path)
        if model is None:
            stop_event.set()
            return
            
        prev_time = time.time()

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Run object detection
            results = model.predict(frame, conf=confidence_threshold)
            
            # Clear detection count for this frame
            frame_detections = {}
            
            # Draw detections
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    conf = box.conf[0].item()

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Count detections by class
                    if label not in frame_detections:
                        frame_detections[label] = 0
                    frame_detections[label] += 1
            
            # Update global detection counter
            for label, count in frame_detections.items():
                if label not in detection_count:
                    detection_count[label] = 0
                detection_count[label] = max(detection_count[label], count)
            
            # Add FPS counter
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert to RGB for Streamlit and put in queue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Put in queue, remove old frame if queue is full
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame_rgb)
    except Exception as e:
        print(f"Error in detection thread: {str(e)}")
    finally:
        cap.release()