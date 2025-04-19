import cv2
import time
import numpy as np
from ultralytics import YOLO
import queue
import os

def init_model(model_path):
    """Initialize the YOLO model"""
    try:
        return YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def run_detection(camera_id, model_path, confidence_threshold, frame_queue, stop_event, detection_count):
    """Run object detection on camera frames with fallback options"""
    
    # Try to open camera
    cap = cv2.VideoCapture(camera_id)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Warning: Could not open camera. Using demo mode.")
        # Try demo mode options in sequence
        

        demo_video = "assets/detection.mp4"
        if os.path.exists(demo_video):  
            cap = cv2.VideoCapture(demo_video)
        
       
        if not cap.isOpened():
            use_static_images = True
            # List of sample images with objects
            sample_images = ["assets/sample1.jpg", "assets/sample2.jpg"]
            img_index = 0
        else:
            use_static_images = False
    else:
        use_static_images = False

    try:
        model = init_model(model_path)
        if model is None:
            stop_event.set()
            return
        
        prev_time = time.time()
        
        while not stop_event.is_set():
            # Get frame (from camera, video or static images)
            if use_static_images:
                # Cycle through sample images
                if os.path.exists(sample_images[img_index]):
                    frame = cv2.imread(sample_images[img_index])
                    img_index = (img_index + 1) % len(sample_images)
                    time.sleep(2)  # Show each image for 2 seconds
                    ret = True
                else:
                    # Create blank frame with text if no sample images found
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Demo Mode - No Samples Found", (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    ret = True
            else:
                # Regular video capture
                ret, frame = cap.read()
                
            if not ret and not use_static_images and os.path.exists(demo_video):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
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
        if not use_static_images and cap.isOpened():
            cap.release()