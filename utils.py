import cv2
import numpy as np
import os
import time
import pandas as pd
from datetime import datetime

def create_placeholder_image(message, width=640, height=480):
    """Create a placeholder image with text"""
    placeholder = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(placeholder, message, (width//2 - 140, height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return placeholder

def save_screenshot(frame, directory="data"):
    """Save a screenshot of the current frame"""
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(directory, f"detection_{timestamp}.jpg")
    
    # Convert RGB back to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, frame_bgr)
    
    # Also save detection data to CSV
    csv_path = os.path.join(directory, "detections.csv")
    
    # Create or update the CSV
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=["timestamp", "filename"])
        df.to_csv(csv_path, index=False)
    
    # Append the new entry
    df = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": f"detection_{timestamp}.jpg"
    }])
    df.to_csv(csv_path, mode='a', header=False, index=False)
    
    return filepath