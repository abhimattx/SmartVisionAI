import streamlit as st
import numpy as np
import cv2
import time
from threading import Thread, Event
import queue
from detect import run_detection, init_model
from utils import save_screenshot, create_placeholder_image

st.set_page_config(page_title="SmartVision AI", layout="wide")

st.title("📹 SmartVision AI – Real-Time Object Detection")
st.write("Powered by YOLOv8 and OpenCV")

# Configuration parameters
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
camera_id = st.sidebar.selectbox("Select Camera", [0, 1, 2, 3], 0)
model_option = st.sidebar.selectbox(
    "Select Model", 
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"], 
    0
)

# Use session state to track app state
if 'running' not in st.session_state:
    st.session_state.running = False
# To this:
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = Event()  # CORRECTED: Use threading.Event() instead of Event()
if 'frame_queue' not in st.session_state:
    st.session_state.frame_queue = queue.Queue(maxsize=1)
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = {}
if 'screenshot_requested' not in st.session_state:
    st.session_state.screenshot_requested = False    

def toggle_detection():
    st.session_state.running = not st.session_state.running
    if st.session_state.running:
        st.session_state.stop_event.clear()
    else:
        st.session_state.stop_event.set()

def request_screenshot():
    st.session_state.screenshot_requested = True

# Create UI controls
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    start_button = st.button("Start Detection", key="start", on_click=toggle_detection, 
                            disabled=st.session_state.running)
with col2:
    stop_button = st.button("Stop Detection", key="stop", on_click=toggle_detection, 
                        disabled=not st.session_state.running)
with col3:
    screenshot_button = st.button("Save Screenshot", key="screenshot",
                              on_click=request_screenshot,
                              disabled=not st.session_state.running)

# Frame placeholder
stframe = st.empty()

# Detection statistics
if st.session_state.running:
    st.sidebar.subheader("Detection Statistics")
    stats_placeholder = st.sidebar.empty()

# Add this after creating the app interface
if st.session_state.running:
    # Check if camera is available on this platform
    is_demo_mode = False
    
    try:
        temp_cap = cv2.VideoCapture(camera_id)
        is_camera_available = temp_cap.isOpened()
        if temp_cap.isOpened():
            temp_cap.release()
        else:
            is_demo_mode = True
    except:
        is_demo_mode = True
    
    if is_demo_mode:
        st.warning("⚠️ Camera not available. Running in demo mode with sample data.")

# Run the detection in a separate thread when active
if st.session_state.running:
    thread = Thread(
        target=run_detection, 
        args=(
            camera_id,
            model_option,
            confidence_threshold,
            st.session_state.frame_queue,
            st.session_state.stop_event,
            st.session_state.detection_count
        )
    )
    thread.daemon = True
    thread.start()
    
    # Display frames from the queue
    placeholder_displayed = False
    while st.session_state.running:
        try:
            frame = st.session_state.frame_queue.get(timeout=1)
            stframe.image(frame, channels="RGB", use_container_width=True)
            
            # Update statistics
            if hasattr(st.session_state, 'detection_count'):
                stats_text = "Objects detected:\n"
                for obj, count in st.session_state.detection_count.items():
                    stats_text += f"- {obj}: {count}\n"
                stats_placeholder.text(stats_text)
                
            # Handle screenshot button - MODIFIED
            if st.session_state.screenshot_requested:
                saved_path = save_screenshot(frame)
                st.sidebar.success(f"Screenshot saved!")
                st.session_state.screenshot_requested = False  # Reset after taking screenshot
                
            placeholder_displayed = False
        except queue.Empty:
            if not placeholder_displayed:
                placeholder = create_placeholder_image("Waiting for frames...")
                stframe.image(placeholder, channels="RGB", use_container_width=True)
                placeholder_displayed = True
        except Exception as e:
            st.error(f"Error displaying frame: {str(e)}")
            break
else:
    # Display placeholder image when not running
    placeholder = create_placeholder_image("Press 'Start Detection'")
    stframe.image(placeholder, channels="RGB", use_container_width=True)