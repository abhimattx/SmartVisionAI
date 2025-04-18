# 🧠 SmartVision AI

> Real-time object detection dashboard powered by YOLOv8 and OpenCV

<p align="center">
  <img src="assets/detection.jpg" alt="SmartVision AI Screenshot" width="80%">
</p>

SmartVision AI is a comprehensive object detection application that captures live webcam video, detects and labels objects in real-time, counts them, and offers export functionality for screenshots and logs.

**Perfect for:** Real-world demos, AI portfolios, and computer vision job interviews.

## ✨ Features

| 🔍 Detection | 🎛️ Controls | 📊 Data & Export |
|-------------|------------|----------------|
| Real-time YOLOv8 processing | Confidence threshold slider | Screenshot capture & saving |
| Multiple model options (n/s/m) | Camera selection | CSV export of detections |
| Fast frame processing | Start/stop buttons | Object counting statistics |
| Bounding boxes & labels | | Timestamped data |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam access

### Installation

```bash
# Clone the repository
git clone https://github.com/abhimattx/SmartVisionAI.git
cd smartvision-ai

# Install dependencies
pip install -r requirements.txt
```

---

 ▶️ Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

---

📁 Folder Structure

```
smartvision-ai/
├── app.py              # Streamlit UI
├── detect.py           # YOLO model logic
├── utils.py            # Drawing, counting, saving tools
├── requirements.txt
├── README.md
├── assets/             # Screenshots, preview gifs
├── data/               # Saved frames, CSV logs
└── .gitignore
```

---

🧠 Model Notes

This app supports YOLOv8. You can download the models from:

- [yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)
- [yolov8s.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt)
- [yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt)

Important: Model files (.pt) are downloaded automatically at first run.

---

🛰️ Future Additions

- [ ] Live demo on Hugging Face or Streamlit Cloud
- [ ] Object tracking with IDs
- [ ] Live detection analytics dashboard
- [ ] Email/Telegram alert system


🧑‍💻 Author

Made by - Abhishek Singh
Feel free to connect on [LinkedIn](https://www.linkedin.com/in/abhimattx/)

📄 License
This project is licensed under the MIT License.
