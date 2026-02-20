# 🎭 Real-Time Emotion Detection using Deep Learning

A real-time facial emotion detection system built using OpenCV and a Convolutional Neural Network (CNN).  
The system detects faces from webcam input and predicts human emotions instantly.

---

## 🚀 Features

- Real-time face detection using Haar Cascade
- Emotion classification using trained CNN model (.h5)
- Supports multiple emotions:
  - Angry
  - Happy
  - Neutral
  - Sad
  - Surprised
- Live webcam prediction
- Bounding box + emotion label display

---

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Keras / TensorFlow
- CNN (Convolutional Neural Network)

---

## 📂 Project Structure

```
Real-Time-Emotion-Detection/
│
├── Emotion_detection_model.h5
├── haarcascade_frontalface_default.xml
├── main.py
├── data/
│   ├── angry/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
└── README.md
```

---

## ⚙️ How It Works

1. Capture real-time video using webcam
2. Convert frame to grayscale
3. Detect faces using Haar Cascade
4. Resize face image to 48x48
5. Normalize pixel values
6. Predict emotion using trained CNN model
7. Display predicted emotion on screen

---

## ▶️ How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/your-username/real-time-emotion-detection-ml.git
cd real-time-emotion-detection-ml
```

2. Install required libraries:

```bash
pip install opencv-python numpy tensorflow keras
```

3. Run the project:

```bash
python main.py
```

Press **'q'** to exit the webcam.

---

## 🎯 Future Improvements

- Add more emotion categories
- Improve model accuracy
- Deploy as web app using Flask/Streamlit
- Add GUI interface

---

## 👨‍💻 Author

Swetab Baranwal  
B.Tech CSE (3rd Year)  
AIML Certified – CTTC Bhubaneswar  

---

## ⭐ If you like this project

Give it a star on GitHub!
