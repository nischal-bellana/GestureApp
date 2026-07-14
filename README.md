# 🖐️ Gesture Pipeline Controller

A lightweight, machine-learning-powered background application that maps custom hand gestures to automated OS-level actions. 

This project uses a highly optimized **Two-Stage Hybrid Architecture** to balance real-time responsiveness with extremely low CPU/GPU footprint, ensuring it can run silently in the background without lagging your system.

## ✨ Features

* **Two-Stage "Gatekeeper" Logic:** Uses a lightweight static gesture (e.g., Open Palm) to "arm" the system, meaning the heavier dynamic sequence tracking only runs when absolutely necessary.
* **Scale & Translation Invariance:** Uses custom coordinate normalization so gestures are recognized flawlessly regardless of how close or far your hand is from the webcam, or where it appears on screen.
* **Visual Configuration Editor:** Includes a fully-featured desktop UI built in Tkinter to manage static/dynamic gesture mappings, assign OS actions, and launch training pipelines without touching code.
* **Asynchronous Inference:** Utilizes MediaPipe's Task API in `LIVE_STREAM` mode to process frames on a background thread, completely eliminating webcam stutter.

## 🛠️ Tech Stack

* **Computer Vision:** OpenCV, MediaPipe Tasks API
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Data Processing:** NumPy, Pandas
* **GUI & Automation:** Tkinter (Python standard library), PyAutoGUI / Subprocess / AppOpener

## 🗂️ Project Structure

```text
├── configMaker.py            # The main Desktop GUI for managing configs and pipelines
├── main.py                   # Handles webcam capture, feature normalization, and inference
├── training_static_model.py  # ML script to train Random Forests on CSV data
├── actions.py                # Abstraction layer mapping string commands to OS actions
├── hand_landmarker.task      # MediaPipe pre-trained model (Must be downloaded)
├── Configs/                  # Directory containing your .json configuration profiles
├── Models/                   # Pickle files of the Random forest Models
└── Training_Samples/         # Recorded feature vectors of gestures stored in csv files
```

## 🚀 Installation & Setup
**Clone the repository:**

```bash
git clone https://github.com/nischal-bellana/GestureApp.git
```
**Install the dependencies:**
It is recommended to use a virtual environment.

```bash
python -m venv .venv
pip install -r requirements.txt
```

**Download the MediaPipe Model:**
Download the official hand\_landmarker.task file from Google and place it in the root directory of this project.

Download [hand\_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)

## 🎮 How to Use (The Pipeline)
Go to the project root directory
```bash
cd path\To\Project
```
Activate the virtual environment
```bash
.venv\Scripts\activate
```
Run the application:
```bash
python configMaker.py
```

## The 6-Step Workflow:
**Create/Load a Config:** Use the UI to define a Static Gesture (e.g., OPEN\_BROWSER\_READY), map it to a Dynamic Gesture, and assign an action string (e.g., OPEN chrome).

**Record Static Gestures:** Click Step 1 in the UI to open the webcam. Press t to toggle through your gestures and record baseline shape data and press q to end this step.

**Train Static Model:** Click Step 2 to instantly train the Random Forest gatekeeper model.

**Record Dynamic Gestures:** Use Step 3 to record your movement variances. Press q to end this step.

**Train Dynamic Gestures:** Use Step 4 to train the dynamic gesture recognition model.

**Test Pipeline (Live):** Click Step 5 to launch the live background listener. Flash your ready gesture to arm the system, perform the dynamic movement, and watch your OS execute the action!

Watch the following Demo for clarification:
[Demolink](https://drive.google.com/file/d/1_spa047kR2-w8GuJQvnPdbxNBbl5v_Gu/view?usp=sharing)

## 🧠 How the Architecture Works
**STATE 0 (IDLE):** The app monitors frames asynchronously. main.py extracts 21 3D hand landmarks, normalizes them into a 42-element vector, and passes them to a lightweight Random Forest.

**STATE 1 (ARMED):** If the Random Forest detects a "Ready" configuration, it triggers State 1. The app begins recording frame vectors into a sliding frame window (collections.deque).

**STATE 2 (TRIGGERED):** The app calculates the mathematical Euclidean distance between the newest and oldest frame in the buffer. If the shape variance breaches the TRIGGER\_THRESHOLD, the dynamic sequence is evaluated, the action string is sent to actions.py, and the system resets to Idle.
