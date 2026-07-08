import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import csv
import pickle
import sys
from collections import deque
import subprocess
import json


def normalize_landmarks(landmark_list):
    """
    Ensures consistent gesture recognition regardless of hand distance from the camera or hand size variations.
    """
    landmarks = np.array([[lm.x, lm.y] for lm in landmark_list])

    # Aligns all hand shapes to a common origin to prevent spatial screen positioning from skewing the classification
    wrist = landmarks[0]
    translated_landmarks = landmarks - wrist

    # Establishes a reference length based on hand proportions to neutralize depth and hand-size discrepancies
    # This distance acts as a baseline bounding ruler for the hand size.
    scale_factor = np.linalg.norm(translated_landmarks[9])

    # Avoid division by zero if tracking glitches
    if scale_factor == 0:
        scale_factor = 1.0

    normalized_landmarks = translated_landmarks / scale_factor

    # ML models (like SVM/RandomForest) require flat feature vectors rather than multi-dimensional spatial arrays
    return normalized_landmarks.flatten()

static_labels = []
static_sel = 0
dynamic_labels = {}
dynamic_sel = 0
actions = {}

def add_training_Data(feature_vector, config_name, gesture_type="static"):
    global static_labels
    global static_sel
    global dynamic_labels
    global dynamic_sel

    file_path = f'Training_Samples/{config_name}_{gesture_type}_samples.csv'
    gesture_label = ""

    if gesture_type=="static":
        gesture_label = static_labels[static_sel]
    else:
        gesture_label = dynamic_labels.get(static_labels[static_sel])[dynamic_sel]

    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Attaches the ground-truth label necessary for supervised training before saving the sample
        row = [gesture_label] + list(feature_vector)
        writer.writerow(row)
    
    if gesture_type=="static":
        print(f"Recorded frame for {gesture_label}")
    else:
        print(f"Recorded frames for {gesture_label}")

latest_result = None

def hand_detection_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def start_capture_stream(config_name, mode="sample", gesture_type="static"):
    print("args:", mode, gesture_type, config_name)
    config_data = None
    with open(f"Configs/{config_name}.json","r") as file:
        config_data = json.load(file)


    # --- STATE MACHINE SETUP ---
    STATE_IDLE = 0
    STATE_ARMED = 1

    current_state = STATE_IDLE
    frame_buffer = deque(maxlen=10)
    TRIGGER_THRESHOLD = 1.5

    gatekeeper_model_static = None
    gatekeeper_model_dynamic = None

    if mode=="test" or gesture_type=="dynamic":
        with open(f'Models/{config_name}_static_model.pkl', 'rb') as f:
            gatekeeper_model_static = pickle.load(f)
    if mode=="test":
        with open(f'Models/{config_name}_dynamic_model.pkl', 'rb') as f:
            gatekeeper_model_dynamic = pickle.load(f)

    # --- MODEL INITIALIZATION ---
    # Ensure 'hand_landmarker.task' is downloaded and in your working directory for the pipeline to function
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=hand_detection_callback,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Targets the primary system camera; alternative indices map to external hardware inputs
    cap = cv2.VideoCapture(0)

    # Optional: Request a lower hardware resolution to save USB bandwidth and reduce processing overhead
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_capture_time = time.time()
    capture_interval = 0.1  

    print("Pipeline active (Tasks API). Point your hand at the camera...")
    print("Starting background capture. Press 'q' in the debug window to exit.")

    is_capturing = False
    lct_2 = time.time()

    global static_labels
    global static_sel
    global dynamic_labels
    global dynamic_sel
    global actions

    static_labels = list(config_data.get("labels").keys())
    if mode=="sample" and gesture_type=="static":
        static_labels.append("NO_GESTURE")
    dynamic_labels = config_data.get("labels")
    actions = config_data.get("actions")

    while True:
        # Constantly read frames so the hardware buffer doesn't back up with old images
        success, frame = cap.read()

        if not success:
            print("Error: Could not connect to webcam.")
            break

        current_time = time.time()

        if current_time - last_capture_time >= capture_interval:

            optimized_frame = cv2.resize(frame, (320, 240))
            optimized_frame = cv2.flip(optimized_frame, 1)

            rgb_frame = cv2.cvtColor(optimized_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(current_time * 1000)

            # detect_async prevents the main vision pipeline from blocking the main thread during ML inference
            detector.detect_async(mp_image, timestamp_ms)

            if latest_result and latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                    feature_vector = normalize_landmarks(hand_landmarks)

                    if mode=="sample" and gesture_type=="static":
                        if is_capturing:
                            add_training_Data(feature_vector, config_name)
                    elif mode=="test" or gesture_type=="dynamic":
                        if current_state == STATE_IDLE:
                            prediction = gatekeeper_model_static.predict([feature_vector])[0]

                            if (mode=="test" and prediction in static_labels) or (prediction == static_labels[static_sel]):
                                current_state = STATE_ARMED
                                frame_buffer.clear()
                                frame_buffer.append(feature_vector)
                                print(f"{prediction} gesture detected! Shifting to State 1 (Armed)...")
                        elif current_state == STATE_ARMED:
                            frame_buffer.append(feature_vector)
                    
                            # Only check for triggers once we have gathered a base history to compare against
                            if len(frame_buffer) == 10:
                                oldest_frame = frame_buffer[0]
                                current_frame = frame_buffer[-1]
                                
                                # Measures the magnitude of spatial distortion over the timeframe to determine if a dynamic movement occurred
                                shape_distance = np.linalg.norm(current_frame - oldest_frame)
                                
                                print(f"\rTracking Motion... Shape Variance: {shape_distance:.3f} / Threshold: {TRIGGER_THRESHOLD}", end="")
                                
                                if shape_distance > TRIGGER_THRESHOLD:
                                    print(f"\n\n[STATE CHANGE] -> TRIGGERED!")
                                    print(f"Significant gesture change detected ({shape_distance:.2f} > {TRIGGER_THRESHOLD})")
                                    
                                    # --- STATE 2: ACTION EXECUTION ---
                                    sequence_to_classify = np.array(frame_buffer).flatten() 
                                    
                                    if mode=="sample":
                                        add_training_Data(sequence_to_classify, config_name, gesture_type)
                                    else:
                                        dyn_prediction = gatekeeper_model_dynamic.predict([sequence_to_classify])[0]
                                        if dyn_prediction in dynamic_labels.get(prediction):
                                            print(f"{dyn_prediction} gesture detected!!")
                                            print(f"Action: {actions.get(dyn_prediction)}")
                                            if actions.get(dyn_prediction) == "OPEN BROWSER":
                                                subprocess.Popen(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
                                            elif actions.get(dyn_prediction) == "CLOSE WINDOW":
                                                pyautogui.hotkey('alt', 'f4')
                                    
                                    # Prevents double-triggering the action by enforcing a cooldown/reset cycle
                                    print("Action completed. Returning to STATE 0 (IDLE).")
                                    current_state = STATE_IDLE
                                    frame_buffer.clear()
                    print(f"[{time.strftime('%H:%M:%S')}] Hand Detected!")
            else:
               if current_state == STATE_ARMED:
                    print("\n[RESET] Hand lost. Returning to STATE 0 (IDLE).")
                    current_state = STATE_IDLE
                    frame_buffer.clear() 

            if mode=="sample" and gesture_type=="static":
                if current_time - lct_2 >= 2:
                    is_capturing = not (is_capturing)
                    lct_2 = current_time

            last_capture_time = current_time

        # --- DEBUG FEED ---
        # Drawing simple dots to avoid depending on older, heavier mediapipe drawing utilities
        h, w, _ = frame.shape
        
        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                for lm in hand_landmarks:
                    # Requires scaling up by frame dimensions to map the network's normalized coordinate space back to pixel space
                    cv2.circle(frame, (int((1-lm.x) * w), int(lm.y * h)), 4, (0, 255, 0), -1)

        if mode=="sample" and gesture_type=="static":
            cv2.circle(frame, (w-50, h-50), 10, (0, 255, 0) if is_capturing else (255, 0, 0), -1)

        cv2.imshow('Gesture Controller - Debug Feed', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
           break
        elif mode=="sample" and key == ord('t'):
            if gesture_type=="static":
                static_sel += 1
                static_sel %= len(static_labels)
                print(f"Selected {static_labels[static_sel]}")
            else:
                dynamic_sel += 1
                if dynamic_sel >= len(dynamic_labels.get(static_labels[static_sel])):
                    dynamic_sel = 0
                    static_sel += 1
                    static_sel %= len(static_labels)
                print(f"Selected {dynamic_labels.get(static_labels[static_sel])[dynamic_sel]}")

    # Prevents memory leaks and explicitly releases the hardware lock on the camera device
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_capture_stream(*sys.argv[1:])
