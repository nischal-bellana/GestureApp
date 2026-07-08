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


def normalize_landmarks(landmark_list):
    """
    Normalizes landmarks to make them translation and scale invariant.
    1. Translation: Shifts the wrist (landmark 0) to (0, 0).
    2. Scale: Divides all points by the distance between the wrist and middle finger base.
    """
    # Convert landmarks to a 2D numpy array (X, Y)
    landmarks = np.array([[lm.x, lm.y] for lm in landmark_list])

    # 1. Translation Normalization (Subtract wrist coordinates from all points)
    wrist = landmarks[0]
    translated_landmarks = landmarks - wrist

    # 2. Scale Normalization (Calculate distance between wrist [0] and MCP of middle finger [9])
    # This distance acts as a baseline bounding ruler for the hand size.
    scale_factor = np.linalg.norm(translated_landmarks[9])

    # Avoid division by zero if tracking glitches
    if scale_factor == 0:
        scale_factor = 1.0

    normalized_landmarks = translated_landmarks / scale_factor

    # Flatten the array into a 1D vector of 42 values (x0, y0, x1, y1, ...) 
    # This format is perfect for feeding into a machine learning classifier.
    return normalized_landmarks.flatten()

GESTURE_LABEL = "OPEN_BROWSER"
NO_GESTURE = "NO_GESTURE"
gesture_label = GESTURE_LABEL 
def add_training_Data(feature_vector, gesture_type="static"):
    global gesture_label

    # Open a CSV file in append mode
    file_path = 'Training_Samples/gesture_data.csv'
    if gesture_type=="dynamic":
        file_path = 'Training_Samples/gesture_dynamic_data.csv'
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Prepend the label to the normalized features
        row = [gesture_label] + list(feature_vector)
        writer.writerow(row)
        
    print(f"Recorded 1 or 10 frames for {gesture_label}")

latest_result = None

def hand_detection_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def start_capture_stream(mode="sample", gesture_type="static"):
    print("args:", mode, gesture_type)
    # --- 1. STATE MACHINE SETUP ---
    STATE_IDLE = 0
    STATE_ARMED = 1

    current_state = STATE_IDLE
    frame_buffer = deque(maxlen=10)
    TRIGGER_THRESHOLD = 1.5

    gatekeeper_model_static = None
    gatekeeper_model_dynamic = None
    if mode=="test" or gesture_type=="dynamic":
        with open('static_gesture_model.pkl', 'rb') as f:
            gatekeeper_model_static = pickle.load(f)
        with open('dynamic_gesture_model.pkl', 'rb') as f:
            gatekeeper_model_dynamic = pickle.load(f)



    # --- MODEL INITIALIZATION (NEW API) ---
    # Ensure 'hand_landmarker.task' is downloaded and in your working directory
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

    # 0 accesses the default system webcam. Change to 1 or 2 if you have multiple cameras.
    cap = cv2.VideoCapture(0)

    # Optional: Request a lower hardware resolution to save USB bandwidth
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize our 1-second timer
    last_capture_time = time.time()
    capture_interval = 0.1  # 1 second

    print("Pipeline active (Tasks API). Point your hand at the camera...")
    print("Starting background capture. Press 'q' in the debug window to exit.")

    #mode == sample
    is_capturing = False
    lct_2 = time.time()
    global gesture_label

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

            # NEW: Calculate timestamp in milliseconds
            timestamp_ms = int(current_time * 1000)

            # NEW: Use detect_async instead of detect
            detector.detect_async(mp_image, timestamp_ms)

            # NEW: Process the result from the global variable (updated by the callback)
            if latest_result and latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                    feature_vector = normalize_landmarks(hand_landmarks)

                    if mode=="sample" and gesture_type=="static":
                        if is_capturing:
                            add_training_Data(feature_vector)
                    elif mode=="test" or gesture_type=="dynamic":
                        if current_state == STATE_IDLE:
                            prediction = gatekeeper_model_static.predict([feature_vector])[0]
                            if prediction == "OPEN_BROWSER_READY":
                                current_state = STATE_ARMED
                                frame_buffer.clear()
                                frame_buffer.append(feature_vector)
                                print("Ready gesture detected! Shifting to State 1 (Armed)...")
                        elif current_state == STATE_ARMED:
                            frame_buffer.append(feature_vector)
                    
                            # Only check for triggers once we have gathered a base history
                            if len(frame_buffer) == 10:
                                oldest_frame = frame_buffer[0]
                                current_frame = frame_buffer[-1]
                                
                                # Compute Euclidean distance between oldest frame shape and current frame shape
                                shape_distance = np.linalg.norm(current_frame - oldest_frame)
                                
                                print(f"\rTracking Motion... Shape Variance: {shape_distance:.3f} / Threshold: {TRIGGER_THRESHOLD}", end="")
                                
                                if shape_distance > TRIGGER_THRESHOLD:
                                    print(f"\n\n[STATE CHANGE] -> TRIGGERED!")
                                    print(f"Significant gesture change detected ({shape_distance:.2f} > {TRIGGER_THRESHOLD})")
                                    
                                    # --- STATE 2: ACTION EXECUTION ---
                                    # Extract the full 10-frame history matrix for your dynamic model
                                    sequence_to_classify = np.array(frame_buffer).flatten() 
                                    
                                    if mode=="sample":
                                        add_training_Data(sequence_to_classify, gesture_type)
                                    else:
                                        prediction = gatekeeper_model_dynamic.predict([sequence_to_classify])[0]
                                        if prediction == "OPEN_BROWSER": 
                                            print("OPEN BROWSER gesture detected!!")
                                            subprocess.Popen([r"C:\Program Files\Google\Chrome\Application\chrome.exe"])
                                    
                                    # Reset back to Idle post-execution
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
        # Drawing simple dots to avoid depending on old mediapipe drawing utilities
        h, w, _ = frame.shape

        
        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                for lm in hand_landmarks:
                    # The new API provides .x and .y as normalized floats between 0.0 and 1.0
                    cv2.circle(frame, (int((1-lm.x) * w), int(lm.y * h)), 4, (0, 255, 0), -1)

        if mode=="sample" and gesture_type=="static":
            cv2.circle(frame, (w-50, h-50), 10, (0, 255, 0) if is_capturing else (255, 0, 0), -1)


        cv2.imshow('Gesture Controller - Debug Feed', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
           break
        elif mode=="sample" and key == ord('t'):
            print("Toggled!")
            gesture_label = GESTURE_LABEL if gesture_label!=GESTURE_LABEL else NO_GESTURE


    # Clean up hardware resources when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_capture_stream(*sys.argv[1:])
