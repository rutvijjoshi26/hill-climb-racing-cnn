import time
import cv2
import mss
import numpy as np
import tensorflow as tf
import pyautogui

# === CONFIGURATION ===
# MUST match what you used in training!
CAPTURE_REGION = {"top": 75, "left": 17, "width": 1401, "height": 790}
IMG_SIZE = 64
MODEL_PATH = "model/best_model.h5"

# Control keys

def mask_ui(frame):
    # match preprocess.py!
    h, w = frame.shape[:2]
    frame[0:int(h*0.20), :] = 0    # Top
    frame[int(h*0.65):, :] = 0     # Bottom
    return frame

def predict_action(model, frame):
    # Preprocess
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img = mask_ui(img)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0) # Add batch dimension (1, 64, 64, 3)
    
    # Predict (Optimized)
    predictions = model(img, training=False).numpy()
    action = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return action, confidence

def main():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded!")
    
    print("Starting in 3 seconds... Switch to game!")
    time.sleep(3)
    
    running = True
    
    with mss.mss() as sct:
        while running:
            start_time = time.time()
            
            # 1. Capture
            screenshot = sct.grab(CAPTURE_REGION)
            frame = np.array(screenshot)
            
            # 2. Predict
            action, confidence = predict_action(model, frame)
            
            # 3. Act
            if action == 0:   # Accelerate (D)
                pyautogui.keyDown('d')
                pyautogui.keyUp('a')
                action_str = "ACCELERATE"
            elif action == 1: # Brake (A)
                pyautogui.keyDown('a')
                pyautogui.keyUp('d')
                action_str = "BRAKE"
            else:             # None
                pyautogui.keyUp('d')
                pyautogui.keyUp('a')
                action_str = "NONE"
            
            print(f"Action: {action_str} ({confidence:.2f})")
            
            # 4. Display (Overlay)
            display_frame = cv2.resize(frame, (400, 300))
            cv2.putText(display_frame, f"Action: {action_str}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Conf: {confidence:.2f}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("AI Vision", display_frame)
            
            # Stop if 'q' is pressed in the preview window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()