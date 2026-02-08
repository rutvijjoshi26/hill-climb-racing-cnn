from pynput.keyboard import Listener
import os
import time
import numpy as np 
import mss
import cv2
from pynput import keyboard
from datetime import datetime

CAPTURE_REGION = {"top": 75, "left": 17, "width": 1401, "height": 790}
SAVE_DIR = "data/raw_frames"
FPS = 10

current_action = 2
running = True

def on_press(key):
    global current_action
    try:
        if key.char == 'd':
            current_action = 0
        elif key.char == 'a':
            current_action = 1
    except AttributeError:
        pass

def on_release(key):
    global current_action, running
    try:
        if key.char in ['a', 'd']:
            current_action = 2
    except AttributeError:
        pass
    if key == keyboard.Key.esc:
        running = False

def main():
    global running

    os.makedirs(SAVE_DIR, exist_ok=True)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("Starting capture in 3 second....Switch to Hill Climb Racing!")
    time.sleep(3)
    print("Capturing! Press Esc to stop.")

    frame_count = 0

    with mss.mss() as sct:
        while running:
            screenshot = sct.grab(CAPTURE_REGION)

            frame = np.array(screenshot)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

            filename = f"frame_{frame_count}_action_{current_action}.png"

            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            frame_count += 1
            time.sleep(1/ FPS)
    
    print(f"Captured {frame_count} frames!")

    listener.stop()

if __name__ == "__main__":
    main()