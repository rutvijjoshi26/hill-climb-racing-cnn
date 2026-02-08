import os
import cv2
import numpy as np 
from glob import glob

RAW_DIR = "data/raw_frames"
OUTPUT_FILE = "data/hcr_dataset.npz"
IMG_SIZE = 64

def mask_ui(frame):
    h, w = frame.shape[:2]
    frame[0:int(h*0.20), :] = 0    # Top
    frame[int(h*0.65):, :] = 0     # Bottom
    return frame

def main():
    files = sorted(glob(os.path.join(RAW_DIR, "*.png")))
    print(f"Found {len(files)} frames")

    X = []
    y = []

    for filepath in files:
        filename = os.path.basename(filepath)

        label = int(filename.split("_")[3].replace(".png", ""))

        frame = cv2.imread(filepath)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = mask_ui(frame)

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        X.append(frame)
        y.append(label)
    
    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.int32)

    print(f"Dataset shape: X = {X.shape}, y = {y.shape}")
    print(f"Label distribution: 0 = {np.sum(y==0)}, 1 = {np.sum(y==1)}, 2 = {np.sum(y==2)}")

    np.savez_compressed(OUTPUT_FILE, X=X, y=y)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
