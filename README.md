# 🚗 Hill Climb Racing AI (Behavioral Cloning)

This project trains an AI to play *Hill Climb Racing* by watching you play. It uses a Convolutional Neural Network (CNN) to predict driving actions (Accelerate/Brake) from game screenshots in real-time.

## 🛠️ Setup

1. **Install Dependencies** (Optimized for macOS Apple Silicon)
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Uses `tensorflow-macos` and `tensorflow-metal` for M1/M2/M3 chips.*

2. **Grant Accessibility Permissions**
   - Go to **System Settings > Privacy & Security > Accessibility**
   - Allow your Terminal (or VS Code / IDE) to control the computer.
   - *Why?* The script needs to listen to your keys (data collection) and press keys (AI driving).

---

## 🚀 How to Run (Step-by-Step)

### Phase 1: data Collection 📸
Play the game and let the script record your moves.

1. Open **Hill Climb Racing**.
2. Run the capture script:
   ```bash
   python scripts/datacapture.py
   ```
3. Switch to the game window immediately.
4. **Play normally!**
   - Controls: `D` = Gas, `A` = Brake.
   - The script records screen frames + your key presses.
5. Press **ESC** to stop recording.
   - Frames are saved in `data/raw_frames/`.
   - *Goal:* Collect at least 5,000 - 10,000 frames for a good model.

### Phase 2: Preprocessing 🧹
Clean the data and prepare it for training.

1. Run the preprocessor:
   ```bash
   python scripts/preprocess.py
   ```
2. What it does:
   - Loads raw frames.
   - **Masks UI:** Blacks out fuel bar, scores, and pedals (so AI focuses on the hill).
   - **Resizes:** Downscales to 64x64 pixels.
   - Saves everything to `data/hcr_dataset.npz`.

### Phase 3: Train the Model
Train the Neural Network on your gameplay data.

1. Run the training script:
   ```bash
   python scripts/train.py
   ```
2. It will:
   - Load the dataset.
   - Train for up to 50 epochs (stops early if no improvement).
   - Save the best model to `model/best_model.h5`.
   - *Tip:* If accuracy is low (<60%), record more data!

### Phase 4: AI Driver (Inference)
Let the AI take the wheel!

1. Open **Hill Climb Racing**.
2. Run the driver script:
   ```bash
   python scripts/run.py
   ```
3. Switch to the game window.
4. Watch it drive!
5. **To Stop:** Move your mouse cursor to the top-left corner of the screen (failsafe) or press `Ctrl+C` in terminal.

---

## ⚙️ Configuration

If you move the game window or use a different screen, you MUST update the capture coordinates!

1. **Find Coordinates:**
   Use a screenshot tool or trial-and-error to find `top`, `left`, `width`, `height`.

2. **Update Files:**
   Update `CAPTURE_REGION` in **BOTH** files:
   - `scripts/datacapture.py`
   - `scripts/run.py`

   ```python
   # Example
   CAPTURE_REGION = {"top": 75, "left": 17, "width": 1401, "height": 790}
   ```

---

## 🔧 Troubleshooting

| Issue | Fix |
|-------|-----|
| **Keys not working** | Check Accessibility permissions in System Settings. Toggle off/on. |
| **"System Events" popup** | Allow the permission. Ensure you are using `pyautogui` (in run.py). |
| **AI drives off cliffs** | You need more training data! Record more "recovery" moves. |
| **Model prediction error** | Ensure `IMG_SIZE` is 64 in all scripts. |

---

## 🧠 Model Architecture

- **Input:** 64x64 RGB Image
- **Layers:** 3x Conv2D + MaxPooling (Feature Extraction) -> Flatten -> Dense (Decision)
- **Output:** 3 Actions (0=Gas, 1=Brake, 2=None)
