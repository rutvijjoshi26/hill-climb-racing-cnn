import time

import cv2
import mss
import numpy as np
import pyautogui
import tensorflow as tf

from scripts.rl_env import CAPTURE_REGION, preprocess_observation


MODEL_PATH = "model/ppo_actor_critic.keras"


def predict_action(model, frame):
    obs = preprocess_observation(frame)
    obs = np.expand_dims(obs, axis=0)
    outputs = model(obs, training=False)
    logits = outputs["policy_logits"].numpy()[0]
    policy = tf.nn.softmax(logits).numpy()
    action = int(np.argmax(policy))
    confidence = float(np.max(policy))
    return action, confidence


def main():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded!")

    print("Starting in 3 seconds... Switch to game!")
    time.sleep(3)

    with mss.mss() as sct:
        while True:
            screenshot = sct.grab(CAPTURE_REGION)
            frame = np.array(screenshot)
            action, confidence = predict_action(model, frame)

            if action == 0:
                pyautogui.keyDown("d")
                pyautogui.keyUp("a")
                action_str = "ACCELERATE"
            elif action == 1:
                pyautogui.keyDown("a")
                pyautogui.keyUp("d")
                action_str = "BRAKE"
            else:
                pyautogui.keyUp("d")
                pyautogui.keyUp("a")
                action_str = "NONE"

            print(f"Action: {action_str} ({confidence:.2f})")

            display_frame = cv2.resize(frame, (400, 300))
            cv2.putText(
                display_frame,
                f"Action: {action_str}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                f"Conf: {confidence:.2f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("AI Vision", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(0.12)

    pyautogui.keyUp("a")
    pyautogui.keyUp("d")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
