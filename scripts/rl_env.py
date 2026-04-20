import time
from dataclasses import dataclass

import cv2
import mss
import numpy as np
import pyautogui


CAPTURE_REGION = {"top": 75, "left": 17, "width": 1401, "height": 790}
IMG_SIZE = 64

# Ratios inside CAPTURE_REGION. Tune these for your HUD layout if needed.
FUEL_REGION = (0.77, 0.03, 0.97, 0.09)
PROGRESS_REGION = (0.12, 0.32, 0.88, 0.72)
AUTO_RESET_KEYS = []


def mask_ui(frame):
    masked = frame.copy()
    h, _ = masked.shape[:2]
    masked[0:int(h * 0.20), :] = 0
    masked[int(h * 0.65):, :] = 0
    return masked


def preprocess_observation(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    masked = mask_ui(rgb)
    resized = cv2.resize(masked, (IMG_SIZE, IMG_SIZE))
    return resized.astype("float32") / 255.0


def crop_by_ratio(frame, region):
    h, w = frame.shape[:2]
    x0 = int(region[0] * w)
    y0 = int(region[1] * h)
    x1 = int(region[2] * w)
    y1 = int(region[3] * h)
    return frame[y0:y1, x0:x1]


def estimate_fuel_level(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    fuel_crop = crop_by_ratio(rgb, FUEL_REGION)
    hsv = cv2.cvtColor(fuel_crop, cv2.COLOR_RGB2HSV)

    green_mask = cv2.inRange(hsv, (35, 60, 50), (95, 255, 255))
    yellow_mask = cv2.inRange(hsv, (15, 60, 50), (40, 255, 255))
    mask = cv2.bitwise_or(green_mask, yellow_mask)

    column_activity = (mask > 0).mean(axis=0)
    active_columns = column_activity > 0.2
    if not np.any(active_columns):
        return 0.0

    filled_columns = np.where(active_columns)[0]
    return float((filled_columns[-1] - filled_columns[0] + 1) / len(active_columns))


def estimate_forward_progress(prev_frame, curr_frame):
    prev_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGRA2RGB)
    curr_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGRA2RGB)

    prev_crop = crop_by_ratio(prev_rgb, PROGRESS_REGION)
    curr_crop = crop_by_ratio(curr_rgb, PROGRESS_REGION)

    prev_gray = cv2.cvtColor(prev_crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    curr_gray = cv2.cvtColor(curr_crop, cv2.COLOR_RGB2GRAY).astype(np.float32)

    shift, response = cv2.phaseCorrelate(prev_gray, curr_gray)
    horizontal_shift = -shift[0]
    if response < 0.02:
        return 0.0
    return float(max(horizontal_shift, 0.0))


@dataclass
class RewardSignals:
    progress_pixels: float
    fuel_level: float
    fuel_delta: float
    efficiency_bonus: float


class HillClimbEnv:
    def __init__(
        self,
        step_delay=0.12,
        max_steps=1200,
        stagnation_limit=45,
        auto_reset_keys=None,
        reset_delay=2.5,
    ):
        self.step_delay = step_delay
        self.max_steps = max_steps
        self.stagnation_limit = stagnation_limit
        self.auto_reset_keys = AUTO_RESET_KEYS if auto_reset_keys is None else auto_reset_keys
        self.reset_delay = reset_delay

        pyautogui.FAILSAFE = True

        self.sct = mss.mss()
        self.prev_frame = None
        self.prev_fuel_level = None
        self.steps = 0
        self.stagnation_steps = 0

    def _capture_frame(self):
        return np.array(self.sct.grab(CAPTURE_REGION))

    def _release_controls(self):
        pyautogui.keyUp("a")
        pyautogui.keyUp("d")

    def _apply_action(self, action):
        if action == 0:
            pyautogui.keyDown("d")
            pyautogui.keyUp("a")
        elif action == 1:
            pyautogui.keyDown("a")
            pyautogui.keyUp("d")
        else:
            self._release_controls()

    def _compute_reward(self, frame, action):
        progress = estimate_forward_progress(self.prev_frame, frame) if self.prev_frame is not None else 0.0
        fuel_level = estimate_fuel_level(frame)
        fuel_delta = 0.0 if self.prev_fuel_level is None else max(self.prev_fuel_level - fuel_level, 0.0)

        survival_reward = 0.05
        control_penalty = {0: 0.01, 1: 0.015, 2: 0.0}[action]
        progress_reward = min(progress * 0.04, 2.0)
        efficiency_bonus = 0.0

        if progress > 0.5:
            self.stagnation_steps = 0
            efficiency_bonus = min(progress / max(fuel_delta, 0.02), 25.0) * 0.01
        else:
            self.stagnation_steps += 1

        if fuel_level < 0.05:
            efficiency_bonus -= 0.25

        reward = survival_reward + progress_reward + efficiency_bonus - control_penalty
        return reward, RewardSignals(
            progress_pixels=progress,
            fuel_level=fuel_level,
            fuel_delta=fuel_delta,
            efficiency_bonus=efficiency_bonus,
        )

    def reset(self):
        self._release_controls()
        for key in self.auto_reset_keys:
            pyautogui.press(key)
            time.sleep(0.35)

        time.sleep(self.reset_delay)
        frame = self._capture_frame()
        self.prev_frame = frame
        self.prev_fuel_level = estimate_fuel_level(frame)
        self.steps = 0
        self.stagnation_steps = 0
        return preprocess_observation(frame)

    def step(self, action):
        self._apply_action(action)
        time.sleep(self.step_delay)

        frame = self._capture_frame()
        obs = preprocess_observation(frame)
        reward, signals = self._compute_reward(frame, action)

        self.steps += 1
        done = self.steps >= self.max_steps or self.stagnation_steps >= self.stagnation_limit
        if done:
            reward -= 1.0

        info = {
            "progress_pixels": signals.progress_pixels,
            "fuel_level": signals.fuel_level,
            "fuel_delta": signals.fuel_delta,
            "efficiency_bonus": signals.efficiency_bonus,
            "stagnation_steps": self.stagnation_steps,
        }

        self.prev_frame = frame
        self.prev_fuel_level = signals.fuel_level
        return obs, float(reward), done, info

    def close(self):
        self._release_controls()
        self.sct.close()
