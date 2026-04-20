# Hill Climb Racing AI: Behavioral Cloning + PPO

This project trains a visual driving agent for *Hill Climb Racing* from raw `64x64` pixel frames. The training flow now matches the supervised-pretraining-then-RL pattern used in modern agent systems:

1. Record human gameplay and train a CNN policy with behavioral cloning.
2. Fine-tune that visual policy online with PPO.
3. Optimize against reward signals derived from forward progress and fuel efficiency proxies extracted directly from the game screen.

The project includes the full loop end to end: screen capture, preprocessing, action dispatch, reward extraction, PPO optimization, and a `pyautogui` failsafe.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Grant Accessibility permissions on macOS:
   - Open **System Settings > Privacy & Security > Accessibility**
   - Allow your terminal or IDE to control the computer

## Workflow

### Phase 1: Record Human Gameplay

Capture state-action pairs from your own driving:

```bash
python scripts/datacapture.py
```

- `D` = gas
- `A` = brake
- `ESC` stops capture
- Frames are saved in `data/raw_frames/`

Target at least `10K+` labeled frames if you want PPO to start from a useful policy instead of learning from scratch.

### Phase 2: Preprocess the Dataset

```bash
python scripts/preprocess.py
```

This converts the captured images into `data/hcr_dataset.npz`:

- UI is masked for policy learning
- Frames are resized to `64x64`
- Labels are preserved from the filename convention

### Phase 3: Behavioral Cloning Pretraining

```bash
python scripts/train.py
```

This trains the initial CNN policy and saves:

- `model/policy_pretrained.keras`
- `model/policy_final.keras`

### Phase 4: PPO Fine-Tuning

Open the game and run:

```bash
python scripts/ppo_finetune.py
```

The PPO loop:

- Captures the live game screen
- Feeds `64x64` pixel observations into the policy/value network
- Samples actions and dispatches them with `pyautogui`
- Computes rewards from forward progress and fuel-efficiency heuristics
- Updates the policy with PPO
- Saves checkpoints to `model/ppo_actor_critic.keras`
- Uses `AUTO_RESET_KEYS` in `scripts/rl_env.py` if you want episode resets to be automated

### Phase 5: Run the Fine-Tuned Agent

```bash
python scripts/run.py
```

- Loads `model/ppo_actor_critic.keras`
- Runs the PPO policy head in real time
- Press `q` in the preview window or `Ctrl+C` in the terminal to stop
- Moving the cursor to the top-left corner will trigger the `pyautogui` failsafe

## Reward Design

There is no native game API in this project, so PPO uses reward proxies extracted from the live screen:

- **Distance traveled proxy:** estimated from horizontal scene motion in a configurable crop
- **Fuel efficiency proxy:** progress normalized against observed fuel depletion
- **Survival bonus:** small positive reward each step
- **Failure penalty:** negative reward when the vehicle stagnates too long or the episode ends

This is enough to build a real RL fine-tuning loop, but the reward extraction is still heuristic. Expect to tune the reward regions for your machine and HUD layout.

## Configuration

If your game window moves or the HUD layout changes, update these constants:

- `CAPTURE_REGION` in [scripts/datacapture.py](/Users/rutvijjoshi/Projects/hill-climb-racing/scripts/datacapture.py)
- `CAPTURE_REGION` in [scripts/rl_env.py](/Users/rutvijjoshi/Projects/hill-climb-racing/scripts/rl_env.py)
- `FUEL_REGION` in [scripts/rl_env.py](/Users/rutvijjoshi/Projects/hill-climb-racing/scripts/rl_env.py)
- `PROGRESS_REGION` in [scripts/rl_env.py](/Users/rutvijjoshi/Projects/hill-climb-racing/scripts/rl_env.py)
- `AUTO_RESET_KEYS` in [scripts/rl_env.py](/Users/rutvijjoshi/Projects/hill-climb-racing/scripts/rl_env.py) if the game can be restarted from the keyboard

Example:

```python
CAPTURE_REGION = {"top": 75, "left": 17, "width": 1401, "height": 790}
```

## Model Design

- Input: `64x64x3` RGB frame
- Shared visual backbone: 3 convolution blocks plus a dense feature layer
- Behavioral cloning stage: categorical CNN policy
- RL stage: PPO actor-critic with policy logits and a scalar value head
- Actions: `0=Gas`, `1=Brake`, `2=None`
