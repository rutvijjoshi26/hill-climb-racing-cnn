import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from scripts.modeling import (
    build_actor_critic_model,
    categorical_entropy,
    categorical_log_probs,
    copy_shared_weights,
)
from scripts.rl_env import HillClimbEnv
from scripts.rl_env import AUTO_RESET_KEYS


MODEL_DIR = Path("model")
PRETRAINED_POLICY_PATH = MODEL_DIR / "policy_pretrained.keras"
PPO_CHECKPOINT_PATH = MODEL_DIR / "ppo_actor_critic.keras"

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
ROLLOUT_STEPS = 256
PPO_EPOCHS = 6
MINIBATCH_SIZE = 32
TOTAL_UPDATES = 200
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
LEARNING_RATE = 2.5e-4
MAX_GRAD_NORM = 0.5


def compute_gae(rewards, dones, values, last_value):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * next_value * non_terminal - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * non_terminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def sample_action(logits):
    action = tf.random.categorical(logits, 1)[0, 0]
    log_prob = categorical_log_probs(logits, tf.convert_to_tensor([action]))[0]
    return int(action.numpy()), float(log_prob.numpy())


def collect_rollout(env, model, obs):
    observations = []
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []
    episode_rewards = []
    current_episode_reward = 0.0

    for _ in range(ROLLOUT_STEPS):
        batch_obs = np.expand_dims(obs, axis=0).astype(np.float32)
        outputs = model(batch_obs, training=False)
        logits = outputs["policy_logits"]
        value = float(outputs["value"][0, 0].numpy())

        action, log_prob = sample_action(logits)
        next_obs, reward, done, _ = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(float(done))
        values.append(value)
        log_probs.append(log_prob)

        current_episode_reward += reward
        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            obs = env.reset()
        else:
            obs = next_obs

    last_value = float(model(np.expand_dims(obs, axis=0), training=False)["value"][0, 0].numpy())

    batch = {
        "obs": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
        "log_probs": np.asarray(log_probs, dtype=np.float32),
    }
    batch["advantages"], batch["returns"] = compute_gae(
        batch["rewards"], batch["dones"], batch["values"], last_value
    )

    return obs, batch, episode_rewards


def ppo_update(model, optimizer, batch):
    obs = batch["obs"]
    actions = batch["actions"]
    old_log_probs = batch["log_probs"]
    returns = batch["returns"]
    advantages = batch["advantages"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset = tf.data.Dataset.from_tensor_slices(
        (obs, actions, old_log_probs, returns, advantages)
    ).shuffle(len(obs)).batch(MINIBATCH_SIZE)

    metrics = {"policy_loss": [], "value_loss": [], "entropy": []}

    for _ in range(PPO_EPOCHS):
        for obs_mb, actions_mb, old_log_probs_mb, returns_mb, advantages_mb in dataset:
            with tf.GradientTape() as tape:
                outputs = model(obs_mb, training=True)
                logits = outputs["policy_logits"]
                values = tf.squeeze(outputs["value"], axis=1)

                new_log_probs = categorical_log_probs(logits, actions_mb)
                ratio = tf.exp(new_log_probs - old_log_probs_mb)
                clipped_ratio = tf.clip_by_value(ratio, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO)

                surrogate_1 = ratio * advantages_mb
                surrogate_2 = clipped_ratio * advantages_mb
                policy_loss = -tf.reduce_mean(tf.minimum(surrogate_1, surrogate_2))

                value_loss = tf.reduce_mean(tf.square(returns_mb - values))
                entropy = tf.reduce_mean(categorical_entropy(logits))
                total_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            gradients = tape.gradient(total_loss, model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, MAX_GRAD_NORM)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            metrics["policy_loss"].append(float(policy_loss.numpy()))
            metrics["value_loss"].append(float(value_loss.numpy()))
            metrics["entropy"].append(float(entropy.numpy()))

    return {name: float(np.mean(values)) for name, values in metrics.items()}


def load_pretrained_weights(model):
    if not PRETRAINED_POLICY_PATH.exists():
        print("No behavioral cloning checkpoint found. PPO will start from scratch.")
        return

    pretrained_policy = keras.models.load_model(PRETRAINED_POLICY_PATH)
    copied = copy_shared_weights(pretrained_policy, model)
    print(f"Loaded pretrained policy weights into PPO model: {copied}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    model = build_actor_critic_model()
    load_pretrained_weights(model)
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    env = HillClimbEnv(auto_reset_keys=AUTO_RESET_KEYS)
    obs = env.reset()

    try:
        for update in range(1, TOTAL_UPDATES + 1):
            obs, batch, episode_rewards = collect_rollout(env, model, obs)
            losses = ppo_update(model, optimizer, batch)

            mean_reward = float(np.mean(episode_rewards)) if episode_rewards else float(np.sum(batch["rewards"]))
            print(
                f"Update {update:03d} | reward={mean_reward:.2f} "
                f"| policy_loss={losses['policy_loss']:.4f} "
                f"| value_loss={losses['value_loss']:.4f} "
                f"| entropy={losses['entropy']:.4f}"
            )

            if update % 10 == 0:
                model.save(PPO_CHECKPOINT_PATH)
                print(f"Saved PPO checkpoint to {PPO_CHECKPOINT_PATH}")
    finally:
        env.close()

    model.save(PPO_CHECKPOINT_PATH)
    print(f"Saved final PPO checkpoint to {PPO_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
