import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMG_SIZE = 64
NUM_ACTIONS = 3


def _encoder_body(inputs):
    x = layers.Conv2D(32, (3, 3), activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="shared_dense")(x)
    return layers.Dropout(0.3, name="shared_dropout")(x)


def build_policy_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_actions=NUM_ACTIONS):
    inputs = keras.Input(shape=input_shape, name="image")
    features = _encoder_body(inputs)
    logits = layers.Dense(num_actions, name="policy_logits")(features)
    outputs = layers.Activation("softmax", name="policy")(logits)
    return keras.Model(inputs=inputs, outputs=outputs, name="hill_climb_policy")


def build_actor_critic_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_actions=NUM_ACTIONS):
    inputs = keras.Input(shape=input_shape, name="image")
    features = _encoder_body(inputs)
    policy_logits = layers.Dense(num_actions, name="policy_logits")(features)
    value = layers.Dense(1, name="value")(features)
    return keras.Model(
        inputs=inputs,
        outputs={"policy_logits": policy_logits, "value": value},
        name="hill_climb_actor_critic",
    )


def copy_shared_weights(source_model, target_model):
    source_layers = {layer.name: layer for layer in source_model.layers}
    copied = []

    for target_layer in target_model.layers:
        source_layer = source_layers.get(target_layer.name)
        if source_layer is None:
            continue

        source_weights = source_layer.get_weights()
        target_weights = target_layer.get_weights()
        if not source_weights or len(source_weights) != len(target_weights):
            continue

        try:
            target_layer.set_weights(source_weights)
            copied.append(target_layer.name)
        except ValueError:
            continue

    return copied


def categorical_log_probs(logits, actions):
    log_probs = tf.nn.log_softmax(logits)
    action_mask = tf.one_hot(actions, depth=NUM_ACTIONS)
    return tf.reduce_sum(log_probs * action_mask, axis=1)


def categorical_entropy(logits):
    log_probs = tf.nn.log_softmax(logits)
    probs = tf.nn.softmax(logits)
    return -tf.reduce_sum(probs * log_probs, axis=1)
