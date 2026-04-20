import os

import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from scripts.modeling import build_policy_model


def main():
    os.makedirs("model", exist_ok=True)

    data = np.load("data/hcr_dataset.npz")
    X, y = data["X"], data["y"]
    X = X.astype("float32") / 255.0

    print(f"Dataset: {X.shape}, Labels: {y.shape}")
    print(f"Distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}, 2={np.sum(y==2)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")

    model = build_policy_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            "model/policy_pretrained.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    model.save("model/policy_final.keras")
    print("Saved model/policy_final.keras")


if __name__ == "__main__":
    main()
