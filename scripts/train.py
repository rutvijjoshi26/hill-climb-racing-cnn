from tensorflow.keras import metrics
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# LOAD DATA
data = np.load("data/hcr_dataset.npz")
X, y = data["X"], data["y"]
# Normalize pixel values to 0-1
X = X.astype("float32") / 255.0
print(f"Dataset: {X.shape}, Labels: {y.shape}")
print(f"Distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}, 2={np.sum(y==2)}")

# SPLIT DATA
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# COMPUTE CLASS WEIGHTS
# This helps the model learn rare classes (like brake)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# BUILD MODEL
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        'model/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=[X_val, y_val],
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks 
)

model.save("model/final_model.h5")
print("Saved model/final_model.h5")