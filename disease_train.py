
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# -------------------------------
# Configuration
# -------------------------------
DATASET_PATH = r"D:\crop\PlantVillage"
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 3

MODEL_PATH = "disease_model.h5"
LABELS_PATH = "disease_labels.json"

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_PATH}"
    )

# -------------------------------
# Load training / validation data
# -------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:", num_classes)
for i, name in enumerate(class_names):
    print(i, "->", name)

with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(
    1000, seed=SEED
).prefetch(AUTOTUNE)

val_ds = val_ds.cache().prefetch(AUTOTUNE)

# -------------------------------
# Data augmentation
# -------------------------------
augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1)
])

# -------------------------------
# Transfer learning model
# -------------------------------
base_model = MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
x = augmentation(inputs)
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(
    num_classes, activation="softmax"
)(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

print("\nStarting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Load the best saved model for final evaluation
best_model = tf.keras.models.load_model(MODEL_PATH)

val_loss, val_accuracy = best_model.evaluate(
    val_ds, verbose=1
)

print("\n========== FINAL RESULTS ==========")
print(f"Classes: {num_classes}")
print(f"Best validation accuracy: {val_accuracy * 100:.2f}%")
print(f"Validation loss: {val_loss:.4f}")
print(f"Model saved: {MODEL_PATH}")
print(f"Labels saved: {LABELS_PATH}")
print("===================================")
