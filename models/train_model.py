import os
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# Allow importing emotion_model.py
sys.path.append(os.path.dirname(__file__))

from emotion_model import build_model

# -----------------------
# Dataset Paths
# -----------------------

TRAIN_DIR = os.path.join("dataset", "fer2013", "train")
TEST_DIR = os.path.join("dataset", "fer2013", "test")

IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 20

# -----------------------
# Data Augmentation
# -----------------------

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# -----------------------
# Load Datas
# -----------------------

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\nEmotion Classes:")
print(train_generator.class_indices)

# -----------------------
# Build Model
# -----------------------

model = build_model()

# -----------------------
# Callbacks
# -----------------------

checkpoint = ModelCheckpoint(
    "models/emotion_cnn.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    verbose=1
)

# -----------------------
# Train Model
# -----------------------

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)

# -----------------------
# Save Final Model
# -----------------------

model.save("models/emotion_cnn.keras")

print("\nModel Saved Successfully!")

# -----------------------
# Evaluation
# -----------------------

loss, accuracy = model.evaluate(test_generator)

print(f"\nTest Accuracy : {accuracy*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

# -----------------------
# Accuracy Graph
# -----------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("accuracy_graph.png")
plt.show()

# -----------------------
# Loss Graph
# -----------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("loss_graph.png")
plt.show()

print("\nTraining Completed Successfully!")