"""
=============================================================
  NeuroScan AI — CNN Training Script
  Dataset : Data/ with 3 classes:
              - Mild Dementia
              - Moderate Dementia
              - Non Demented
  Run     : python train_model.py
  Output  : models/cnn_model.h5
=============================================================
"""
import os, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# ── Config ────────────────────────────────────────────────────
DATA_DIR   = "Data"          # folder extracted from Data__2_.zip
MODEL_PATH = "models/cnn_model.h5"
IMG_SIZE   = 64
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 0.001
SEED       = 42

CLASSES = ["Mild Dementia", "Moderate Dementia", "Non Demented"]
NUM_CLASSES = len(CLASSES)

os.makedirs("models", exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Load Dataset ──────────────────────────────────────────────
print("=" * 55)
print("  NeuroScan AI — CNN Training")
print("=" * 55)
print(f"\nLoading dataset from: {DATA_DIR}")

X, y = [], []
for label_idx, cls in enumerate(CLASSES):
    cls_path = os.path.join(DATA_DIR, cls)
    if not os.path.exists(cls_path):
        print(f"  [WARNING] Not found: {cls_path}")
        continue
    files = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg",".jpeg",".png"))]
    print(f"  {cls}: {len(files)} images")
    for fname in files:
        try:
            img = Image.open(os.path.join(cls_path, fname)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            X.append(np.array(img, dtype=np.float32) / 255.0)
            y.append(label_idx)
        except Exception as e:
            pass

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
print(f"\nTotal: {len(X)} images  |  Shape: {X.shape}")

# ── Class Weights (handle imbalance) ─────────────────────────
class_weights_arr = compute_class_weight("balanced", classes=np.unique(y), y=y)
class_weights = dict(enumerate(class_weights_arr))
print(f"\nClass weights: { {CLASSES[k]: round(v,3) for k,v in class_weights.items()} }")

# ── Split ─────────────────────────────────────────────────────
y_cat = to_categorical(y, NUM_CLASSES)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=SEED)

print(f"\nTrain: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

# ── Data Augmentation ─────────────────────────────────────────
aug_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ── EDA Plots ─────────────────────────────────────────────────
print("\nGenerating EDA plots...")
counts = [np.sum(y == i) for i in range(NUM_CLASSES)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(CLASSES, counts, color=["#E74C3C", "#F39C12", "#2ECC71"])
axes[0].set_title("Class Distribution")
axes[0].tick_params(axis="x", rotation=15)
for i, c in enumerate(counts):
    axes[0].text(i, c + 50, str(c), ha="center", fontweight="bold")

# Sample images
n_sample = 3
for row_idx, cls_idx in enumerate(range(NUM_CLASSES)):
    idxs = np.where(y == cls_idx)[0][:n_sample]
for i, cls_idx in enumerate(range(NUM_CLASSES)):
    axes[1].text(0.02, 0.9 - i*0.3, f"{CLASSES[cls_idx]}: {counts[cls_idx]} images",
                 transform=axes[1].transAxes, fontsize=12,
                 color=["#E74C3C","#F39C12","#2ECC71"][i])
axes[1].set_title("Dataset Summary"); axes[1].axis("off")
plt.suptitle("EDA — Alzheimer MRI Dataset", fontweight="bold")
plt.tight_layout()
plt.savefig("models/eda_plot.png", dpi=120)
plt.close()
print("  Saved: models/eda_plot.png")

# ── Build CNN Model ───────────────────────────────────────────
def build_cnn():
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        Conv2D(32, (3,3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3,3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3,3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(256, (3,3), activation="relu", padding="same", kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(
        optimizer=Adam(LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_cnn()
model.summary()

# ── Callbacks ─────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
]

# ── Train ─────────────────────────────────────────────────────
print(f"\nTraining CNN for up to {EPOCHS} epochs...")
t0 = time.time()

history = model.fit(
    aug_gen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=SEED),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weights
)

elapsed = time.time() - t0
print(f"\nTraining complete in {elapsed:.1f}s")

# ── Evaluate ──────────────────────────────────────────────────
best_model = tf.keras.models.load_model(MODEL_PATH)
y_pred_prob = best_model.predict(X_test, verbose=0)
y_pred = y_pred_prob.argmax(axis=1)
y_true = y_test.argmax(axis=1)

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test Loss     : {test_loss:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# ── Save Plots ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(history.history["loss"],     label="Train Loss",  color="#3498DB")
axes[0].plot(history.history["val_loss"], label="Val Loss",    color="#E74C3C")
axes[0].set_title("Loss Curve"); axes[0].set_xlabel("Epoch")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(history.history["accuracy"],     label="Train Acc", color="#2ECC71")
axes[1].plot(history.history["val_accuracy"], label="Val Acc",   color="#F39C12")
axes[1].set_title("Accuracy Curve"); axes[1].set_xlabel("Epoch")
axes[1].legend(); axes[1].grid(True)
plt.suptitle("Training History", fontweight="bold")
plt.tight_layout()
plt.savefig("models/training_history.png", dpi=120)
plt.close()
print("\nSaved: models/training_history.png")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=120)
plt.close()
print("Saved: models/confusion_matrix.png")

print(f"\n{'='*55}")
print(f"  Model saved to : {MODEL_PATH}")
print(f"  Test Accuracy  : {test_acc:.4f}")
print(f"{'='*55}")
