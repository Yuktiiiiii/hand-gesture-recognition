"""
MODEL 1 - MLP BASELINE
Static Hand Gesture Recognition - ICT-4442

Lightweight version aligned with the CNN and MobileNetV2 experiments:
- 64 x 64 grayscale images
- same subject-wise split: 00-06 train, 07 validation, 08-09 test
- default 20 images per subject/gesture class
- memory-conscious tf.data pipeline (no full-dataset cache)

Run:
    python model1_mlp.py --data_dir ".\\leapGestRecog"

Optional:
    python model1_mlp.py --data_dir ".\\leapGestRecog" --use_hog
    python model1_mlp.py --data_dir ".\\leapGestRecog" --limit_per_class 20
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import time
import hashlib
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_SIZE = 64
NUM_CLASSES = 10
SEED = 42
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
N_TRAIN_SUBJ, N_VAL_SUBJ, N_TEST_SUBJ = 7, 1, 2
DEFAULT_LIMIT_PER_CLASS = 20
BATCH_SIZE = 16

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_metadata(data_dir, out_dir, limit_per_class=DEFAULT_LIMIT_PER_CLASS):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"No images found under {data_dir}")

    rows, removed, seen, dups = [], [], {}, []
    for p in files:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
        except Exception as e:
            removed.append({"path": str(p), "reason": f"unreadable: {e}"})
            continue

        digest = hashlib.md5(p.read_bytes()).hexdigest()
        if digest in seen:
            dups.append({"path": str(p), "duplicate_of": seen[digest]})
        else:
            seen[digest] = str(p)

        rows.append({
            "path": str(p),
            "subject": p.parent.parent.name,
            "gesture": p.parent.name
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No readable image files were found.")

    pd.DataFrame(removed, columns=["path", "reason"]).to_csv(
        out_dir / "removed_files.csv", index=False
    )
    pd.DataFrame(dups, columns=["path", "duplicate_of"]).to_csv(
        out_dir / "duplicate_files.csv", index=False
    )

    class_names = sorted(df["gesture"].unique())
    if len(class_names) != NUM_CLASSES:
        raise SystemExit(
            f"Expected {NUM_CLASSES} gesture classes, found {len(class_names)}: {class_names}"
        )
    class_to_index = {name: i for i, name in enumerate(class_names)}
    df["label"] = df["gesture"].map(class_to_index)

    # Match the CNN/MobileNet sampling convention: up to 20 images
    # from every subject/class group, sampled reproducibly.
    if limit_per_class is not None and limit_per_class > 0:
        df = (
            df.groupby(["subject", "label"], group_keys=False)
              .apply(lambda g: g.sample(
                  n=min(len(g), limit_per_class), random_state=SEED
              ))
              .reset_index(drop=True)
        )

    subjects = sorted(df["subject"].unique())
    if len(subjects) < N_TRAIN_SUBJ + N_VAL_SUBJ + N_TEST_SUBJ:
        raise SystemExit(
            f"Expected at least 10 subject folders for the 7/1/2 split; found {len(subjects)}."
        )

    train_subjects = subjects[:7]
    val_subjects = subjects[7:8]
    test_subjects = subjects[8:10]
    split_map = {
        **{s: "train" for s in train_subjects},
        **{s: "val" for s in val_subjects},
        **{s: "test" for s in test_subjects},
    }
    df["split"] = df["subject"].map(split_map)

    print(
        f"[data] images={len(df)}; corrupt removed={len(removed)}; "
        f"duplicates logged={len(dups)}"
    )
    print(f"[data] train subjects={train_subjects}")
    print(f"[data] validation subjects={val_subjects}")
    print(f"[data] test subjects={test_subjects}")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))

    df.to_csv(out_dir / "metadata.csv", index=False)
    (out_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))
    return df, class_names


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=1, expand_animations=False)
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.float32)
    return image, label


def make_dataset(df, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((
        df["path"].values,
        df["label"].values.astype("int32")
    ))
    ds = ds.map(load_image, num_parallel_calls=1)
    if shuffle:
        ds = ds.shuffle(min(len(df), 1000), seed=SEED,
                        reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    return ds.prefetch(1)


def augmentation_block():
    return keras.Sequential([
        layers.RandomRotation(0.03),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomZoom(0.10),
    ], name="augmentation")


def build_mlp():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = augmentation_block()(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="mlp_baseline")


def get_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7
        ),
    ]


def plot_history(history, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history.get("accuracy", []), label="train")
    axes[0].plot(history.history.get("val_accuracy", []), label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history.get("loss", []), label="train")
    axes[1].plot(history.history.get("val_loss", []), label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Lightweight MLP baseline")
    parser.add_argument("--data_dir", required=True,
                        help="Path to leapGestRecog dataset folder")
    parser.add_argument("--out_dir", default="outputs",
                        help="Folder for saved model and results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit_per_class", type=int,
                        default=DEFAULT_LIMIT_PER_CLASS,
                        help="Images sampled per subject/gesture; use 20 to match CNN/MobileNet")
    args = parser.parse_args()

    if args.batch_size < 1 or args.epochs < 1:
        raise SystemExit("--batch_size and --epochs must be positive integers.")

    output_dir = Path(args.out_dir) / "mlp"
    output_dir.mkdir(parents=True, exist_ok=True)

    df, class_names = build_metadata(
        args.data_dir, args.out_dir, args.limit_per_class
    )
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(
        f"[data] train={len(train_df)} validation={len(val_df)} "
        f"test={len(test_df)}; image size={IMG_SIZE}x{IMG_SIZE}"
    )

    train_ds = make_dataset(train_df, args.batch_size, shuffle=True)
    val_ds = make_dataset(val_df, args.batch_size)
    test_ds = make_dataset(test_df, args.batch_size)

    model = build_mlp()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=get_callbacks()
    )
    training_time_seconds = time.perf_counter() - start

    probabilities = model.predict(test_ds, verbose=0)
    y_true = test_df["label"].to_numpy()
    y_pred = probabilities.argmax(axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0
    )
    print(report)
    (output_dir / "classification_report.txt").write_text(report)

    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names)))
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("MLP - Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Approximate single-image inference latency.
    sample = tf.random.uniform([1, IMG_SIZE, IMG_SIZE, 1], 0, 255)
    for _ in range(3):
        model(sample, training=False)
    start = time.perf_counter()
    for _ in range(20):
        model(sample, training=False)
    latency_ms = (time.perf_counter() - start) / 20 * 1000

    metrics = {
        "model": "mlp",
        "test_accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "parameters": int(model.count_params()),
        "latency_ms_per_image": float(latency_ms),
        "training_time_seconds": float(training_time_seconds),
        "train_images": int(len(train_df)),
        "validation_images": int(len(val_df)),
        "test_images": int(len(test_df)),
        "input_size": IMG_SIZE,
        "batch_size": args.batch_size,
        "limit_per_subject_class": args.limit_per_class,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.history.get("loss", [])),
        "subjects_train": sorted(train_df["subject"].unique().tolist()),
        "subjects_validation": sorted(val_df["subject"].unique().tolist()),
        "subjects_test": sorted(test_df["subject"].unique().tolist()),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_history(history, output_dir / "training_curves.png")

    model.save(output_dir / "mlp.keras")
    (output_dir / "info.json").write_text(json.dumps({
        "name": "mlp",
        "display": "MLP Baseline (64x64, sampled dataset)",
        "model_file": "mlp.keras",
        "input_type": "image",
        "input_shape": [IMG_SIZE, IMG_SIZE, 1],
        "class_names": class_names
    }, indent=2))

    print("\n[metrics]")
    print(json.dumps(metrics, indent=2))
    print(f"\n[done] Model and results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
