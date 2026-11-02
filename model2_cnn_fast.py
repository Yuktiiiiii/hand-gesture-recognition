"""
MODEL 2 - LIGHTWEIGHT CUSTOM CNN
Static Hand Gesture Recognition - ICT-4442

Designed for a lower-memory laptop:
- 64x64 grayscale images
- 20 images per subject per gesture class
- 1,400 training images when using 7 training subjects and 10 classes
- Small two-block CNN
- Maximum 5 epochs, with early stopping

Run from your project folder:
    python model2_cnn_fast.py --data_dir ".\leapGestRecog"

The dataset sampling is subject-wise and reproducible. All 10 gesture classes
are retained. The train/validation/test subject split remains 7/1/2.
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
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------ Configuration ------------------------------
IMG_SIZE = 64
NUM_CLASSES = 10
SEED = 42
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
N_TRAIN_SUBJ, N_VAL_SUBJ, N_TEST_SUBJ = 7, 1, 2
DEFAULT_LIMIT_PER_SUBJECT_CLASS = 20

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ------------------------------ Data metadata -------------------------------
def build_metadata(data_dir, out_dir, limit_per_class=DEFAULT_LIMIT_PER_SUBJECT_CLASS):
    """
    Expected folder layout:
        data_dir/subject/gesture/image.png

    limit_per_class is applied separately to each subject and gesture.
    With 7 train subjects x 10 classes x 20 images = 1,400 train images.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"No images found under: {data_dir}")

    rows, removed, seen, duplicates = [], [], {}, []

    print(f"[data] Scanning {len(files)} image files and checking integrity...")
    for i, path in enumerate(files, start=1):
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.load()
        except Exception as exc:
            removed.append({"path": str(path), "reason": f"unreadable: {exc}"})
            continue

        # Hashing identifies duplicate files; duplicates are logged, not removed.
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen:
            duplicates.append({"path": str(path), "duplicate_of": seen[digest]})
        else:
            seen[digest] = str(path)

        rows.append({
            "path": str(path),
            "subject": path.parent.parent.name,
            "gesture": path.parent.name,
        })

        if i % 2000 == 0:
            print(f"[data] Checked {i}/{len(files)} files")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No readable images were found.")

    pd.DataFrame(removed, columns=["path", "reason"]).to_csv(
        out_dir / "removed_files.csv", index=False
    )
    pd.DataFrame(duplicates, columns=["path", "duplicate_of"]).to_csv(
        out_dir / "duplicate_files.csv", index=False
    )

    class_names = sorted(df["gesture"].unique())
    if len(class_names) != NUM_CLASSES:
        raise SystemExit(
            f"Expected {NUM_CLASSES} gesture classes, found {len(class_names)}: "
            f"{class_names}"
        )

    class_to_index = {name: index for index, name in enumerate(class_names)}
    df["label"] = df["gesture"].map(class_to_index)

    # Keep the original subject-wise split: first 7 train, next 1 validation,
    # next 2 test. A subject never appears in more than one split.
    subjects = sorted(df["subject"].unique())
    if len(subjects) < N_TRAIN_SUBJ + N_VAL_SUBJ + N_TEST_SUBJ:
        raise SystemExit(
            f"Expected at least 10 subjects for the 7/1/2 split; found {len(subjects)}."
        )

    train_subjects = subjects[:N_TRAIN_SUBJ]
    val_subjects = subjects[N_TRAIN_SUBJ:N_TRAIN_SUBJ + N_VAL_SUBJ]
    test_subjects = subjects[N_TRAIN_SUBJ + N_VAL_SUBJ:
                             N_TRAIN_SUBJ + N_VAL_SUBJ + N_TEST_SUBJ]

    split_map = {
        **{s: "train" for s in train_subjects},
        **{s: "val" for s in val_subjects},
        **{s: "test" for s in test_subjects},
    }
    df = df[df["subject"].isin(split_map)].copy()
    df["split"] = df["subject"].map(split_map)

    # Balanced, reproducible sample from EACH subject/class pair.
    if limit_per_class is not None and limit_per_class > 0:
        df = (
            df.groupby(["subject", "label"], group_keys=False)
              .apply(
                  lambda group: group.sample(
                      n=min(len(group), limit_per_class),
                      random_state=SEED
                  )
              )
              .reset_index(drop=True)
        )

    print(f"[data] Readable images: {len(rows)}")
    print(f"[data] Corrupt files removed: {len(removed)}")
    print(f"[data] Duplicate files logged: {len(duplicates)}")
    print(f"[data] Train subjects: {train_subjects}")
    print(f"[data] Validation subjects: {val_subjects}")
    print(f"[data] Test subjects: {test_subjects}")
    print(f"[data] Sampled images per split:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))

    df.to_csv(out_dir / "metadata.csv", index=False)
    with open(out_dir / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    return df, class_names


# --------------------------- Image loading pipeline -------------------------
def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes, channels=1, expand_animations=False
    )
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(tf.clip_by_value(image, 0, 255), tf.float32)
    return image, label


def make_dataset(df, batch_size, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((
        df["path"].values,
        df["label"].values.astype("int32"),
    ))
    dataset = dataset.map(load_image, num_parallel_calls=1)

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(df), 1000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)


# -------------------------------- Augmentation ------------------------------
def augmentation_block():
    return keras.Sequential([
        layers.RandomRotation(0.025),
        layers.RandomTranslation(0.04, 0.04),
        layers.RandomZoom(0.08),
    ], name="augmentation")


# --------------------------------- CNN model --------------------------------
def build_cnn():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")
    x = augmentation_block()(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)

    # Small CNN: fewer filters and only two convolutional blocks.
    x = layers.Conv2D(16, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.20)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="lightweight_custom_cnn")


# -------------------------------- Callbacks ---------------------------------
def get_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# ------------------------------- Save results -------------------------------
def plot_history(history, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history.get("accuracy", []), label="Train")
    axes[0].plot(history.history.get("val_accuracy", []), label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history.get("loss", []), label="Train")
    axes[1].plot(history.history.get("val_loss", []), label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Lightweight custom CNN")
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to leapGestRecog folder (subject/gesture/images)"
    )
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--limit_per_class", type=int,
        default=DEFAULT_LIMIT_PER_SUBJECT_CLASS,
        help="Maximum images sampled per subject per gesture class; default 20"
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.limit_per_class < 1:
        raise SystemExit("--epochs, --batch_size and --limit_per_class must be positive.")

    output_root = Path(args.out_dir)
    model_dir = output_root / "cnn"
    model_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    df, class_names = build_metadata(
        args.data_dir, output_root, args.limit_per_class
    )

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(
        f"[data] train={len(train_df)}  validation={len(val_df)} "
        f"test={len(test_df)}"
    )
    if train_df.empty or val_df.empty or test_df.empty:
        raise SystemExit("Train, validation or test split is empty. Check dataset folders.")

    model = build_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        make_dataset(train_df, args.batch_size, shuffle=True),
        validation_data=make_dataset(val_df, args.batch_size),
        epochs=args.epochs,
        callbacks=get_callbacks(),
        verbose=1,
    )

    # Evaluate only on the held-out test subjects.
    probabilities = model.predict(
        make_dataset(test_df, args.batch_size), verbose=1
    )
    predictions = probabilities.argmax(axis=1)
    y_true = test_df["label"].to_numpy()

    accuracy = accuracy_score(y_true, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="macro", zero_division=0
    )
    report = classification_report(
        y_true,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    print("\nClassification report:\n")
    print(report)
    (model_dir / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    matrix = confusion_matrix(
        y_true, predictions, labels=list(range(len(class_names)))
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, matrix[row, col], ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Lightweight CNN - Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(model_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    plot_history(history, model_dir / "training_curves.png")

    model_path = model_dir / "cnn.keras"
    model.save(model_path)

    elapsed_seconds = time.perf_counter() - start_time
    metrics = {
        "model": "lightweight_cnn",
        "test_accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "parameters": int(model.count_params()),
        "image_size": IMG_SIZE,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.history.get("loss", [])),
        "limit_per_subject_per_class": args.limit_per_class,
        "train_images": int(len(train_df)),
        "validation_images": int(len(val_df)),
        "test_images": int(len(test_df)),
        "training_and_evaluation_seconds": float(elapsed_seconds),
    }
    (model_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (model_dir / "info.json").write_text(
        json.dumps({
            "name": "cnn",
            "display": "Lightweight Custom CNN",
            "model_file": "cnn.keras",
            "input_type": "grayscale_image",
            "image_size": IMG_SIZE,
            "class_names_file": "../class_names.json",
        }, indent=2),
        encoding="utf-8",
    )

    print("\n[metrics]")
    print(json.dumps(metrics, indent=2))
    print(f"\n[done] Model saved to: {model_path.resolve()}")
    print(f"[done] Total elapsed time: {elapsed_seconds / 60:.1f} minutes")


if __name__ == "__main__":
    main()
