"""
MODEL 3 - MobileNetV2 TRANSFER LEARNING (lightweight version)
Static Hand Gesture Recognition - ICT-4442

Based on the original MobileNetV2 code:
Stage 1: freeze the MobileNetV2 backbone and train the classifier head.
Stage 2: fine-tune the top backbone layers with a very small learning rate.

Memory/time adjustments for an 8 GB RAM computer:
- 64x64 input images instead of 128x128.
- Default 20 images per subject per gesture class.
- Batch size 8.
- No tf.data cache() and limited prefetching.
- 5 epochs for stage 1 and 3 for stage 2 by default.
- Fine-tune only the top 10 backbone layers by default.

Run:
    python model3_mobilenetv2.py --data_dir ".\\leapGestRecog"

To train only the frozen-backbone stage:
    python model3_mobilenetv2.py --data_dir ".\\leapGestRecog" --no_finetune

To use the complete dataset (not recommended on a low-memory laptop):
    python model3_mobilenetv2.py --data_dir ".\\leapGestRecog" --limit_per_class 0
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
DEFAULT_LIMIT_PER_CLASS = 20

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ------------------------------ Data preparation -----------------------------
def build_metadata(data_dir, out_dir, limit_per_class=None):
    """Scan images, check readability, log duplicates and create subject split."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"No images found under {data_dir}")

    rows, removed, seen, duplicates = [], [], {}, []
    print(f"[data] Found {len(files)} image files. Checking them...")

    for i, path in enumerate(files, start=1):
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.load()
        except Exception as exc:
            removed.append({"path": str(path), "reason": f"unreadable: {exc}"})
            continue

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
        raise SystemExit("No readable images found.")

    pd.DataFrame(removed, columns=["path", "reason"]).to_csv(
        out_dir / "removed_files.csv", index=False
    )
    pd.DataFrame(duplicates, columns=["path", "duplicate_of"]).to_csv(
        out_dir / "duplicate_files.csv", index=False
    )

    print(
        f"[data] images={len(df)}  removed(corrupt)={len(removed)}  "
        f"duplicates logged={len(duplicates)}"
    )

    class_names = sorted(df["gesture"].unique())
    if len(class_names) != NUM_CLASSES:
        raise SystemExit(
            f"Expected {NUM_CLASSES} gesture classes, found {len(class_names)}: {class_names}"
        )

    class_to_index = {name: index for index, name in enumerate(class_names)}
    df["label"] = df["gesture"].map(class_to_index)

    # Sample independently for each subject and gesture class.
    if limit_per_class is not None:
        df = (
            df.groupby(["subject", "label"], group_keys=False)
              .apply(
                  lambda group: group.sample(
                      n=min(len(group), limit_per_class), random_state=SEED
                  )
              )
              .reset_index(drop=True)
        )

    # Keep the original subject-wise split: first 7 train, next 1 val, last 2 test.
    subjects = sorted(df["subject"].unique())
    if len(subjects) >= N_TRAIN_SUBJ + N_VAL_SUBJ + N_TEST_SUBJ:
        train_subjects = subjects[:7]
        val_subjects = subjects[7:8]
        test_subjects = subjects[8:10]
    else:
        n = len(subjects)
        n_test = max(1, round(n * 0.2))
        n_val = max(1, round(n * 0.1))
        train_subjects = subjects[:max(1, n - n_test - n_val)]
        val_subjects = subjects[len(train_subjects):len(train_subjects) + n_val]
        test_subjects = subjects[len(train_subjects) + n_val:]

    split_map = {
        **{subject: "train" for subject in train_subjects},
        **{subject: "val" for subject in val_subjects},
        **{subject: "test" for subject in test_subjects},
    }
    df["split"] = df["subject"].map(split_map)

    print(f"[data] train subjects={train_subjects}")
    print(f"[data] validation subjects={val_subjects}")
    print(f"[data] test subjects={test_subjects}")
    print("[data] class distribution:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))

    df.to_csv(out_dir / "metadata.csv", index=False)
    (out_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    return df, class_names


# ------------------------------ Image pipeline -------------------------------
def _load(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes, channels=1, expand_animations=False
    )
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return image, label


def make_ds(df, batch_size, shuffle=False, as_float=True):
    """Load images without caching them all in RAM."""
    dataset = tf.data.Dataset.from_tensor_slices(
        (df["path"].values, df["label"].values.astype("int32"))
    )
    dataset = dataset.map(_load, num_parallel_calls=1)

    if shuffle:
        dataset = dataset.shuffle(
            min(len(df), 1000), seed=SEED, reshuffle_each_iteration=True
        )

    dataset = dataset.batch(batch_size)

    if as_float:
        dataset = dataset.map(
            lambda x, y: (tf.cast(x, tf.float32), y),
            num_parallel_calls=1,
        )

    return dataset.prefetch(1)


# ------------------------------ Augmentation ---------------------------------
def augmentation_block():
    """Augmentation is active only during training; no horizontal flip."""
    return keras.Sequential([
        layers.RandomRotation(0.03),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.10, value_range=(0, 255)),
        layers.RandomContrast(0.10),
    ], name="augmentation")


# ------------------------------ Training helpers -----------------------------
def get_callbacks(patience=2):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-7,
        ),
    ]


def plot_history(histories, out_path, title):
    accuracy, val_accuracy, loss, val_loss = [], [], [], []
    for history in histories:
        accuracy += history.history.get("accuracy", [])
        val_accuracy += history.history.get("val_accuracy", [])
        loss += history.history.get("loss", [])
        val_loss += history.history.get("val_loss", [])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(accuracy, label="train")
    axes[0].plot(val_accuracy, label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------ Evaluation ------------------------------------
def get_flops(model, input_shape):
    try:
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph,
        )
        function = tf.function(lambda x: model(x, training=False))
        concrete = function.get_concrete_function(
            tf.TensorSpec([1, *input_shape], tf.float32)
        )
        _, graph_def = convert_variables_to_constants_v2_as_graph(concrete)

        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name="")
            options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            options["output"] = "none"
            info = tf.compat.v1.profiler.profile(graph=graph, options=options)
            return int(info.total_float_ops) if info else None
    except Exception as exc:
        print(f"[warn] FLOPs could not be computed: {exc}")
        return None


def measure_latency_ms(model, input_shape, runs=20):
    sample = tf.random.uniform([1, *input_shape], 0, 255)
    for _ in range(3):
        model(sample, training=False)

    start = time.perf_counter()
    for _ in range(runs):
        model(sample, training=False)
    return (time.perf_counter() - start) / runs * 1000


def evaluate_and_save(
    model, probabilities, y_true, class_names, out_dir, name, input_shape, extra=None
):
    out_dir = Path(out_dir)
    y_pred = probabilities.argmax(axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    print("\n[results] Classification report:")
    print(report)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names)))
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col, row, matrix[row, col],
                ha="center", va="center",
                color="white" if matrix[row, col] > matrix.max() / 2 else "black",
                fontsize=8,
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} - test confusion matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    metrics = {
        "model": name,
        "test_accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "parameters": int(model.count_params()),
        "flops": get_flops(model, input_shape),
        "latency_ms_per_image": measure_latency_ms(model, input_shape),
    }
    if extra:
        metrics.update(extra)

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("\n[results] Metrics:")
    print(json.dumps(metrics, indent=2))
    return metrics


# ------------------------------ MobileNetV2 model -----------------------------
def build_mobilenet(weights="imagenet"):
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 1))
    x = augmentation_block()(inputs)
    x = layers.Concatenate(name="grayscale_to_rgb")([x, x, x])
    # Equivalent to MobileNetV2 preprocess_input: convert 0..255 to -1..1.
    x = layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_preprocessing")(x)

    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=weights,
        alpha=0.35,
    )
    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="mobilenetv2_tl"), base


# -------------------------------- Main -----------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MobileNetV2 transfer learning")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--finetune_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--unfreeze_layers", type=int, default=10)
    parser.add_argument("--limit_per_class", type=int, default=DEFAULT_LIMIT_PER_CLASS,
                        help="Images per subject/class; use 0 for all images")
    parser.add_argument("--no_finetune", action="store_true")
    parser.add_argument("--weights", default="imagenet",
                        help="'imagenet' or 'none' for offline training from scratch")
    args = parser.parse_args()

    if args.epochs < 1 or args.finetune_epochs < 1 or args.batch_size < 1:
        raise SystemExit("Epoch counts and batch size must be positive.")
    if args.unfreeze_layers < 1:
        raise SystemExit("--unfreeze_layers must be at least 1.")

    limit = None if args.limit_per_class == 0 else args.limit_per_class
    weights = None if args.weights.lower() == "none" else args.weights

    output_root = Path(args.out_dir)
    output_dir = output_root / "mobilenetv2"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    df, class_names = build_metadata(args.data_dir, output_root, limit)

    train_df, val_df, test_df = (
        df[df["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    )
    if train_df.empty or val_df.empty or test_df.empty:
        raise SystemExit("Train, validation or test split is empty. Check dataset folders.")

    print(
        f"\n[data] train={len(train_df)}  validation={len(val_df)} "
        f"test={len(test_df)}"
    )
    train_ds = make_ds(train_df, args.batch_size, shuffle=True)
    val_ds = make_ds(val_df, args.batch_size)

    try:
        model, base = build_mobilenet(weights)
    except Exception as exc:
        raise SystemExit(
            f"Could not load MobileNetV2 weights ({args.weights}): {exc}\n"
            "For offline training, try adding --weights none."
        )

    histories = []
    loss = "sparse_categorical_crossentropy"

    # Stage 1: train classifier head with the backbone frozen.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=["accuracy"],
    )
    model.summary()
    print("\n[stage 1] Training classifier head; backbone is frozen.")
    histories.append(
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=get_callbacks(patience=2),
            verbose=1,
        )
    )

    # Stage 2: fine-tune only the top selected backbone layers.
    if not args.no_finetune:
        base.trainable = True
        for layer in base.layers[:-args.unfreeze_layers]:
            layer.trainable = False
        # Keep Batch Normalization layers frozen for stable small-batch training.
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss=loss,
            metrics=["accuracy"],
        )
        print(
            f"\n[stage 2] Fine-tuning top {args.unfreeze_layers} "
            "backbone layers with learning rate 1e-5."
        )
        histories.append(
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.finetune_epochs,
                callbacks=get_callbacks(patience=2),
                verbose=1,
            )
        )

    test_ds = make_ds(test_df, args.batch_size)
    probabilities = model.predict(test_ds, verbose=1)

    plot_history(
        histories,
        output_dir / "training_curves.png",
        "MobileNetV2 Transfer Learning",
    )

    metrics = evaluate_and_save(
        model,
        probabilities,
        test_df["label"].values,
        class_names,
        output_dir,
        "mobilenetv2",
        (IMG_SIZE, IMG_SIZE, 1),
        extra={
            "train_images": int(len(train_df)),
            "validation_images": int(len(val_df)),
            "test_images": int(len(test_df)),
            "epochs_stage1_completed": len(histories[0].history.get("loss", [])),
            "epochs_stage2_completed": (
                len(histories[1].history.get("loss", [])) if len(histories) > 1 else 0
            ),
            "input_size": IMG_SIZE,
            "batch_size": args.batch_size,
            "limit_per_subject_class": limit,
            "weights": args.weights,
            "fine_tuning": not args.no_finetune,
        },
    )

    model.save(output_dir / "mobilenetv2.keras")
    (output_dir / "info.json").write_text(
        json.dumps({
            "name": "mobilenetv2",
            "display": "MobileNetV2 Transfer Learning",
            "model_file": "mobilenetv2.keras",
            "input_type": "grayscale_image",
            "input_shape": [IMG_SIZE, IMG_SIZE, 1],
            "class_names_file": "../class_names.json",
        }, indent=2),
        encoding="utf-8",
    )

    elapsed = (time.perf_counter() - start_time) / 60
    print(f"\n[done] Model and results saved to: {output_dir}")
    print(f"[done] Total elapsed time: {elapsed:.1f} minutes")


if __name__ == "__main__":
    main()
