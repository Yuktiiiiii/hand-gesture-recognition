import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
#  EDIT THESE THREE LINES, THEN JUST RUN:  python predict.py
# ============================================================
IMAGE_PATH = r"C:\Users\Prisha Chadha\Downloads\fistimage.jpg"
MODEL_NAME = "mobilenetv2"        # choose: "mlp", "cnn", or "mobilenetv2"
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
CLASSES_PATH = os.path.join(OUT_DIR, "class_names.json")

MODEL_FILES = {
    "mlp":         os.path.join(OUT_DIR, "mlp", "mlp.keras"),
    "cnn":         os.path.join(OUT_DIR, "cnn", "cnn.keras"),
    "mobilenetv2": os.path.join(OUT_DIR, "mobilenetv2", "mobilenetv2.keras"),
}

IMG_SIZE = 64


def load_image(image_path):
    """Load an image and prepare it as a (1, 64, 64, 1) grayscale array."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("L")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.asarray(image, dtype=np.float32)  # keep 0-255 (models rescale internally)
    return np.expand_dims(image_array, axis=(0, -1))


def main():
    if MODEL_NAME not in MODEL_FILES:
        raise ValueError(f"MODEL_NAME must be one of {list(MODEL_FILES.keys())}")

    model_path = MODEL_FILES[MODEL_NAME]

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            f"Train the '{MODEL_NAME}' model first."
        )
    if not os.path.isfile(CLASSES_PATH):
        raise FileNotFoundError(
            f"Class names file not found: {CLASSES_PATH}\n"
            "Check that outputs\\class_names.json exists."
        )

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = tf.keras.models.load_model(model_path)

    image_array = load_image(IMAGE_PATH)
    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    print(f"\n--- {MODEL_NAME.upper()} GESTURE PREDICTION ---")
    print(f"Image: {IMAGE_PATH}")
    print(f"Predicted Gesture: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    print("\nTop 3 predictions:")
    top_indices = np.argsort(predictions)[::-1][:3]
    for rank, index in enumerate(top_indices, start=1):
        print(f"{rank}. {class_names[int(index)]}: {predictions[index] * 100:.2f}%")


if __name__ == "__main__":
    main()
