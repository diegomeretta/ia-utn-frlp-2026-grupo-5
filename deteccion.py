import sys
import os
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


MODEL_PATH = Path('modelo_sign_language.keras')
DEFAULT_IMAGE = Path('letra_a.png')


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {path.resolve()}")
    # Carga el modelo (formato Keras .keras o carpeta SavedModel)
    model = tf.keras.models.load_model(str(path))
    return model


def preprocess_image(image_path: Path):
    if not image_path.exists():
        raise FileNotFoundError(f"Imagen no encontrada en {image_path.resolve()}")
    # Abrir con PIL y convertir a escala de grises
    img = Image.open(str(image_path)).convert('L')  # 'L' -> 8-bit pixels, black and white
    # Redimensionar la imagen a 28x28
    resample = Image.Resampling.LANCZOS
    img = img.resize((28, 28), resample)
    arr = np.array(img, dtype=np.float32)
    # Asegurarnos de que los valores estén en 0-255
    if arr.max() <= 1.0:
        # imagen ya normalizada en 0-1, convertir a 0-255
        arr = arr * 255.0
    # Normalizar al rango [0,1] para el modelo (mismo preprocesamiento que en main.py)
    arr = arr / 255.0
    # Añadir canales y batch dimension: (28,28) -> (1,28,28,1)
    arr = arr.reshape(1, 28, 28, 1)
    return arr


def predict_image(model, processed_image: np.ndarray):
    preds = model.predict(processed_image)
    class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return class_idx, confidence


def main(image_path: str | None = None):
    img_path = Path(image_path) if image_path else DEFAULT_IMAGE

    print(f"Cargando modelo desde: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    print(f"Preprocesando imagen: {img_path}")
    img = preprocess_image(img_path)

    print("Realizando predicción...")
    idx, conf = predict_image(model, img)
    # Mapeo por defecto: 0 -> 'A', 1 -> 'B', ..., 24 -> 'Y'
    default_labels = {i: chr(ord('A') + i) for i in range(25)}
    mapped = default_labels.get(idx, None)

    print(f"Predicción: clase={idx}, confianza={conf*100:.2f}%")
    if mapped:
        print(f"Letra estimada: {mapped}")
    else:
        print("No hay mapeo para esta clase. Proporciona un diccionario índice->letra si quieres otro orden.")


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        main(arg)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
