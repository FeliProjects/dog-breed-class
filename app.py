# -*- coding: utf-8 -*-
import os
import json
import logging
import numpy as np
import streamlit as st
from tensorflow import keras
from PIL import Image
import io
import time

# ------------------------------------------------------------------ Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ Config
MODEL_PATH = os.getenv("KERAS_MODEL_PATH", "mi_modelo_perros.keras")
CLASSES_PATH = os.getenv("DOG_CLASSES_PATH", "class_names.json")
IMAGE_SIZE = (224, 224)
BASE_MODEL = os.getenv("BASE_MODEL", "mobilenet")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Preprocesadores
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as _mobilenet
from tensorflow.keras.applications.resnet50 import preprocess_input as _resnet
from tensorflow.keras.applications.efficientnet import preprocess_input as _efficientnet
from tensorflow.keras.applications.vgg16 import preprocess_input as _vgg16

PREPROCESSORS = {
    "mobilenet": _mobilenet,
    "resnet": _resnet,
    "efficientnet": _efficientnet,
    "vgg16": _vgg16,
}


# ------------------------------------------------------------------ Funciones
def clean_breed_name(breed_string):
    """Extrae el nombre legible de 'codigo-nombre'."""
    if isinstance(breed_string, str) and "-" in breed_string:
        return breed_string.split("-")[-1].replace("_", " ").title()
    return breed_string


@st.cache_resource
def load_model_and_classes():
    """Carga el modelo y las clases una sola vez (cacheado por Streamlit)."""
    logger.info("Cargando modelo Keras desde %s ...", MODEL_PATH)
    model = keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH, "r") as f:
        class_names = json.load(f)

    logger.info("[OK] Modelo cargado | Base: %s | Clases: %d", BASE_MODEL, len(class_names))
    return model, class_names


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocesa la imagen según el modelo base."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image, dtype="float32")

    preprocess_fn = PREPROCESSORS.get(BASE_MODEL, lambda x: x / 255.0)
    img_array = preprocess_fn(img_array)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(image_array, model, class_names):
    """Devuelve las top-3 predicciones con sus probabilidades."""
    preds = model.predict(image_array, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:3]

    results = []
    for idx in top_indices:
        breed_raw = class_names[idx] if isinstance(class_names, list) else str(idx)
        breed = clean_breed_name(breed_raw)
        results.append({"breed": breed, "probability": float(preds[idx])})

    return results


# ------------------------------------------------------------------ CSS personalizado
CUSTOM_CSS = """
<style>
/* Fondo con gradiente */
.stApp {
    background: linear-gradient(135deg, #667eea, #764ba2);
}

/* Tarjeta principal */
.main .block-container {
    background: #ffffff;
    border-radius: 16px;
    padding: 2.5rem;
    box-shadow: 0 20px 60px rgba(0,0,0,.3);
    max-width: 650px;
}

/* Título */
h1 {
    text-align: center;
    color: #333 !important;
    font-size: 1.8rem !important;
}

/* Items de predicción */
.prediction-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: .75rem;
    border-left: 5px solid transparent;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.prediction-item:hover {
    transform: translateX(5px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

/* Medallas */
.rank {
    font-size: 1.5rem;
    min-width: 40px;
    text-align: center;
}

/* Info de raza */
.breed-info {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}
.breed-name {
    font-weight: 600;
    color: #333;
    font-size: 1.05rem;
    text-transform: capitalize;
}
.confidence-badge {
    font-size: 0.8rem;
    color: #667eea;
    font-weight: 500;
}

/* Barra de progreso */
.probability-bar {
    flex: 1;
    height: 12px;
    background: #e9ecef;
    border-radius: 6px;
    overflow: hidden;
    min-width: 120px;
}
.probability-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.probability-text {
    min-width: 60px;
    text-align: right;
    font-weight: 700;
    color: #555;
    font-size: 1rem;
}

/* Primer lugar destacado */
.prediction-gold {
    background: linear-gradient(135deg, #fffde7, #ffffff) !important;
    border-left: 4px solid #ffc107 !important;
    box-shadow: 0 4px 16px rgba(255, 193, 7, 0.15);
}
.prediction-silver {
    border-left: 4px solid #c0c0c0 !important;
}
.prediction-bronze {
    border-left: 4px solid #cd7f32 !important;
}
</style>
"""


# ------------------------------------------------------------------ Render
def render_prediction(prediction, rank_index):
    """Genera el HTML para un item de predicción."""
    pct = (prediction["probability"] * 100)
    rank_emoji = ["🥇", "🥈", "🥉"][rank_index]
    css_class = ["prediction-gold", "prediction-silver", "prediction-bronze"][rank_index]

    confidence_badge = ""
    if rank_index == 0 and prediction["probability"] > 0.7:
        confidence_badge = '<span class="confidence-badge">✅ Alta confianza</span>'

    html = f"""
    <div class="prediction-item {css_class}">
        <span class="rank">{rank_emoji}</span>
        <div class="breed-info">
            <span class="breed-name">{prediction['breed']}</span>
            {confidence_badge}
        </div>
        <div class="probability-bar">
            <div class="probability-fill" style="width:{pct:.1f}%"></div>
        </div>
        <span class="probability-text">{pct:.1f}%</span>
    </div>
    """
    return html


# ------------------------------------------------------------------ App
def main():
    st.set_page_config(
        page_title="Identificador de Razas",
        page_icon="🐕",
        layout="centered",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown("# 🐕 Identificador de Razas")
    st.markdown('<p style="text-align:center; color:#777; margin-top:-0.5rem;">Sube una foto de tu perro y descubre su raza</p>', unsafe_allow_html=True)

    st.divider()

    # --- Cargar modelo ---
    try:
        model, class_names = load_model_and_classes()
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {e}")
        st.info("Verifica que los archivos `mi_modelo_perros.keras` y `class_names.json` estén en el mismo directorio que `app.py`.")
        return

    # --- Upload ---
    uploaded_file = st.file_uploader(
        "Arrastra una imagen o haz clic aquí",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Validar tamaño
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"El archivo excede el tamaño máximo de 10 MB.")
            return

        # Mostrar imagen
        image = Image.open(uploaded_file)
        st.image(image, caption="Vista previa", use_container_width=True)

        # Botón analizar
        if st.button("🔍 Analizar", use_container_width=True, type="primary"):
            with st.spinner("Analizando imagen..."):
                # Pequeño delay para que el spinner renderice
                time.sleep(0.3)

                try:
                    # Preprocesar y predecir
                    processed = preprocess_image(image)
                    results = predict(processed, model, class_names)

                    # Mostrar resultados
                    st.markdown("## Resultados de Análisis")
                    st.markdown("---")

                    html_block = ""
                    for i, pred in enumerate(results):
                        html_block += render_prediction(pred, i)

                    st.markdown(html_block, unsafe_allow_html=True)

                    # Botón nueva clasificación
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🔄 Nueva clasificación", use_container_width=True):
                        st.rerun()

                except Exception as e:
                    st.error(f"Error al procesar la imagen: {e}")


if __name__ == "__main__":
    main()