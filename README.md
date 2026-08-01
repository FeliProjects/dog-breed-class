# 🐕 Identificador de Razas de Perros

App web que clasifica razas de perros usando un modelo de deep learning basado en MobileNet.

## Uso
Sube una foto de un perro y obtén las 3 razas más probables con sus niveles de confianza.

## Modelo
- Arquitectura: Transfer Learning con MobileNetV2
- Entrada: 224x224 RGB
- Salida: Top-3 predicciones con probabilidades

## Ejecutar localmente
\```bash
pip install -r requirements.txt
streamlit run app.py
\```