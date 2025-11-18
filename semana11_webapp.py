# semana11_webapp.py
import streamlit as st
import pandas as pd
import numpy as np

# ======= Título y secciones =======
st.title("WebApp Interactiva - Semana 11")
st.header("Sección de Control del Modelo")
st.subheader("Configuración del modelo")

st.write("Aquí puedes configurar tu modelo y parámetros para la inferencia.")

# ======= Inputs de usuario =======
name = st.text_input("Nombre del modelo", "MiModelo")
options = st.selectbox("Elige un tipo de modelo", ["CNN", "RNN", "YOLO", "Diffusion"])
threshold = st.slider("Umbral de confianza", 0.0, 1.0, 0.5)
run_button = st.button("Ejecutar inferencia")

# ======= Generación de datos de ejemplo =======
df = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

st.subheader("Datos y visualización")
st.data
