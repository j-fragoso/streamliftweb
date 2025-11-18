# semana11_webapp.py
import streamlit as st
import pandas as pd
import numpy as np

# ====== Estilos CSS personalizados ======
st.markdown(
    """
    <style>
    /* Fondo y color de texto */
    .stApp {
        background-color: #2f4f4f;  /* Verde oscuro desaturado */
        color: white;
        font-family: 'Verdana', sans-serif;
        text-align: center;
    }

    /* Centrar títulos y subtítulos */
    .css-1v3fvcr h1, .css-1v3fvcr h2, .css-1v3fvcr h3 {
        text-align: center;
        color: white;
    }

    /* Botones */
    .stButton>button {
        background-color: #4CAF50; /* Verde más claro para botones */
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 10em;
        font-size: 16px;
        font-weight: bold;
        margin: auto;
        display: block;
    }

    /* Inputs de texto y selectbox */
    .stTextInput>div>div>input, .stSelectbox>div>div>div>select, .stSlider>div>div>div>div>div {
        background-color: #3e5f5f;
        color: white;
        border-radius: 5px;
    }

    /* Dataframe */
    .stDataFrame, .stTable {
        color: black;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======= Título y secciones =======
st.title("WebApp Interactiva - Semana 11")
st.header("Control del Modelo")
st.subheader("Configuración de parámetros")

st.write("Ajusta los parámetros del modelo y visualiza los datos generados automáticamente.")

# ======= Inputs de usuario =======
name = st.text_input("Nombre del modelo", "MiModelo")
options = st.selectbox("Tipo de modelo", ["CNN", "RNN", "YOLO", "Diffusion"])
threshold = st.slider("Umbral de confianza", 0.0, 1.0, 0.5)
run_button = st.button("Ejecutar inferencia")

# ======= Datos de ejemplo =======
df = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

# ======= Visualización =======
st.subheader("Tabla de datos")
st.dataframe(df)

st.subheader("Gráfica de datos")
st.line_chart(df)

# ======= Mensaje de ejecución =======
if run_button:
    st.success(f"Inferencia ejecutada para el modelo {name} ({options}) con umbral {threshold}")
