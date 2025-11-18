# semana11_webapp.py
import streamlit as st
import pandas as pd
import numpy as np

# ====== Título y secciones ======
st.title("Semana 11 - Interfaz de Modelos")
st.header("Dashboard Interactivo")
st.subheader("Explora tus modelos y datos")

st.write("Aquí puedes probar distintos modelos, configurar parámetros y visualizar resultados.")

# ====== Inputs de usuario ======
# Nombre del modelo
model_name = st.text_input("Nombre del modelo", "MiModelo")

# Selección de modelo
model_type = st.selectbox("Selecciona un modelo", ["CNN", "RNN", "YOLO", "Diffusion"])

# Slider de threshold con valores correctos
threshold = st.slider("Umbral de confianza", 0.0, 1.0, 0.5)

# Botón para ejecutar la inferencia
run_button = st.button("Ejecutar inferencia")

# ====== Mostrar resultados aleatorios (ejemplo) ======
# Generamos un DataFrame de ejemplo
df = pd.DataFrame(np.random.randn(20, 3), columns=["A", "B", "C"])

st.write("## Datos de ejemplo")
st.dataframe(df)
st.line_chart(df)

# ====== Mensaje personalizado ======
user_name = st.text_input("Escribe tu nombre", "Jordi")
user_age = st.slider("Tu edad", 0, 100, 32)
st.write(f"Hola {user_name}, tienes {user_age} años y estás usando el modelo {model_name} ({model_type}) con un umbral de {threshold}.")

# ====== Acción del botón ======
if run_button:
    st.success(f"Inferencia ejecutada para el modelo {model_name} ({model_type}) con threshold {threshold}.")
    # Aquí puedes agregar la llamada a tu modelo si tienes uno


