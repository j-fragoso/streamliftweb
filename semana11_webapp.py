# semana11_webapp.py
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# ====== Configuración de la página ======
st.set_page_config(
    page_title="Asignación #11: Interacción con GPT-2",
    page_icon="🤖",
    layout="centered"
)

# ====== Estilo CSS ======
st.markdown("""
<style>
body {
    background-color: #f5f5f5;
    font-family: 'Arial', sans-serif;
}
h1 {
    color: #4B0082;
    text-align: center;
}
.stButton>button {
    background-color: #4B0082;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    padding: 8px 16px;
}
.stTextInput>div>input {
    border-radius: 8px;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)

# ====== Título ======
st.title("Interacción con GPT-2 📚")

# ====== Cargar modelo ======
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model

tokenizer, model = load_model()

# ====== Función para generar texto ======
def generar_texto(prompt, max_length=150):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_length=max_length, 
            do_sample=True, 
            top_k=50, 
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ====== Entrada de usuario ======
prompt_usuario = st.text_input("Escribe tu prompt aquí:")

# ====== Botón de envío ======
if st.button("Generar respuesta"):
    if prompt_usuario.strip() == "":
        st.warning("Por favor, escribe un prompt antes de enviar.")
    else:
        with st.spinner("Generando respuesta..."):
            resultado = generar_texto(prompt_usuario)
            st.success("Respuesta generada:")
            st.write(resultado)
