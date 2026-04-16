import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
from datetime import datetime

# --- CONFIGURACIÓN DE CONEXIÓN ---
USER = "postgres.yuhxymowxtckzpqptmnz"
PASSWORD = "gDJ2zSM2mmnuXMRk"
HOST = "aws-1-us-west-2.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

# --- FUNCIONES DE BASE DE DATOS ---

def get_connection():
    return psycopg2.connect(
        user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME
    )

def save_prediction(l_p, l_s, a_s, a_p, prediccion):
    """Guarda el registro en la tabla EXACTA de tu imagen"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        # IMPORTANTE: Usamos "ml"."irir_data" para que coincida con tu imagen
        query = """
            INSERT INTO "ml"."Irir_data" 
            (created_at, l_p, l_s, a_s, a_p, prediccion)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cur.execute(query, (datetime.now(), l_p, l_s, a_s, a_p, prediccion))
        conn.commit() # Esto guarda los cambios físicamente
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Error al guardar en DB: {e}")

def get_history():
    """Consulta la tabla ml.irir_data en orden descendente"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        # Traemos los datos de la tabla correcta
        cur.execute('SELECT created_at, prediccion FROM "ml"."irir_data" ORDER BY created_at DESC LIMIT 10')
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except:
        return []

# --- CARGA DE MODELOS ---

@st.cache_resource
def load_models():
    try:
        model = joblib.load('componets/iris_model.pkl')
        scaler = joblib.load('componets/iris_scaler.pkl')
        with open('componets/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except Exception:
        return None, None, None

# --- LÓGICA DE LA APP ---

st.title("🌸 Predictor de Especies de Iris")
model, scaler, model_info = load_models()

if model is not None:
    st.header("Ingresa las características de la flor:")
    
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Longitud Sépalo (l_s)", 0.0, 10.0, 5.0)
        sepal_width = st.number_input("Ancho Sépalo (a_s)", 0.0, 10.0, 3.0)
    with col2:
        petal_length = st.number_input("Longitud Pétalo (l_p)", 0.0, 10.0, 4.0)
        petal_width = st.number_input("Ancho Pétalo (a_p)", 0.0, 10.0, 1.0)
    
    if st.button("Predecir Especie"):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        target_names = model_info['target_names']
        predicted_species = target_names[prediction]
        
        st.success(f"Especie predicha: **{predicted_species}**")
        
        # --- EL PASO CLAVE: Guardar usando los nombres de tu tabla ---
        # l_p = petal_length, l_s = sepal_length, a_s = sepal_width, a_p = petal_width
        save_prediction(petal_length, sepal_length, sepal_width, petal_width, predicted_species)
        st.info("¡Datos enviados a la tabla ml.irir_data!")

    # --- SECCIÓN DE HISTORIAL ---
    st.markdown("---")
    st.subheader("📜 Historial (Últimos 10)")
    
    history = get_history()
    if history:
        for res in history:
            st.write(f"⏱ `{res[0].strftime('%H:%M:%S')}` | 🌸 Especie: **{res[1]}**")
    else:
        st.info("No hay datos en ml.Irir_data aún.")
