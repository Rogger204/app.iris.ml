import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
from datetime import datetime

# Configuración de la base de datos
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

def init_db():
    """Crea la tabla si no existe"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS historial_predicciones (
            id SERIAL PRIMARY KEY,
            fecha TIMESTAMP,
            sepal_length FLOAT,
            sepal_width FLOAT,
            petal_length FLOAT,
            petal_width FLOAT,
            especie_predicha TEXT,
            confianza FLOAT
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def save_prediction(sl, sw, pl, pw, species, confidence):
    """Guarda el registro en la DB"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO historial_predicciones 
        (fecha, sepal_length, sepal_width, petal_length, petal_width, especie_predicha, confianza)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (datetime.now(), sl, sw, pl, pw, species, confidence))
    conn.commit()
    cur.close()
    conn.close()

def get_history():
    """Obtiene los registros ordenados por fecha descendente"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT fecha, especie_predicha, confianza FROM historial_predicciones ORDER BY fecha DESC LIMIT 10")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

# --- LÓGICA DE LA APP ---

# Inicializar tabla al arrancar
init_db()

@st.cache_resource
def load_models():
    try:
        model = joblib.load('componets/iris_model.pkl')
        scaler = joblib.load('componets/iris_scaler.pkl')
        with open('componets/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except Exception:
        st.error("Error al cargar archivos del modelo.")
        return None, None, None

st.title("🌸 Predictor de Especies de Iris")
model, scaler, model_info = load_models()

if model is not None:
    st.header("Ingresa las características de la flor:")
    
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Longitud Sépalo (cm)", 0.0, 10.0, 5.0)
        sepal_width = st.number_input("Ancho Sépalo (cm)", 0.0, 10.0, 3.0)
    with col2:
        petal_length = st.number_input("Longitud Pétalo (cm)", 0.0, 10.0, 4.0)
        petal_width = st.number_input("Ancho Pétalo (cm)", 0.0, 10.0, 1.0)
    
    if st.button("Predecir Especie"):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        target_names = model_info['target_names']
        predicted_species = target_names[prediction]
        confidence = float(max(probabilities))
        
        # 1. Mostrar resultado actual
        st.success(f"Especie predicha: **{predicted_species}** (Confianza: {confidence:.1%})")
        
        # 2. GUARDAR EN BASE DE DATOS
        save_prediction(sepal_length, sepal_width, petal_length, petal_width, predicted_species, confidence)

    # --- SECCIÓN DE HISTORIAL ---
    st.markdown("---")
    st.subheader("📜 Historial de Ejecuciones (Últimas 10)")
    
    history = get_history()
    if history:
        # Formatear datos para mostrar en una tabla
        for res in history:
            # res[0] es la fecha, res[1] especie, res[2] confianza
            st.write(f"⏱ **{res[0].strftime('%Y-%m-%d %H:%M:%S')}** | 🏷 {res[1]} | ✅ {res[2]:.1%}")
    else:
        st.info("Aún no hay registros en el historial.")
