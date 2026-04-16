import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
import pandas as pd
from datetime import datetime

# --- CONFIGURACIÓN DE CONEXIÓN ---
USER = "postgres.yuhxymowxtckzpqptmnz"
PASSWORD = "gDJ2zSM2mmnuXMRk"
HOST = "aws-1-us-west-2.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

st.set_page_config(page_title="Predictor de Iris", page_icon="🌸", layout="wide")

# --- FUNCIONES DE BASE DE DATOS ---

def get_connection():
    return psycopg2.connect(
        user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME
    )

def save_prediction(l_p, l_s, a_s, a_p, prediccion):
    """Guarda en ml.Irir_data usando comillas para respetar la mayúscula"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        # IMPORTANTE: "Irir_data" con I mayúscula y entre comillas
        query = """
            INSERT INTO "ml"."Irir_data" 
            (created_at, l_p, l_s, a_s, a_p, prediccion)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cur.execute(query, (datetime.now(), l_p, l_s, a_s, a_p, prediccion))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Error al guardar en DB: {e}")

def get_history_df():
    """Obtiene el historial de "ml"."Irir_data" y lo devuelve como DataFrame"""
    try:
        conn = get_connection()
        # También usamos comillas aquí para la tabla
        query = 'SELECT created_at, l_s, a_s, l_p, a_p, prediccion FROM "ml"."Irir_data" ORDER BY created_at DESC LIMIT 10'
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Renombrar para la vista del usuario
        df.columns = ["Fecha/Hora", "Sépalo Long.", "Sépalo Ancho", "Pétalo Long.", "Pétalo Ancho", "Predicción"]
        return df
    except Exception as e:
        st.error(f"Error al obtener historial: {e}")
        return pd.DataFrame()

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

# --- INTERFAZ ---

st.title("🌸 Predictor de Especies de Iris")
model, scaler, model_info = load_models()

if model is not None:
    st.header("Entrada de Datos:")
    
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Longitud Sépalo (l_s)", 0.0, 10.0, 5.0)
        sepal_width = st.number_input("Ancho Sépalo (a_s)", 0.0, 10.0, 3.0)
    with col2:
        petal_length = st.number_input("Longitud Pétalo (l_p)", 0.0, 10.0, 4.0)
        petal_width = st.number_input("Ancho Pétalo (a_p)", 0.0, 10.0, 1.0)
    
    if st.button("Predecir y Guardar"):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        target_names = model_info['target_names']
        predicted_species = target_names[prediction]
        
        st.success(f"Resultado: **{predicted_species}**")
        
        # Guardar (I mayúscula gestionada internamente por la función)
        save_prediction(petal_length, sepal_length, sepal_width, petal_width, predicted_species)
        st.toast("✅ Registro añadido a Irir_data")

    # --- SECCIÓN DE HISTORIAL ---
    st.markdown("---")
    st.subheader("📜 Cuadro de Historial (Últimas 10 ejecuciones)")
    
    df_history = get_history_df()
    
    if not df_history.empty:
        # Mostramos la tabla interactiva
        st.dataframe(
            df_history, 
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.info("La tabla ml.Irir_data está vacía o no se pudo leer.")
