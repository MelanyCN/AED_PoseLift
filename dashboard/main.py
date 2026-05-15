"""
Dashboard Principal - PoseLift Data Exploration
Interfaz principal para visualizar datos del dataset PoseLift.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from config import PAGE_CONFIG, PREPROCESSING_OUTPUTS

# Configurar página
st.set_page_config(
    page_title=PAGE_CONFIG['page_title'],
    page_icon=PAGE_CONFIG['page_icon'],
    layout=PAGE_CONFIG['layout'],
    initial_sidebar_state=PAGE_CONFIG['initial_sidebar_state']
)

# Estilos
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2C3E50;
        padding: 20px;
    }
    .metric-card {
        background-color: #F0F2F6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #E8F4F8;
        padding: 15px;
        border-left: 4px solid #3498DB;
        border-radius: 4px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Encabezado principal
st.markdown("""
<div class="main-header" style="color: white;">
    <h1 style="color: white;">Dashboard PoseLift</h1>
    <h3 style="color: white;">Análisis Exploratorio de Datos - Detección de Anomalías en Vigilancia</h3>
</div>
""", unsafe_allow_html=True)

# Descripción del proyecto
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Acerca del Proyecto
        
    - **Visualizar keypoints 2D** del dataset PoseLift en tiempo real
    - **Análisis estadístico descriptivo** de movimientos corporales
    - **Explorar patrones** de comportamiento normal vs anomalías
    - **Comparar personas y videos** para entender variabilidad
    - **Detectar anomalías** usando análisis multivariado
    - **Validar hipótesis** sobre patrones de comportamiento
    """)

with col2:
    st.markdown("""
    ### Dataset PoseLift
    
    **Estructura de datos:**
    - Videos de vigilancia (train + test splits)
    - Keypoints COCO17 detectados por AlphaPose
    - Unidad de análisis: (video_id, frame_id, person_id)
    - Etiqueta: Normal vs Shoplifting (a nivel de frame)
    
    **Archivos disponibles:**
    - `Json_files/` → Keypoints JSON AlphaPose
    - `pre_procesamiento/outputs/` → Datos tabulares (CSV)
    - `hipotesis_1-3/outputs/` → Análisis completados
    """)

# Estadísticas rápidas
st.divider()
st.markdown("### Estadísticas Rápidas del Dataset")

try:
    # Cargar datos test
    test_csv = PREPROCESSING_OUTPUTS / "test_tabular.csv"
    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Registros (Test)", f"{len(df_test):,}")
        with col2:
            st.metric("Columnas (Test)", len(df_test.columns))
        
        if 'label' in df_test.columns:
            normal_count = (df_test['label'] == 0).sum()
            shoplifting_count = (df_test['label'] == 1).sum()
            
            with col3:
                st.metric("Normal", f"{normal_count:,}")
            with col4:
                st.metric("Shoplifting", f"{shoplifting_count:,}")
    else:
        st.warning(f"No se encontró: {test_csv}")
        
except Exception as e:
    st.warning(f"No se pudieron cargar estadísticas: {str(e)}")

st.markdown("---")

# Información clave sobre el AED
st.markdown("### Conceptos Clave")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Unidad de Análisis**
    
    Cada registro representa:
    - Una **persona detectada**
    - En un **frame específico**
    - De un **video dado**
    - Con su etiqueta a nivel de frame
    """)

with col2:
    st.markdown("""
    **Tipo de Datos**
    
    - **Serie temporal multivariada**
    - 17 keypoints COCO (articulations)
    - 3 coordenadas por keypoint (x, y, confidence)
    - ~59 columnas en dataset tabular
    """)

with col3:
    st.markdown("""
    **Etiquetado**
    
    - Label = 0: Normal
    - Label = 1: Shoplifting
    - Etiqueta a **nivel de frame** (no persona)
    """)

st.divider()

# Instrucciones de navegación
st.markdown("""
### Metodología del AED

**Por qué Mann-Whitney U en lugar de t-test?**
- 100% de variables NO son normales (Shapiro-Wilk p ≤ 0.05)
- Mann-Whitney U es test NO paramétrico basado en rangos
- Más robusto con distribuciones asimétricas

**Por qué Spearman en lugar de Pearson?**
- Spearman funciona con datos no normales
- Basado en correlación de rangos
- Mejor para relaciones monótonas

**Análisis Multivariado**
- El comportamiento de shoplifting NO se explica por una sola coordenada
- Requiere patrones multivariados: postura, velocidad, relación de articulaciones
- Temporal: dependencia entre frames

""")

st.divider()

# Footer
st.markdown("""
<div style='text-align: center; color: #7F8C8D; font-size: 12px;'>
    <p>Dashboard PoseLift | Análisis Exploratorio de Datos</p>
    <p>Detección de Anomalías en Vigilancia | COCO17 Keypoints</p>
</div>
""", unsafe_allow_html=True)

