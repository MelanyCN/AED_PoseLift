"""
Página 3: Análisis Hipótesis 1
Visualización dinámica del análisis espacial por keypoints y grupos corporales.
Genera gráficos interactivos desde los CSVs de resultados.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from config import H1_DIR
from utils.display_utils import show_csv


st.set_page_config(page_title="Análisis Hipótesis 1", layout="wide")


# ============================================================
# Utilidades
# ============================================================

def load_csv_if_exists(path):
    if path.exists():
        return pd.read_csv(path)
    return None


def find_column(df, candidates):
    """
    Busca una columna posible dentro de un dataframe.
    Sirve porque los nombres pueden variar entre scripts.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def plot_ranking_keypoints(df_keypoints, metric_col, top_n):
    df_plot = df_keypoints.copy()

    keypoint_col = find_column(
        df_plot,
        ["keypoint", "variable", "nombre_keypoint", "punto_corporal"]
    )

    if keypoint_col is None or metric_col is None:
        return None

    df_plot = df_plot.sort_values(metric_col, ascending=False).head(top_n)
    df_plot = df_plot.sort_values(metric_col, ascending=True)

    fig = px.bar(
        df_plot,
        x=metric_col,
        y=keypoint_col,
        orientation="h",
        title=f"Top {top_n}: keypoints por {metric_col}",
        labels={
            metric_col: "Valor de la métrica",
            keypoint_col: "Keypoint",
        },
        hover_data=df_plot.columns,
    )

    fig.update_layout(height=max(500, 35 * len(df_plot)))
    return fig


def plot_ranking_grupos(df_grupos, metric_col="effect_total_mediana_grupo"):
    df_plot = df_grupos.copy()

    group_col = find_column(
        df_plot,
        ["grupo", "grupo_corporal", "body_group", "region"]
    )

    if metric_col not in df_plot.columns:
        st.warning(
            f"No se encontró la métrica esperada `{metric_col}`. "
            f"Columnas disponibles: {df_plot.columns.tolist()}"
        )
        return None

    if group_col is None:
        st.warning("No se encontró columna de grupo corporal.")
        return None

    df_plot = df_plot.sort_values(metric_col, ascending=True)

    fig = px.bar(
        df_plot,
        x=metric_col,
        y=group_col,
        orientation="h",
        title="Hipótesis 1: grupos corporales más distintivos entre normal y shoplifting",
        labels={
            metric_col: "Tamaño de efecto mediano del grupo",
            group_col: "Grupo corporal",
        },
        hover_data=df_plot.columns,
    )

    fig.update_layout(
        height=550,
        xaxis_title="Tamaño de efecto mediano del grupo",
        yaxis_title="Grupo corporal",
    )

    return fig

def plot_heatmap_metricas_grupo(df_grupos, selected_metrics):
    group_col = find_column(
        df_grupos,
        ["grupo", "grupo_corporal", "body_group", "region"]
    )

    if group_col is None or not selected_metrics:
        return None

    heat_df = df_grupos[[group_col] + selected_metrics].copy()
    heat_df = heat_df.set_index(group_col)

    # Normalización por columna para comparar métricas de distinta escala
    heat_norm = heat_df.copy()
    for col in heat_norm.columns:
        col_min = heat_norm[col].min()
        col_max = heat_norm[col].max()
        denom = col_max - col_min

        if denom == 0:
            heat_norm[col] = 0
        else:
            heat_norm[col] = (heat_norm[col] - col_min) / denom

    fig = px.imshow(
        heat_norm,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Heatmap normalizado de métricas por grupo corporal",
        labels={
            "x": "Métrica",
            "y": "Grupo corporal",
            "color": "Valor normalizado",
        },
    )

    fig.update_layout(height=600)
    return fig


def plot_comparacion_metricas_keypoints(df_keypoints, x_metric, y_metric, size_metric=None):
    keypoint_col = find_column(
        df_keypoints,
        ["keypoint", "variable", "nombre_keypoint", "punto_corporal"]
    )

    group_col = find_column(
        df_keypoints,
        ["grupo", "grupo_corporal", "body_group", "region"]
    )

    if keypoint_col is None or x_metric is None or y_metric is None:
        return None

    fig = px.scatter(
        df_keypoints,
        x=x_metric,
        y=y_metric,
        size=size_metric if size_metric else None,
        color=group_col if group_col else None,
        hover_name=keypoint_col,
        hover_data=df_keypoints.columns,
        title=f"Relación entre {x_metric} y {y_metric} por keypoint",
        labels={
            x_metric: x_metric,
            y_metric: y_metric,
        },
    )

    fig.update_layout(height=650)
    return fig


def get_numeric_metric_columns(df, exclude_cols):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    forbidden_cols = [
        "keypoints",
        "n_keypoints",
        "num_keypoints",
        "cantidad_keypoints",
        "count",
        "cantidad",
    ]

    return [
        col for col in numeric_cols
        if col not in exclude_cols and col not in forbidden_cols
    ]

# ============================================================
# Encabezado
# ============================================================

st.markdown("""
<h1 style='text-align: center;'>Análisis de Hipótesis 1</h1>
<p style='text-align: center; color: #7F8C8D;'>
Diferencias espaciales entre grupos corporales y keypoints
</p>
""", unsafe_allow_html=True)


# ============================================================
# Verificación de archivos
# ============================================================

if not H1_DIR.exists():
    st.error(
        f"No se encontró el directorio de Hipótesis 1:\n\n"
        f"`{H1_DIR}`\n\n"
        f"Ejecuta primero: `hipotesis_1/analisis_hipotesis_1_grupos.py`"
    )
    st.stop()

keypoints_path = H1_DIR / "h1_metricas_keypoints.csv"
grupos_path = H1_DIR / "h1_metricas_grupos.csv"

df_keypoints = load_csv_if_exists(keypoints_path)
df_grupos = load_csv_if_exists(grupos_path)

missing = []
if df_keypoints is None:
    missing.append(keypoints_path.name)
if df_grupos is None:
    missing.append(grupos_path.name)

if missing:
    st.error(
        "Faltan archivos necesarios para generar gráficos dinámicos:\n\n"
        + "\n".join([f"- {name}" for name in missing])
        + "\n\nEjecuta primero el script de Hipótesis 1."
    )
    st.stop()


# ============================================================
# Resumen
# ============================================================

st.markdown("### Resumen de Hipótesis 1")

st.info(
    """
    **Pregunta:** ¿Qué grupo corporal presenta mayores diferencias espaciales
    entre frames normales y frames asociados a shoplifting?

    **Idea principal:** se comparan keypoints y grupos corporales usando posiciones
    normalizadas respecto al bounding box. Como las variables no siguen normalidad,
    se priorizan métricas no paramétricas como Mann-Whitney, tamaño de efecto y Spearman.
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Keypoints evaluados", len(df_keypoints))

with col2:
    st.metric("Grupos corporales", len(df_grupos))

with col3:
    st.metric("Archivos de resultados", 2)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.markdown("## Configuración")

keypoint_col = find_column(
    df_keypoints,
    ["keypoint", "variable", "nombre_keypoint", "punto_corporal"]
)

group_col_keypoints = find_column(
    df_keypoints,
    ["grupo", "grupo_corporal", "body_group", "region"]
)

group_col_grupos = find_column(
    df_grupos,
    ["grupo", "grupo_corporal", "body_group", "region"]
)

exclude_key_cols = [
    col for col in [keypoint_col, group_col_keypoints]
    if col is not None
]

exclude_group_cols = [
    col for col in [group_col_grupos]
    if col is not None
]

metric_cols_keypoints = get_numeric_metric_columns(df_keypoints, exclude_key_cols)
metric_cols_grupos = get_numeric_metric_columns(df_grupos, exclude_group_cols)

default_metric_keypoints = None
for candidate in [
    "effect_total",
    "effect_size_total",
    "effect_size_abs",
    "effect_size",
    "mannwhitney_effect",
    "diferencia_medianas_total",
]:
    if candidate in metric_cols_keypoints:
        default_metric_keypoints = candidate
        break

if default_metric_keypoints is None and metric_cols_keypoints:
    default_metric_keypoints = metric_cols_keypoints[0]

default_metric_grupos = None
for candidate in [
    "effect_total_mediana_grupo",
    "effect_total_promedio_grupo",
    "diff_total_mediana_grupo",
    "diff_total_promedio_grupo",
    "abs_spearman_y_promedio",
    "abs_spearman_x_promedio",
]:
    if candidate in metric_cols_grupos:
        default_metric_grupos = candidate
        break

if default_metric_grupos is None and metric_cols_grupos:
    default_metric_grupos = metric_cols_grupos[0]

metric_keypoints = st.sidebar.selectbox(
    "Métrica para ranking de keypoints:",
    metric_cols_keypoints,
    index=metric_cols_keypoints.index(default_metric_keypoints)
    if default_metric_keypoints in metric_cols_keypoints else 0,
)

metric_grupos = st.sidebar.selectbox(
    "Métrica para ranking de grupos:",
    metric_cols_grupos,
    index=metric_cols_grupos.index(default_metric_grupos)
    if default_metric_grupos in metric_cols_grupos else 0,
)

top_n = st.sidebar.slider(
    "Top keypoints a mostrar:",
    min_value=5,
    max_value=min(17, len(df_keypoints)),
    value=min(17, len(df_keypoints)),
)


# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab5 = st.tabs([
    "Ranking Keypoints",
    "Ranking Grupos",
    "Heatmap Métricas",
    "Datos CSV",
])


# ============================================================
# TAB 1
# ============================================================

with tab1:
    st.markdown("#### Ranking de keypoints por tamaño de efecto")

    fig_keypoints = plot_ranking_keypoints(
        df_keypoints,
        metric_keypoints,
        top_n,
    )

    if fig_keypoints is None:
        st.warning("No se pudo construir el ranking de keypoints. Verifica nombres de columnas del CSV.")
        st.dataframe(df_keypoints.head(), use_container_width=True)
    else:
        st.plotly_chart(fig_keypoints, use_container_width=True)

        st.info(
            "Este gráfico permite identificar qué keypoints individuales presentan "
            "mayor diferencia espacial entre comportamiento normal y shoplifting. "
            "Un valor alto no implica causalidad, pero sí indica mayor separación "
            "exploratoria entre clases."
        )

    with st.expander("Ver tabla de keypoints"):
        st.dataframe(df_keypoints, use_container_width=True)


# ============================================================
# TAB 2
# ============================================================

with tab2:
    st.markdown("#### Ranking de grupos corporales")

    fig_grupos = plot_ranking_grupos(
        df_grupos,
        metric_col="effect_total_mediana_grupo",
    )

    if fig_grupos is None:
        st.warning("No se pudo construir el ranking de grupos. Verifica nombres de columnas del CSV.")
        st.dataframe(df_grupos.head(), use_container_width=True)
    else:
        st.plotly_chart(fig_grupos, use_container_width=True)

        st.info(
            "El análisis por grupo corporal resume el comportamiento de varios "
            "keypoints de una misma región anatómica. Esto ayuda a evitar conclusiones "
            "basadas en un solo punto corporal aislado."
        )

    with st.expander("Ver tabla de grupos"):
        st.dataframe(df_grupos, use_container_width=True)


# ============================================================
# TAB 3
# ============================================================

with tab3:
    st.markdown("#### Heatmap de métricas por grupo corporal")

    selected_metrics_heatmap = st.multiselect(
        "Métricas a incluir en el heatmap:",
        metric_cols_grupos,
        default=metric_cols_grupos[:min(5, len(metric_cols_grupos))],
    )

    fig_heat = plot_heatmap_metricas_grupo(
        df_grupos,
        selected_metrics_heatmap,
    )

    if fig_heat is None:
        st.warning("No se pudo construir el heatmap. Selecciona al menos una métrica.")
    else:
        st.plotly_chart(fig_heat, use_container_width=True)

        st.info(
            "El heatmap está normalizado por métrica para poder comparar columnas "
            "con escalas diferentes. Los valores más altos indican mayor intensidad "
            "relativa dentro de cada métrica."
        )


# ============================================================
# TAB 5
# ============================================================

with tab5:
    st.markdown("#### Archivos generados")

    show_csv(
        grupos_path,
        title="Métricas agregadas por grupo corporal",
        download_name="h1_metricas_grupos.csv",
    )

    st.markdown("---")

    show_csv(
        keypoints_path,
        title="Métricas detalladas por keypoint individual",
        download_name="h1_metricas_keypoints.csv",
    )


# ============================================================
# Conclusión
# ============================================================

st.markdown("---")
st.markdown("### Conclusión")

st.success(
    """
    **H1 se acepta parcialmente.**  
    El análisis espacial muestra que las diferencias entre comportamiento normal y
    shoplifting no se concentran únicamente en muñecas y codos. Los resultados sugieren
    una configuración corporal más amplia: cabeza y caderas presentan mayor tamaño de
    efecto mediano, mientras que algunos keypoints de brazos, como codos y muñecas,
    todavía aportan información parcial.
    """
)