"""
Página 4: Análisis Hipótesis 2
Visualización dinámica de análisis temporal y autocorrelación.
Genera gráficos interactivos desde los datos y CSVs, evitando depender de imágenes pegadas.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import mannwhitneyu

from config import H2_DIR, H2_ACF_DIR
from utils import load_tabular_data
from utils.display_utils import show_csv


st.set_page_config(page_title="Análisis Hipótesis 2", layout="wide")


# ============================================================
# Configuración
# ============================================================

LABEL_COL = "label"

KEYPOINTS_TEMPORALES = [
    "left_wrist",
    "right_wrist",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
]

VARIABLES_H2 = [
    "vel_left_wrist",
    "vel_right_wrist",
    "vel_left_elbow",
    "vel_right_elbow",
    "left_wrist_hip_dist",
    "right_wrist_hip_dist",
    "left_elbow_hip_dist",
    "right_elbow_hip_dist",
    "delta_left_wrist_hip_dist",
    "delta_right_wrist_hip_dist",
    "delta_left_elbow_hip_dist",
    "delta_right_elbow_hip_dist",
]


# ============================================================
# Utilidades
# ============================================================

def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def rank_biserial_effect(x, y):
    """
    Tamaño de efecto para Mann-Whitney.
    Aproximación: r_rb = 1 - 2U/(n1*n2)
    """
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()

    if len(x) < 2 or len(y) < 2:
        return np.nan

    try:
        u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
        effect = 1 - (2 * u_stat) / (len(x) * len(y))
        return abs(effect)
    except Exception:
        return np.nan


@st.cache_data
def preparar_dataset_temporal(split: str):
    df = load_tabular_data(split=split, csv_type="raw")

    required = [
        "video_id", "frame_id", "person_id",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas necesarias: {missing}")

    df = df.copy()
    df["video_id"] = df["video_id"].astype(str)
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce")
    df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce")

    df = df.dropna(subset=["frame_id", "person_id"])
    df["frame_id"] = df["frame_id"].astype(int)
    df["person_id"] = df["person_id"].astype(int)

    if LABEL_COL in df.columns:
        df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").astype("Int64")

    df = normalizar_keypoints(df)
    df = calcular_variables_temporales(df)

    return df


def normalizar_keypoints(df: pd.DataFrame):
    df = df.copy()

    bbox_w = (df["bbox_x2"] - df["bbox_x1"]).replace(0, np.nan)
    bbox_h = (df["bbox_y2"] - df["bbox_y1"]).replace(0, np.nan)

    for kp in KEYPOINTS_TEMPORALES:
        x_col = f"{kp}_x"
        y_col = f"{kp}_y"

        if x_col in df.columns and y_col in df.columns:
            df[f"{kp}_x_norm"] = (df[x_col] - df["bbox_x1"]) / bbox_w
            df[f"{kp}_y_norm"] = (df[y_col] - df["bbox_y1"]) / bbox_h

    return df


def calcular_variables_temporales(df: pd.DataFrame):
    df = df.copy()
    df = df.sort_values(["video_id", "person_id", "frame_id"])

    group_cols = ["video_id", "person_id"]

    # Velocidades normalizadas
    for kp in ["left_wrist", "right_wrist", "left_elbow", "right_elbow"]:
        x_col = f"{kp}_x_norm"
        y_col = f"{kp}_y_norm"

        if x_col not in df.columns or y_col not in df.columns:
            continue

        dx = df.groupby(group_cols)[x_col].diff()
        dy = df.groupby(group_cols)[y_col].diff()
        dframe = df.groupby(group_cols)["frame_id"].diff().replace(0, np.nan)

        df[f"vel_{kp}"] = np.sqrt(dx**2 + dy**2) / dframe
        df[f"vel_{kp}"] = df[f"vel_{kp}"].where(dframe > 0)

    # Distancias muñeca/codo - cadera correspondiente
    pares = {
        "left_wrist_hip_dist": ("left_wrist", "left_hip"),
        "right_wrist_hip_dist": ("right_wrist", "right_hip"),
        "left_elbow_hip_dist": ("left_elbow", "left_hip"),
        "right_elbow_hip_dist": ("right_elbow", "right_hip"),
    }

    for new_col, (kp1, kp2) in pares.items():
        x1 = f"{kp1}_x_norm"
        y1 = f"{kp1}_y_norm"
        x2 = f"{kp2}_x_norm"
        y2 = f"{kp2}_y_norm"

        if all(col in df.columns for col in [x1, y1, x2, y2]):
            df[new_col] = np.sqrt(
                (df[x1] - df[x2]) ** 2 +
                (df[y1] - df[y2]) ** 2
            )

            df[f"delta_{new_col}"] = df.groupby(group_cols)[new_col].diff()

    return df


def obtener_trayectorias_validas(df: pd.DataFrame):
    counts = (
        df.groupby(["video_id", "person_id"])
        .size()
        .reset_index(name="n_frames")
        .sort_values("n_frames", ascending=False)
    )
    return counts


def calcular_ranking_temporal(df: pd.DataFrame):
    if LABEL_COL not in df.columns:
        return pd.DataFrame()

    rows = []

    for var in VARIABLES_H2:
        if var not in df.columns:
            continue

        normal = df[df[LABEL_COL] == 0][var].dropna()
        shop = df[df[LABEL_COL] == 1][var].dropna()

        effect = rank_biserial_effect(normal, shop)

        rows.append({
            "variable": var,
            "mediana_normal": normal.median(),
            "mediana_shoplifting": shop.median(),
            "diferencia_medianas": shop.median() - normal.median(),
            "effect_size_abs": effect,
            "n_normal": len(normal),
            "n_shoplifting": len(shop),
        })

    ranking_df = pd.DataFrame(rows)

    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values("effect_size_abs", ascending=False)

    return ranking_df


# ============================================================
# Gráficos dinámicos
# ============================================================

def plot_boxplot_velocidades(df: pd.DataFrame, variables: list[str]):
    if LABEL_COL not in df.columns:
        return None

    plot_cols = [col for col in variables if col in df.columns]

    plot_df = df[[LABEL_COL] + plot_cols].melt(
        id_vars=LABEL_COL,
        var_name="variable",
        value_name="valor",
    ).dropna()

    plot_df["clase"] = plot_df[LABEL_COL].map({
        0: "Normal",
        1: "Shoplifting",
    })

    fig = px.box(
        plot_df,
        x="variable",
        y="valor",
        color="clase",
        points=False,
        title="Distribución de variables temporales por clase",
        labels={
            "variable": "Variable temporal",
            "valor": "Valor normalizado",
            "clase": "Clase",
        },
    )

    fig.update_layout(height=650, xaxis_tickangle=-25)
    return fig


def plot_ranking_temporal(ranking_df: pd.DataFrame, top_n: int = 12):
    if ranking_df.empty:
        return None

    top = ranking_df.head(top_n).sort_values("effect_size_abs", ascending=True)

    fig = px.bar(
        top,
        x="effect_size_abs",
        y="variable",
        orientation="h",
        title=f"Top {top_n}: variables temporales con mayor tamaño de efecto",
        labels={
            "effect_size_abs": "Tamaño de efecto absoluto",
            "variable": "Variable",
        },
        hover_data=[
            "mediana_normal",
            "mediana_shoplifting",
            "diferencia_medianas",
            "n_normal",
            "n_shoplifting",
        ],
    )

    fig.update_layout(height=600)
    return fig


def plot_serie_temporal_trayectoria(df: pd.DataFrame, video_id: str, person_id: int, variables: list[str]):
    traj = df[
        (df["video_id"].astype(str) == str(video_id)) &
        (df["person_id"].astype(int) == int(person_id))
    ].copy()

    traj = traj.sort_values("frame_id")

    if traj.empty:
        return None

    fig = go.Figure()

    for var in variables:
        if var not in traj.columns:
            continue

        fig.add_trace(go.Scatter(
            x=traj["frame_id"],
            y=traj[var],
            mode="lines+markers",
            name=var,
        ))

    if LABEL_COL in traj.columns:
        shop_frames = traj[traj[LABEL_COL] == 1]["frame_id"]

        if len(shop_frames) > 0:
            fig.add_vrect(
                x0=shop_frames.min(),
                x1=shop_frames.max(),
                fillcolor="red",
                opacity=0.12,
                layer="below",
                line_width=0,
                annotation_text="Shoplifting",
                annotation_position="top left",
            )

    fig.update_layout(
        title=f"Serie temporal - video_id={video_id}, person_id={person_id}",
        xaxis_title="Frame",
        yaxis_title="Valor normalizado",
        height=600,
    )

    return fig


def plot_heatmap_temporal_trayectoria(df: pd.DataFrame, video_id: str, person_id: int, variables: list[str]):
    traj = df[
        (df["video_id"].astype(str) == str(video_id)) &
        (df["person_id"].astype(int) == int(person_id))
    ].copy()

    traj = traj.sort_values("frame_id")

    if traj.empty:
        return None

    available = [var for var in variables if var in traj.columns]

    if not available:
        return None

    matrix = traj[available].T

    # Normalización por variable para visualización
    matrix_norm = matrix.sub(matrix.min(axis=1), axis=0)
    denom = matrix.max(axis=1) - matrix.min(axis=1)
    matrix_norm = matrix_norm.div(denom.replace(0, np.nan), axis=0)

    fig = px.imshow(
        matrix_norm,
        aspect="auto",
        color_continuous_scale="Viridis",
        title=f"Mapa temporal de variables dinámicas - video_id={video_id}, person_id={person_id}",
        labels={
            "x": "Índice temporal",
            "y": "Variable",
            "color": "Valor normalizado",
        },
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(traj))),
        ticktext=traj["frame_id"].astype(str).tolist(),
    )

    fig.update_layout(height=650)

    return fig


def plot_acf_lag1(acf_lag1_df: pd.DataFrame):
    required = {"grupo_corporal", "acf_lag1_normal", "acf_lag1_shoplifting"}

    if not required.issubset(acf_lag1_df.columns):
        return None

    plot_df = acf_lag1_df.melt(
        id_vars="grupo_corporal",
        value_vars=["acf_lag1_normal", "acf_lag1_shoplifting"],
        var_name="clase",
        value_name="acf_lag1",
    )

    plot_df["clase"] = plot_df["clase"].map({
        "acf_lag1_normal": "Normal",
        "acf_lag1_shoplifting": "Shoplifting",
    })

    fig = px.bar(
        plot_df,
        x="grupo_corporal",
        y="acf_lag1",
        color="clase",
        barmode="group",
        title="Autocorrelación lag 1 por grupo corporal",
        labels={
            "grupo_corporal": "Grupo corporal",
            "acf_lag1": "ACF lag 1 promedio",
            "clase": "Clase",
        },
    )

    fig.update_layout(height=550)
    return fig


def plot_curvas_acf(acf_group_df: pd.DataFrame):
    required = {"grupo_corporal", "label", "lag", "acf_promedio_grupo"}

    if not required.issubset(acf_group_df.columns):
        return None

    plot_df = acf_group_df.copy()
    plot_df["clase"] = plot_df["label"].map({
        0: "Normal",
        1: "Shoplifting",
    })

    fig = px.line(
        plot_df,
        x="lag",
        y="acf_promedio_grupo",
        color="clase",
        facet_col="grupo_corporal",
        facet_col_wrap=3,
        markers=True,
        title="Curvas de autocorrelación por grupo corporal y clase",
        labels={
            "lag": "Lag",
            "acf_promedio_grupo": "ACF promedio",
            "clase": "Clase",
        },
    )

    fig.update_yaxes(range=[-1, 1])
    fig.update_layout(height=850)

    return fig


# ============================================================
# Encabezado
# ============================================================

st.markdown("""
<h1 style='text-align: center;'>Análisis de Hipótesis 2</h1>
<p style='text-align: center; color: #7F8C8D;'>
Patrones temporales: velocidades, distancias, autocorrelación y evolución de la pose
</p>
""", unsafe_allow_html=True)


# ============================================================
# Sidebar y carga
# ============================================================

st.sidebar.markdown("## Configuración")
split = st.sidebar.radio("Selecciona split:", ["test", "train"], index=0)

try:
    df_temp = preparar_dataset_temporal(split)
except Exception as e:
    st.error(f"No se pudo preparar el dataset temporal: {e}")
    st.stop()

st.sidebar.markdown("### Variables temporales")
available_h2_vars = [var for var in VARIABLES_H2 if var in df_temp.columns]

default_vars = [
    var for var in [
        "vel_left_wrist",
        "vel_right_wrist",
        "vel_left_elbow",
        "vel_right_elbow",
        "left_wrist_hip_dist",
        "right_wrist_hip_dist",
        "left_elbow_hip_dist",
        "right_elbow_hip_dist",
    ]
    if var in available_h2_vars
]

selected_vars = st.sidebar.multiselect(
    "Variables a visualizar:",
    available_h2_vars,
    default=default_vars,
)

st.markdown("### Resumen de Hipótesis 2")

st.info(
    """
    **Pregunta:** ¿La variabilidad, velocidad o desplazamiento relativo de muñecas y codos
    respecto al torso/cadera difiere entre comportamiento normal y shoplifting?

    **Idea principal:** esta hipótesis incorpora la dimensión temporal del dataset.
    En lugar de mirar solo la pose en un frame, se calcula cómo cambian las posiciones
    corporales entre frames consecutivos.
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Registros", f"{len(df_temp):,}")
with col2:
    st.metric("Variables temporales calculadas", len(available_h2_vars))
with col3:
    if LABEL_COL in df_temp.columns:
        st.metric("Clases disponibles", df_temp[LABEL_COL].nunique())
    else:
        st.metric("Clases disponibles", "Sin label")


# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Boxplot Velocidad",
    "Ranking Features",
    "Serie Temporal",
    "Heatmap Temporal",
    "Autocorrelación",
    "Datos CSV",
])


# ============================================================
# TAB 1
# ============================================================

with tab1:
    st.markdown("#### Distribución de variables temporales por clase")

    if LABEL_COL not in df_temp.columns:
        st.warning("El split seleccionado no contiene label. Usa test para comparar Normal vs Shoplifting.")
    elif not selected_vars:
        st.warning("Selecciona al menos una variable temporal en la barra lateral.")
    else:
        fig_box = plot_boxplot_velocidades(df_temp, selected_vars)
        st.plotly_chart(fig_box, use_container_width=True)

        st.info(
            "Este gráfico permite comparar si las velocidades o distancias relativas "
            "presentan distribuciones diferentes entre comportamiento normal y shoplifting."
        )


# ============================================================
# TAB 2
# ============================================================

with tab2:
    st.markdown("#### Ranking de variables temporales")

    if LABEL_COL not in df_temp.columns:
        st.warning("El ranking supervisado requiere label. Usa el split test.")
    else:
        ranking_df = calcular_ranking_temporal(df_temp)

        if ranking_df.empty:
            st.warning("No se pudo calcular el ranking temporal.")
        else:
            top_n = st.slider("Top variables:", 5, min(20, len(ranking_df)), min(12, len(ranking_df)))
            fig_rank = plot_ranking_temporal(ranking_df, top_n=top_n)
            st.plotly_chart(fig_rank, use_container_width=True)
            st.dataframe(ranking_df, use_container_width=True)

            st.info(
                "El tamaño de efecto resume qué variables temporales separan más ambas clases. "
                "Valores más altos indican mayor diferencia entre comportamiento normal y shoplifting."
            )


# ============================================================
# TAB 3
# ============================================================

with tab3:
    st.markdown("#### Serie temporal por trayectoria")

    trayectorias = obtener_trayectorias_validas(df_temp)

    if trayectorias.empty:
        st.warning("No se encontraron trayectorias disponibles.")
    else:
        video_options = trayectorias["video_id"].astype(str).drop_duplicates().tolist()

        selected_video = st.selectbox("Selecciona video_id:", video_options)

        persons_for_video = (
            trayectorias[trayectorias["video_id"].astype(str) == str(selected_video)]
            ["person_id"]
            .astype(int)
            .tolist()
        )

        selected_person = st.selectbox("Selecciona person_id:", persons_for_video)

        fig_series = plot_serie_temporal_trayectoria(
            df_temp,
            selected_video,
            selected_person,
            selected_vars,
        )

        if fig_series is None:
            st.warning("No se pudo construir la serie temporal.")
        else:
            st.plotly_chart(fig_series, use_container_width=True)

            st.info(
                "La zona sombreada en rojo representa frames etiquetados como shoplifting, "
                "si el split contiene label. Esto permite observar cambios temporales "
                "dentro de una misma trayectoria."
            )


# ============================================================
# TAB 4
# ============================================================

with tab4:
    st.markdown("#### Mapa temporal de variables dinámicas")

    trayectorias = obtener_trayectorias_validas(df_temp)

    if trayectorias.empty:
        st.warning("No se encontraron trayectorias disponibles.")
    else:
        video_options = trayectorias["video_id"].astype(str).drop_duplicates().tolist()

        selected_video_h = st.selectbox(
            "Selecciona video_id:",
            video_options,
            key="heatmap_video",
        )

        persons_for_video_h = (
            trayectorias[trayectorias["video_id"].astype(str) == str(selected_video_h)]
            ["person_id"]
            .astype(int)
            .tolist()
        )

        selected_person_h = st.selectbox(
            "Selecciona person_id:",
            persons_for_video_h,
            key="heatmap_person",
        )

        fig_heat = plot_heatmap_temporal_trayectoria(
            df_temp,
            selected_video_h,
            selected_person_h,
            selected_vars,
        )

        if fig_heat is None:
            st.warning("No se pudo construir el mapa temporal.")
        else:
            st.plotly_chart(fig_heat, use_container_width=True)

            st.info(
                "El mapa temporal normaliza cada variable para observar cambios relativos "
                "a lo largo de los frames de una misma persona."
            )


# ============================================================
# TAB 5
# ============================================================

with tab5:
    st.markdown("#### Autocorrelación temporal")

    st.info(
        """
        La autocorrelación mide si el movimiento en un frame depende del movimiento
        en frames anteriores. En datos de pose humana, esto es importante porque
        confirma que el dataset tiene estructura temporal.
        """
    )

    acf_lag1_path = H2_ACF_DIR / "acf_lag1_por_grupo_corporal.csv"
    acf_group_path = H2_ACF_DIR / "acf_promedio_por_grupo_corporal.csv"

    acf_lag1_df = load_csv_if_exists(acf_lag1_path)
    acf_group_df = load_csv_if_exists(acf_group_path)

    if acf_lag1_df is None:
        st.warning(f"No se encontró: {acf_lag1_path}")
    else:
        fig_acf_lag1 = plot_acf_lag1(acf_lag1_df)

        if fig_acf_lag1 is not None:
            st.plotly_chart(fig_acf_lag1, use_container_width=True)

        st.dataframe(acf_lag1_df, use_container_width=True)

    if acf_group_df is None:
        st.warning(f"No se encontró: {acf_group_path}")
    else:
        fig_acf_curves = plot_curvas_acf(acf_group_df)

        if fig_acf_curves is not None:
            st.plotly_chart(fig_acf_curves, use_container_width=True)

        with st.expander("Ver datos ACF por grupo y lag"):
            st.dataframe(acf_group_df, use_container_width=True)


# ============================================================
# TAB 6
# ============================================================

with tab6:
    st.markdown("#### Archivos generados para Hipótesis 2")

    h2_metrics_path = H2_DIR / "h2_metricas_temporales.csv"

    if h2_metrics_path.exists():
        show_csv(
            h2_metrics_path,
            title="Métricas temporales generadas",
            download_name="h2_metricas_temporales.csv",
        )
    else:
        st.warning(f"No se encontró: {h2_metrics_path}")

    if H2_ACF_DIR.exists():
        for csv_path in sorted(H2_ACF_DIR.glob("*.csv")):
            show_csv(
                csv_path,
                title=f"Autocorrelación - {csv_path.stem}",
                download_name=csv_path.name,
                show_warning=False,
            )
    else:
        st.warning(f"No se encontró el directorio de autocorrelación: {H2_ACF_DIR}")


# ============================================================
# Conclusión
# ============================================================

st.markdown("---")
st.markdown("### Conclusión")

st.success(
    """
    **H2 se acepta parcialmente.**  
    El análisis temporal muestra que las diferencias entre comportamiento normal y
    shoplifting no dependen de un único movimiento aislado. Las velocidades de algunas
    extremidades superiores, las distancias brazo-cadera y la autocorrelación temporal
    aportan evidencia de patrones dinámicos distintos entre clases.
    """
)