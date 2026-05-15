"""
Página 5: Análisis Hipótesis 3
Visualización dinámica del análisis de inicio de anomalías.
Genera gráficos interactivos desde los CSVs producidos por el análisis.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import H3_DIR
from utils.display_utils import show_csv, show_text_file


st.set_page_config(page_title="Análisis Hipótesis 3", layout="wide")


# ============================================================
# Configuración
# ============================================================

VARIABLES_VERTICALES = [
    "left_wrist_hip_vertical",
    "right_wrist_hip_vertical",
    "wrist_center_hip_vertical",
]

VARIABLES_DISTANCIA = [
    "left_wrist_hip_dist",
    "right_wrist_hip_dist",
    "wrist_center_hip_dist",
]

VARIABLES_H3 = VARIABLES_VERTICALES + VARIABLES_DISTANCIA

VARIABLES_HEATMAP = [
    "left_wrist_y_norm",
    "right_wrist_y_norm",
    "left_hip_y_norm",
    "right_hip_y_norm",
    "left_wrist_hip_vertical",
    "right_wrist_hip_vertical",
    "wrist_center_hip_vertical",
    "left_wrist_hip_dist",
    "right_wrist_hip_dist",
    "wrist_center_hip_dist",
]


# ============================================================
# Utilidades
# ============================================================

def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def add_periodo(ventanas_df: pd.DataFrame) -> pd.DataFrame:
    df = ventanas_df.copy()

    def periodo(relative_frame):
        if relative_frame < 0:
            return "Antes"
        if relative_frame <= 5:
            return "Inicio"
        return "Después"

    df["periodo"] = df["relative_frame"].apply(periodo)

    return df


def plot_boxplot_ventanas(ventanas_df: pd.DataFrame, selected_vars: list[str]):
    df = add_periodo(ventanas_df)

    available_vars = [var for var in selected_vars if var in df.columns]

    plot_df = df[["periodo"] + available_vars].melt(
        id_vars="periodo",
        var_name="variable",
        value_name="valor",
    ).dropna()

    periodo_order = ["Antes", "Inicio", "Después"]

    fig = px.box(
        plot_df,
        x="periodo",
        y="valor",
        color="periodo",
        facet_col="variable",
        facet_col_wrap=3,
        category_orders={"periodo": periodo_order},
        points=False,
        title="Comparación antes, inicio y después del comportamiento anómalo",
        labels={
            "periodo": "Periodo temporal",
            "valor": "Valor normalizado",
            "variable": "Variable",
        },
    )

    fig.update_yaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("variable=", "")))
    fig.update_layout(height=max(500, 260 * int(np.ceil(len(available_vars) / 3))))

    return fig


def plot_perfil_transiciones(ventanas_df: pd.DataFrame, selected_vars: list[str], agg_method: str):
    available_vars = [var for var in selected_vars if var in ventanas_df.columns]

    if agg_method == "Media":
        agg_df = (
            ventanas_df
            .groupby("relative_frame")[available_vars]
            .mean()
            .reset_index()
            .sort_values("relative_frame")
        )
    else:
        agg_df = (
            ventanas_df
            .groupby("relative_frame")[available_vars]
            .median()
            .reset_index()
            .sort_values("relative_frame")
        )

    fig = go.Figure()

    for var in available_vars:
        fig.add_trace(go.Scatter(
            x=agg_df["relative_frame"],
            y=agg_df[var],
            mode="lines+markers",
            name=var,
        ))

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Inicio shoplifting",
        annotation_position="top left",
    )

    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray",
    )

    fig.update_layout(
        title=f"Perfil {agg_method.lower()} alrededor de transiciones normal → shoplifting",
        xaxis_title="Frame relativo al inicio de shoplifting",
        yaxis_title="Valor normalizado",
        height=600,
    )

    return fig


def get_transition_options(transiciones_df: pd.DataFrame):
    df = transiciones_df.copy()

    if "transition_frame" not in df.columns:
        return []

    df["option"] = (
        "video=" + df["video_id"].astype(str)
        + " | person=" + df["person_id"].astype(str)
        + " | inicio=" + df["transition_frame"].astype(str)
    )

    return df["option"].tolist()


def parse_transition_option(option: str):
    parts = option.split("|")
    video_id = parts[0].replace("video=", "").strip()
    person_id = int(parts[1].replace("person=", "").strip())
    transition_frame = int(parts[2].replace("inicio=", "").strip())

    return video_id, person_id, transition_frame


def filter_transition_window(ventanas_df: pd.DataFrame, video_id: str, person_id: int, transition_frame: int):
    df = ventanas_df[
        (ventanas_df["video_id"].astype(str) == str(video_id))
        & (ventanas_df["person_id"].astype(int) == int(person_id))
        & (ventanas_df["transition_frame"].astype(int) == int(transition_frame))
    ].copy()

    return df.sort_values("relative_frame")


def plot_serie_transicion(ventana_df: pd.DataFrame, selected_vars: list[str], video_id: str, person_id: int, transition_frame: int):
    available_vars = [var for var in selected_vars if var in ventana_df.columns]

    fig = go.Figure()

    for var in available_vars:
        fig.add_trace(go.Scatter(
            x=ventana_df["relative_frame"],
            y=ventana_df[var],
            mode="lines+markers",
            name=var,
        ))

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Inicio shoplifting",
        annotation_position="top left",
    )

    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray",
    )

    fig.update_layout(
        title=(
            f"Serie muñeca-cadera alrededor del inicio anómalo<br>"
            f"video_id={video_id}, person_id={person_id}, frame_inicio={transition_frame}"
        ),
        xaxis_title="Frame relativo al inicio de shoplifting",
        yaxis_title="Valor normalizado",
        height=600,
    )

    return fig


def plot_heatmap_transicion(ventana_df: pd.DataFrame, selected_vars: list[str], video_id: str, person_id: int, transition_frame: int):
    available_vars = [var for var in selected_vars if var in ventana_df.columns]

    if not available_vars:
        return None

    matrix = ventana_df[available_vars].T

    matrix_norm = matrix.sub(matrix.min(axis=1), axis=0)
    denom = matrix.max(axis=1) - matrix.min(axis=1)
    matrix_norm = matrix_norm.div(denom.replace(0, np.nan), axis=0)

    fig = px.imshow(
        matrix_norm,
        aspect="auto",
        color_continuous_scale="Viridis",
        title=(
            f"Mapa temporal alrededor del inicio anómalo<br>"
            f"video_id={video_id}, person_id={person_id}, frame_inicio={transition_frame}"
        ),
        labels={
            "x": "Frame relativo",
            "y": "Variable",
            "color": "Valor normalizado",
        },
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(ventana_df))),
        ticktext=ventana_df["relative_frame"].astype(int).astype(str).tolist(),
    )

    # posición del frame relativo 0 dentro de la ventana
    zero_positions = np.where(ventana_df["relative_frame"].to_numpy() == 0)[0]
    if len(zero_positions) > 0:
        fig.add_vline(
            x=int(zero_positions[0]),
            line_dash="dash",
            line_color="red",
        )

    fig.update_layout(height=650)

    return fig


def plot_resumen_cambios(resumen_df: pd.DataFrame):
    required = {
        "variable",
        "delta_inicio_vs_antes",
        "delta_despues_vs_antes",
    }

    if not required.issubset(resumen_df.columns):
        return None

    plot_df = resumen_df[
        ["variable", "delta_inicio_vs_antes", "delta_despues_vs_antes"]
    ].melt(
        id_vars="variable",
        var_name="comparacion",
        value_name="delta",
    )

    plot_df["comparacion"] = plot_df["comparacion"].map({
        "delta_inicio_vs_antes": "Inicio - Antes",
        "delta_despues_vs_antes": "Después - Antes",
    })

    fig = px.bar(
        plot_df,
        x="delta",
        y="variable",
        color="comparacion",
        barmode="group",
        orientation="h",
        title="Cambios de mediana respecto al periodo anterior",
        labels={
            "delta": "Cambio de mediana",
            "variable": "Variable",
            "comparacion": "Comparación",
        },
    )

    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(height=600)

    return fig


# ============================================================
# Encabezado
# ============================================================

st.markdown("""
<h1 style='text-align: center;'>Análisis de Hipótesis 3</h1>
<p style='text-align: center; color: #7F8C8D;'>
Detección del inicio de anomalías y transiciones normal → shoplifting
</p>
""", unsafe_allow_html=True)


# ============================================================
# Verificación de archivos
# ============================================================

st.sidebar.markdown("## Configuración")

if not H3_DIR.exists():
    st.error(
        f"No se encontró el directorio de Hipótesis 3:\n\n"
        f"`{H3_DIR}`\n\n"
        f"Ejecuta primero: `hipotesis_3/analisis_hipotesis_3_inicio_anomalia.py`"
    )
    st.stop()

resumen_path = H3_DIR / "h3_resumen_ventanas.csv"
transiciones_path = H3_DIR / "h3_transiciones_detectadas.csv"
ventanas_path = H3_DIR / "h3_ventanas_transicion.csv"
conclusion_path = H3_DIR / "h3_conclusion_inicio_anomalia.txt"

resumen_df = load_csv_if_exists(resumen_path)
transiciones_df = load_csv_if_exists(transiciones_path)
ventanas_df = load_csv_if_exists(ventanas_path)

missing_files = []
for path in [resumen_path, transiciones_path, ventanas_path]:
    if not path.exists():
        missing_files.append(path.name)

if missing_files:
    st.error(
        "Faltan archivos necesarios para generar gráficos dinámicos:\n\n"
        + "\n".join([f"- {name}" for name in missing_files])
        + "\n\nEjecuta primero el script de Hipótesis 3."
    )
    st.stop()


# ============================================================
# Resumen
# ============================================================

st.markdown("### Resumen de Hipótesis 3")

st.info(
    """
    **Pregunta:** ¿El inicio del comportamiento anómalo puede identificarse mediante
    cambios en la posición relativa de las muñecas respecto a la cadera o al bounding box?

    **Idea principal:** se analizan ventanas temporales alrededor del primer frame
    etiquetado como shoplifting para observar si cambian las distancias y posiciones
    relativas entre muñecas y caderas.
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Transiciones detectadas", len(transiciones_df))

with col2:
    st.metric("Registros en ventanas", f"{len(ventanas_df):,}")

with col3:
    st.metric("Variables resumen", len(resumen_df))


# ============================================================
# Sidebar de variables
# ============================================================

st.sidebar.markdown("### Variables")

selected_vars = st.sidebar.multiselect(
    "Variables a analizar:",
    VARIABLES_H3,
    default=VARIABLES_H3,
)

agg_method = st.sidebar.radio(
    "Agregación para perfil:",
    ["Mediana", "Media"],
    index=0,
)


# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Boxplot Ventanas",
    "Perfil Transiciones",
    "Serie por Transición",
    "Heatmap Transición",
    "Resumen Cambios",
    "Datos",
])


# ============================================================
# TAB 1
# ============================================================

with tab1:
    st.markdown("#### Comparación antes, inicio y después")

    if not selected_vars:
        st.warning("Selecciona al menos una variable en la barra lateral.")
    else:
        fig_box = plot_boxplot_ventanas(ventanas_df, selected_vars)
        st.plotly_chart(fig_box, use_container_width=True)

        st.info(
            "Este gráfico compara las variables muñeca-cadera en tres momentos: "
            "antes del inicio anómalo, durante los primeros frames de shoplifting "
            "y después del inicio."
        )


# ============================================================
# TAB 2
# ============================================================

with tab2:
    st.markdown("#### Perfil agregado alrededor de transiciones")

    if not selected_vars:
        st.warning("Selecciona al menos una variable en la barra lateral.")
    else:
        fig_profile = plot_perfil_transiciones(
            ventanas_df,
            selected_vars,
            agg_method=agg_method,
        )
        st.plotly_chart(fig_profile, use_container_width=True)

        st.info(
            "El frame relativo 0 corresponde al primer frame etiquetado como "
            "shoplifting. La línea roja marca el punto de transición."
        )


# ============================================================
# TAB 3
# ============================================================

with tab3:
    st.markdown("#### Serie temporal de una transición específica")

    transition_options = get_transition_options(transiciones_df)

    if not transition_options:
        st.warning("No hay transiciones disponibles.")
    else:
        default_idx = 0

        # Buscar por defecto la trayectoria representativa usada en el informe
        for idx, option in enumerate(transition_options):
            if "video=1_222" in option and "person=3" in option and "inicio=32" in option:
                default_idx = idx
                break

        selected_option = st.selectbox(
            "Selecciona transición:",
            transition_options,
            index=default_idx,
        )

        video_id, person_id, transition_frame = parse_transition_option(selected_option)

        ventana_selected = filter_transition_window(
            ventanas_df,
            video_id,
            person_id,
            transition_frame,
        )

        if ventana_selected.empty:
            st.warning("No se encontraron datos para la transición seleccionada.")
        else:
            fig_series = plot_serie_transicion(
                ventana_selected,
                selected_vars,
                video_id,
                person_id,
                transition_frame,
            )

            st.plotly_chart(fig_series, use_container_width=True)

            st.info(
                "Esta vista permite inspeccionar una trayectoria concreta antes y después "
                "del inicio anómalo. Es útil para explicar visualmente cómo cambian las "
                "relaciones muñeca-cadera en una persona específica."
            )


# ============================================================
# TAB 4
# ============================================================

with tab4:
    st.markdown("#### Heatmap temporal de una transición")

    transition_options = get_transition_options(transiciones_df)

    if not transition_options:
        st.warning("No hay transiciones disponibles.")
    else:
        default_idx = 0

        for idx, option in enumerate(transition_options):
            if "video=1_222" in option and "person=3" in option and "inicio=32" in option:
                default_idx = idx
                break

        selected_option_h = st.selectbox(
            "Selecciona transición:",
            transition_options,
            index=default_idx,
            key="heatmap_transition",
        )

        video_id_h, person_id_h, transition_frame_h = parse_transition_option(selected_option_h)

        ventana_selected_h = filter_transition_window(
            ventanas_df,
            video_id_h,
            person_id_h,
            transition_frame_h,
        )

        heatmap_vars = st.multiselect(
            "Variables del heatmap:",
            [var for var in VARIABLES_HEATMAP if var in ventanas_df.columns],
            default=[var for var in VARIABLES_HEATMAP if var in ventanas_df.columns],
        )

        if ventana_selected_h.empty:
            st.warning("No se encontraron datos para la transición seleccionada.")
        elif not heatmap_vars:
            st.warning("Selecciona al menos una variable para el heatmap.")
        else:
            fig_heat = plot_heatmap_transicion(
                ventana_selected_h,
                heatmap_vars,
                video_id_h,
                person_id_h,
                transition_frame_h,
            )

            if fig_heat is None:
                st.warning("No se pudo construir el heatmap.")
            else:
                st.plotly_chart(fig_heat, use_container_width=True)

                st.info(
                    "El heatmap normaliza cada variable por fila para resaltar cambios "
                    "relativos a lo largo de la ventana temporal."
                )


# ============================================================
# TAB 5
# ============================================================

with tab5:
    st.markdown("#### Resumen de cambios de mediana")

    fig_delta = plot_resumen_cambios(resumen_df)

    if fig_delta is not None:
        st.plotly_chart(fig_delta, use_container_width=True)

    st.dataframe(resumen_df, use_container_width=True)

    st.info(
        "Los cambios positivos o negativos indican cómo se modifica la mediana de cada "
        "variable desde el periodo anterior hacia el inicio o hacia el periodo posterior."
    )


# ============================================================
# TAB 6
# ============================================================

with tab6:
    st.markdown("#### Archivos generados")

    show_csv(
        resumen_path,
        title="Resumen de ventanas",
        download_name="h3_resumen_ventanas.csv",
    )

    show_csv(
        transiciones_path,
        title="Transiciones detectadas",
        download_name="h3_transiciones_detectadas.csv",
    )

    with st.expander("Ver ventanas de transición completas"):
        st.dataframe(ventanas_df, use_container_width=True)


# ============================================================
# Conclusión
# ============================================================

st.markdown("---")
st.markdown("### Conclusión")

st.success(
    """
    **H3 se acepta parcialmente.**  
    Se observan cambios en la posición relativa entre muñecas y caderas alrededor
    del inicio de shoplifting, especialmente en las distancias normalizadas. Sin embargo,
    no se confirma un descenso uniforme de ambas muñecas hacia la cadera. Por ello,
    el inicio anómalo se interpreta como una reorganización corporal más compleja,
    no como una señal simple de una sola coordenada.
    """
)