"""
Página 2: Estadísticas Descriptivas y AED
Análisis estadístico de distribuciones, normalidad, nulos, correlaciones e información mutua.
Los gráficos se construyen dinámicamente dentro del dashboard.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import streamlit as st

from utils import load_tabular_data
from config import PREPROCESSING_AED


st.set_page_config(page_title="Estadísticas Descriptivas", layout="wide")


# ============================================================
# Utilidades
# ============================================================

ID_COLS = ["video_id", "frame_id", "person_id"]
LABEL_COL = "label"


def get_numeric_pose_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in ID_COLS + [LABEL_COL]]


def classify_variable_group(col: str) -> str:
    if col.startswith("bbox_"):
        return "Bounding box"
    if col.endswith("_conf"):
        return "Confianza"
    if col.endswith("_x"):
        return "Coordenadas X"
    if col.endswith("_y"):
        return "Coordenadas Y"
    return "Otras"


def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def plot_label_distribution(df: pd.DataFrame):
    label_counts = df[LABEL_COL].value_counts().sort_index()
    label_df = pd.DataFrame({
        "label": label_counts.index,
        "cantidad": label_counts.values,
    })
    label_df["clase"] = label_df["label"].map({
        0: "Normal",
        1: "Shoplifting",
    }).fillna(label_df["label"].astype(str))
    label_df["porcentaje"] = label_df["cantidad"] / label_df["cantidad"].sum() * 100

    fig = px.bar(
        label_df,
        x="clase",
        y="cantidad",
        text=label_df["porcentaje"].map(lambda x: f"{x:.2f}%"),
        title="Distribución de la variable label",
        labels={"clase": "Clase", "cantidad": "Cantidad de registros"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_title="Cantidad de registros")

    return fig, label_df


def plot_nulls(df: pd.DataFrame):
    nulls = df.isna().sum()
    nulls_df = pd.DataFrame({
        "columna": nulls.index,
        "nulos": nulls.values,
        "porcentaje": (nulls.values / len(df)) * 100,
    })
    nulls_df = nulls_df.sort_values("nulos", ascending=False)

    fig = px.bar(
        nulls_df[nulls_df["nulos"] > 0],
        x="nulos",
        y="columna",
        orientation="h",
        title="Valores nulos por columna",
        labels={"nulos": "Cantidad de nulos", "columna": "Columna"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    return fig, nulls_df


def plot_histograms(df: pd.DataFrame, selected_cols: list[str]):
    plot_df = df[selected_cols].melt(
        var_name="variable",
        value_name="valor",
    ).dropna()

    fig = px.histogram(
        plot_df,
        x="valor",
        facet_col="variable",
        facet_col_wrap=3,
        nbins=40,
        title="Histogramas de variables seleccionadas",
        labels={"valor": "Valor"},
    )
    fig.update_yaxes(matches=None)
    fig.update_xaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("variable=", "")))
    fig.update_layout(height=max(450, 230 * int(np.ceil(len(selected_cols) / 3))))

    return fig


def plot_qq(df: pd.DataFrame, col: str):
    data = pd.to_numeric(df[col], errors="coerce").dropna()

    if len(data) > 5000:
        data = data.sample(5000, random_state=42)

    osm, osr = stats.probplot(data, dist="norm", fit=False)
    slope, intercept, r = stats.probplot(data, dist="norm", fit=True)[1]

    line_x = np.array([min(osm), max(osm)])
    line_y = slope * line_x + intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=osm,
        y=osr,
        mode="markers",
        name="Cuantiles observados",
        marker=dict(size=5, opacity=0.65),
    ))
    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="Recta normal teórica",
    ))
    fig.update_layout(
        title=f"Q-Q plot: {col}",
        xaxis_title="Cuantiles teóricos normales",
        yaxis_title="Cuantiles observados",
        height=500,
    )

    return fig


def plot_normality_summary(norm_df: pd.DataFrame):
    if "conclusion_normalidad" in norm_df.columns:
        counts = norm_df["conclusion_normalidad"].value_counts().reset_index()
        counts.columns = ["conclusion", "cantidad"]

        fig = px.bar(
            counts,
            x="conclusion",
            y="cantidad",
            text="cantidad",
            title="Resumen de normalidad de variables numéricas",
            labels={"conclusion": "Conclusión", "cantidad": "Cantidad de variables"},
        )
        fig.update_traces(textposition="outside")
        return fig

    return None


def plot_spearman_heatmap(df: pd.DataFrame, cols: list[str]):
    corr = df[cols].corr(method="spearman")

    fig = px.imshow(
        corr,
        text_auto=False,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Matriz de correlación de Spearman",
        aspect="auto",
    )
    fig.update_layout(height=850)

    return fig


def plot_pearson_spearman(comparison_df: pd.DataFrame, top_n: int = 20):
    df_top = comparison_df.copy()

    if "abs_difference" not in df_top.columns:
        df_top["abs_difference"] = (
            df_top["pearson"] - df_top["spearman"]
        ).abs()

    df_top = df_top.sort_values("abs_difference", ascending=False).head(top_n)
    df_top = df_top.sort_values("abs_difference", ascending=True)

    fig = px.bar(
        df_top,
        x="abs_difference",
        y="variable",
        orientation="h",
        title=f"Top {top_n}: diferencia absoluta entre Pearson y Spearman",
        labels={
            "abs_difference": "|Pearson - Spearman|",
            "variable": "Variable",
        },
    )
    fig.update_layout(height=650)

    return fig


def plot_mutual_info(mi_df: pd.DataFrame, top_n: int = 20):
    df_top = mi_df.sort_values("mutual_info_label", ascending=False).head(top_n)
    df_top = df_top.sort_values("mutual_info_label", ascending=True)

    fig = px.bar(
        df_top,
        x="mutual_info_label",
        y="variable",
        orientation="h",
        title=f"Top {top_n}: variables con mayor información mutua con label",
        labels={
            "mutual_info_label": "Información mutua con label",
            "variable": "Variable",
        },
    )
    fig.update_layout(height=650)

    return fig


# ============================================================
# Encabezado
# ============================================================

st.markdown("""
<h1 style='text-align: center;'>Estadísticas Descriptivas y AED</h1>
<p style='text-align: center; color: #7F8C8D;'>
Análisis dinámico de distribución, normalidad, nulos, correlación e información mutua
</p>
""", unsafe_allow_html=True)


# ============================================================
# Carga de datos
# ============================================================

st.sidebar.markdown("## Configuración")
split = st.sidebar.radio("Selecciona split:", ["test", "train"], index=0)

try:
    df = load_tabular_data(split=split, csv_type="raw")
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

numeric_pose_cols = get_numeric_pose_columns(df)

st.write(
    f"**Dataset:** {split} | "
    f"**Filas:** {len(df):,} | "
    f"**Columnas:** {len(df.columns)}"
)

st.markdown("---")


# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Descripción Dataset",
    "Distribución de Label",
    "Nulos",
    "Normalidad",
    "Correlaciones",
    "Información Mutua",
])


# ============================================================
# TAB 1
# ============================================================

with tab1:
    st.markdown("### Descripción del Dataset")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Registros", f"{len(df):,}")
    with col2:
        st.metric("Columnas", len(df.columns))
    with col3:
        st.metric("Memoria (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")
    with col4:
        st.metric("Valores faltantes", f"{df.isna().sum().sum():,}")

    st.info(
        "Unidad de análisis: cada registro representa una persona detectada "
        "en un frame específico, identificada por (video_id, frame_id, person_id). "
        "En test, la etiqueta label está definida a nivel de frame, no de persona."
    )

    st.markdown("#### Estructura de variables")
    estructura_df = pd.DataFrame([
        {"Grupo": "Identificadores", "Columnas": "video_id, frame_id, person_id", "Cantidad": 3},
        {"Grupo": "Bounding box", "Columnas": "bbox_x1, bbox_y1, bbox_x2, bbox_y2", "Cantidad": 4},
        {"Grupo": "Keypoints COCO17", "Columnas": "x, y, conf por cada keypoint", "Cantidad": 51},
        {"Grupo": "Etiqueta", "Columnas": "label solo en test", "Cantidad": 1 if LABEL_COL in df.columns else 0},
    ])
    st.dataframe(estructura_df, use_container_width=True)

    st.markdown("#### Estadísticas descriptivas")
    st.dataframe(df.describe(include="all"), use_container_width=True)

    st.markdown("#### Primeras filas")
    st.dataframe(df.head(10), use_container_width=True)


# ============================================================
# TAB 2
# ============================================================

with tab2:
    st.markdown("### Distribución de la variable objetivo")

    if LABEL_COL not in df.columns:
        st.warning("El dataset seleccionado no contiene la columna label. Esto es esperable en train.")
    else:
        fig_label, label_df = plot_label_distribution(df)

        col1, col2, col3 = st.columns(3)
        normal_count = int(label_df.loc[label_df["label"] == 0, "cantidad"].sum())
        shop_count = int(label_df.loc[label_df["label"] == 1, "cantidad"].sum())
        total = normal_count + shop_count

        with col1:
            st.metric("Normal", f"{normal_count:,}")
        with col2:
            st.metric("Shoplifting", f"{shop_count:,}")
        with col3:
            st.metric("Shoplifting (%)", f"{shop_count / total * 100:.2f}%" if total else "0%")

        st.plotly_chart(fig_label, use_container_width=True)
        st.dataframe(label_df, use_container_width=True)

        st.info(
            "La distribución permite verificar si existe desbalance entre clases. "
            "En test, las clases son relativamente cercanas, lo que permite comparar "
            "patrones de comportamiento normal y shoplifting sin un desbalance extremo."
        )


# ============================================================
# TAB 3
# ============================================================

with tab3:
    st.markdown("### Análisis de valores nulos")

    fig_nulls, nulls_df = plot_nulls(df)

    total_nulls = int(nulls_df["nulos"].sum())
    cols_with_nulls = int((nulls_df["nulos"] > 0).sum())

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total de nulos", f"{total_nulls:,}")
    with col2:
        st.metric("Columnas con nulos", cols_with_nulls)

    if cols_with_nulls > 0:
        st.plotly_chart(fig_nulls, use_container_width=True)
        st.dataframe(nulls_df[nulls_df["nulos"] > 0], use_container_width=True)
    else:
        st.success("No hay valores nulos en el dataset seleccionado.")

    st.markdown("#### Resumen por grupo de variables")
    group_rows = []
    for group_name, group_cols in {
        "Bounding box": [c for c in df.columns if c.startswith("bbox_")],
        "Coordenadas X/Y keypoints": [
            c for c in df.columns
            if (c.endswith("_x") or c.endswith("_y")) and not c.startswith("bbox_")
        ],
        "Confianza keypoints": [c for c in df.columns if c.endswith("_conf")],
    }.items():
        if group_cols:
            max_nulls = int(df[group_cols].isna().sum().max())
            group_rows.append({
                "Grupo": group_name,
                "Cantidad de columnas": len(group_cols),
                "Máximo de nulos por columna": max_nulls,
            })

    st.dataframe(pd.DataFrame(group_rows), use_container_width=True)

    st.info(
        "En PoseLift, los nulos son importantes porque pueden reflejar oclusiones, "
        "baja visibilidad o detecciones parciales del estimador de pose."
    )


# ============================================================
# TAB 4
# ============================================================

with tab4:
    st.markdown("### Distribución y normalidad")

    st.info(
        "Se evaluó normalidad para decidir si correspondía usar métodos paramétricos "
        "o no paramétricos. Si las variables no son normales, Spearman y Mann-Whitney "
        "son más apropiados que Pearson o pruebas paramétricas."
    )

    st.markdown("#### Histogramas interactivos")

    group_options = ["Todas", "Bounding box", "Coordenadas X", "Coordenadas Y", "Confianza"]
    selected_group = st.selectbox("Grupo de variables:", group_options)

    if selected_group == "Todas":
        available_cols = numeric_pose_cols
    else:
        available_cols = [
            col for col in numeric_pose_cols
            if classify_variable_group(col) == selected_group
        ]

    default_cols = available_cols[:6]
    selected_cols = st.multiselect(
        "Variables a graficar:",
        available_cols,
        default=default_cols,
        max_selections=12,
    )

    if selected_cols:
        fig_hist = plot_histograms(df, selected_cols)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("Selecciona al menos una variable para graficar.")

    st.markdown("#### Q-Q plot interactivo")
    qq_col = st.selectbox("Variable para Q-Q plot:", numeric_pose_cols)

    try:
        fig_qq = plot_qq(df, qq_col)
        st.plotly_chart(fig_qq, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudo generar Q-Q plot para {qq_col}: {e}")

    st.markdown("#### Resultados de normalidad")

    normalidad_path = PREPROCESSING_AED / "normalidad_todas_variables.csv"
    norm_df = load_csv_if_exists(normalidad_path)

    if norm_df is None:
        st.warning(f"No se encontró el archivo de normalidad: {normalidad_path}")
    else:
        fig_norm = plot_normality_summary(norm_df)

        if fig_norm is not None:
            st.plotly_chart(fig_norm, use_container_width=True)

        if "conclusion_normalidad" in norm_df.columns:
            total_vars = len(norm_df)
            no_normales = int((norm_df["conclusion_normalidad"] == "no normal").sum())
            normales = total_vars - no_normales

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Variables evaluadas", total_vars)
            with col2:
                st.metric("No normales", no_normales)
            with col3:
                st.metric("Normales", normales)

        cols_to_show = [
            c for c in [
                "variable",
                "n",
                "n_muestra",
                "n_nulos",
                "shapiro_p",
                "dagostino_p",
                "conclusion_normalidad",
            ]
            if c in norm_df.columns
        ]
        st.dataframe(norm_df[cols_to_show], use_container_width=True)


# ============================================================
# TAB 5
# ============================================================

with tab5:
    st.markdown("### Correlaciones")

    st.info(
        "Se utiliza Spearman porque las variables de pose no presentan normalidad "
        "y las relaciones entre keypoints no necesariamente son lineales."
    )

    st.markdown("#### Heatmap Spearman dinámico")

    max_cols = st.slider(
        "Cantidad máxima de variables para el heatmap:",
        min_value=10,
        max_value=min(55, len(numeric_pose_cols)),
        value=min(30, len(numeric_pose_cols)),
        step=5,
    )

    heatmap_cols = st.multiselect(
        "Variables incluidas en el heatmap:",
        numeric_pose_cols + ([LABEL_COL] if LABEL_COL in df.columns else []),
        default=(numeric_pose_cols[:max_cols] + ([LABEL_COL] if LABEL_COL in df.columns else [])),
    )

    if len(heatmap_cols) >= 2:
        fig_corr = plot_spearman_heatmap(df, heatmap_cols)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Selecciona al menos dos variables para calcular correlación.")

    st.markdown("#### Pearson vs Spearman respecto a label")

    pearson_spearman_path = PREPROCESSING_AED / "comparacion_pearson_spearman_label.csv"
    comparison_df = load_csv_if_exists(pearson_spearman_path)

    if comparison_df is None:
        st.warning(f"No se encontró: {pearson_spearman_path}")
    else:
        fig_comp = plot_pearson_spearman(comparison_df, top_n=20)
        st.plotly_chart(fig_comp, use_container_width=True)
        st.dataframe(comparison_df, use_container_width=True)

    st.info(
        "Las correlaciones individuales con label suelen ser bajas. Esto no invalida "
        "el análisis; indica que shoplifting no se explica por una sola coordenada, "
        "sino por patrones corporales y temporales multivariados."
    )


# ============================================================
# TAB 6
# ============================================================

with tab6:
    st.markdown("### Información mutua con label")

    st.info(
        "La información mutua mide dependencia entre cada variable y la etiqueta. "
        "A diferencia de una correlación lineal, puede capturar relaciones no lineales."
    )

    mi_path = PREPROCESSING_AED / "informacion_mutua_label.csv"
    mi_df = load_csv_if_exists(mi_path)

    if mi_df is None:
        st.warning(f"No se encontró: {mi_path}")
    else:
        top_n = st.slider("Top variables a mostrar:", 5, 30, 20, 5)
        fig_mi = plot_mutual_info(mi_df, top_n=top_n)
        st.plotly_chart(fig_mi, use_container_width=True)
        st.dataframe(mi_df, use_container_width=True)

        st.info(
            "Si las variables con mayor información mutua pertenecen a distintas "
            "zonas corporales, esto apoya la idea de que la anomalía no depende "
            "de un solo keypoint, sino de una configuración corporal más amplia."
        )


# ============================================================
# Cierre
# ============================================================

st.markdown("---")
st.markdown("### Conclusión del AED")

st.success(
    """
    El análisis descriptivo muestra que las variables numéricas de pose no siguen
    distribuciones normales y que las asociaciones individuales con label son bajas.
    Por ello, el análisis posterior se apoya en métodos no paramétricos y en hipótesis
    basadas en grupos corporales, relaciones espaciales y evolución temporal.
    """
)