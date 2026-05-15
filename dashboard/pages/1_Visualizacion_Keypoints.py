from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. Configuración general
# ============================================================

st.set_page_config(page_title="Visualización Keypoints", layout="wide")


def find_project_root() -> Path:
    """
    Busca la raíz del proyecto de forma robusta, sin depender del working directory.
    """
    current = Path(__file__).resolve()

    for parent in [current.parent] + list(current.parents):
        if (
            (parent / "Json_files").exists()
            and (parent / "pre_procesamiento").exists()
        ):
            return parent

    # Fallback si el archivo está en dashboard/pages
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = find_project_root()

JSON_DIRS = {
    "train": PROJECT_ROOT / "Json_files" / "data" / "PoseLift" / "pose" / "train",
    "test": PROJECT_ROOT / "Json_files" / "data" / "PoseLift" / "pose" / "test",
}

TEST_TABULAR_PATH = PROJECT_ROOT / "pre_procesamiento" / "outputs" / "test_tabular.csv"

COCO17_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO17_SKELETON = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

COCO17_KEYPOINT_COLORS = {
    "nose": "#1f77b4",
    "left_eye": "#ff7f0e",
    "right_eye": "#2ca02c",
    "left_ear": "#d62728",
    "right_ear": "#9467bd",
    "left_shoulder": "#8c564b",
    "right_shoulder": "#e377c2",
    "left_elbow": "#7f7f7f",
    "right_elbow": "#bcbd22",
    "left_wrist": "#17becf",
    "right_wrist": "#aec7e8",
    "left_hip": "#ffbb78",
    "right_hip": "#98df8a",
    "left_knee": "#ff9896",
    "right_knee": "#c5b0d5",
    "left_ankle": "#c49c94",
    "right_ankle": "#f7b6d2",
}


# ============================================================
# 2. Utilidades de carga
# ============================================================

def ordenar_id(valor):
    try:
        return 0, int(valor)
    except Exception:
        return 1, str(valor)


@st.cache_data
def obtener_json_files(split: str):
    json_dir = JSON_DIRS[split]

    if not json_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta JSON esperada: {json_dir}")

    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No se encontraron archivos JSON en: {json_dir}")

    return [str(path) for path in json_files]


@st.cache_data
def cargar_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def cargar_labels_test():
    """
    Carga etiquetas del test tabular.
    La etiqueta está a nivel de frame, no de persona.
    """
    if not TEST_TABULAR_PATH.exists():
        return None

    df = pd.read_csv(TEST_TABULAR_PATH)

    required = {"video_id", "frame_id", "label"}
    if not required.issubset(df.columns):
        return None

    df = df[["video_id", "frame_id", "label"]].drop_duplicates()
    df["video_id"] = df["video_id"].astype(str)
    df["frame_id"] = df["frame_id"].astype(int)
    df["label"] = df["label"].astype(int)

    return df


def obtener_video_id_desde_json(json_path: Path):
    """
    Convierte nombres de JSON como:
    01_0222_alphapose_tracked_person.json

    al formato usado en test_tabular.csv:
    1_222
    """
    stem = json_path.stem

    # Quitar sufijo fijo del archivo JSON
    stem = stem.replace("_alphapose_tracked_person", "")

    parts = stem.split("_")

    if len(parts) >= 2:
        camara = str(int(parts[0]))   # "01" -> "1"
        video = str(int(parts[1]))    # "0222" -> "222"
        return f"{camara}_{video}"

    return stem

def obtener_personas(json_data):
    return sorted(json_data.keys(), key=ordenar_id)


def obtener_frame_ids(json_data, person_id):
    person_id = str(person_id)

    if person_id not in json_data:
        return []

    frame_ids = sorted(json_data[person_id].keys(), key=ordenar_id)
    return frame_ids


def extraer_keypoints(json_data, person_id, frame_id):
    frame_content = json_data[str(person_id)][str(frame_id)]
    return np.asarray(frame_content["keypoints"], dtype=float).reshape(17, 3)


def obtener_label_frame(split, video_id, frame_id, labels_df):
    """
    Para train no hay label.
    Para test, busca label por video_id y frame_id.
    """
    if split != "test" or labels_df is None:
        return None

    try:
        frame_id = int(frame_id)
    except Exception:
        return None

    subset = labels_df[
        (labels_df["video_id"].astype(str) == str(video_id))
        & (labels_df["frame_id"].astype(int) == frame_id)
    ]

    if subset.empty:
        return None

    return int(subset["label"].iloc[0])
# ============================================================
# 3. Transformación visual
# ============================================================

def punto_valido(punto, conf_threshold):
    x, y, conf = punto
    return (
        np.isfinite(x)
        and np.isfinite(y)
        and np.isfinite(conf)
        and conf >= conf_threshold
    )


def transformar_punto(x, y, modo="rot90_cw"):
    """
    Transformación solo para visualización.
    No modifica los datos originales.
    """
    if modo == "original":
        return x, y

    if modo == "rot90_cw":
        return y, -x

    if modo == "rot90_ccw":
        return -y, x

    if modo == "swap":
        return y, x

    raise ValueError(f"Modo de visualización no reconocido: {modo}")


def calcular_rango_ejes(json_data, person_id, frame_ids, conf_threshold, modo_visualizacion):
    puntos_visibles = []
    puntos_finitos = []

    for frame_id in frame_ids:
        keypoints = extraer_keypoints(json_data, person_id, frame_id)

        for punto in keypoints:
            x, y, conf = punto

            if np.isfinite(x) and np.isfinite(y):
                x_plot, y_plot = transformar_punto(x, y, modo_visualizacion)
                puntos_finitos.append([x_plot, y_plot])

            if punto_valido(punto, conf_threshold):
                x_plot, y_plot = transformar_punto(x, y, modo_visualizacion)
                puntos_visibles.append([x_plot, y_plot])

    puntos = np.asarray(puntos_visibles or puntos_finitos, dtype=float)

    if puntos.size == 0:
        raise ValueError(f"No hay keypoints válidos para person_id={person_id}")

    x_min, y_min = puntos.min(axis=0)
    x_max, y_max = puntos.max(axis=0)

    ancho = max(x_max - x_min, 1.0)
    alto = max(y_max - y_min, 1.0)
    margen = max(ancho, alto) * 0.08 + 20

    return [x_min - margen, x_max + margen], [y_min - margen, y_max + margen]


def calcular_bbox_desde_keypoints(keypoints, conf_threshold, modo_visualizacion, padding=12):
    puntos_bbox = []

    for punto in keypoints:
        if not punto_valido(punto, conf_threshold):
            continue

        x_plot, y_plot = transformar_punto(
            punto[0],
            punto[1],
            modo_visualizacion,
        )

        puntos_bbox.append([x_plot, y_plot])

    if not puntos_bbox:
        return None

    puntos_bbox = np.asarray(puntos_bbox, dtype=float)

    x_min, y_min = puntos_bbox.min(axis=0)
    x_max, y_max = puntos_bbox.max(axis=0)

    return (
        x_min - padding,
        y_min - padding,
        x_max + padding,
        y_max + padding,
    )


# ============================================================
# 4. Creación de trazas
# ============================================================

def obtener_colores_por_label(label):
    """
    Normal = blanco.
    Shoplifting = rojo.
    Train o label desconocido = gris.
    """
    if label == 0:
        return {
            "skeleton": "white",
            "bbox": "white",
            "label_text": "Normal",
            "bg": "#111827",
        }

    if label == 1:
        return {
            "skeleton": "#ef4444",
            "bbox": "#ef4444",
            "label_text": "Shoplifting",
            "bg": "#111827",
        }

    return {
        "skeleton": "#d1d5db",
        "bbox": "#d1d5db",
        "label_text": "Sin label",
        "bg": "#111827",
    }


def crear_trace_bbox(keypoints, conf_threshold, modo_visualizacion, label_color, dibujar_bbox=True, padding=12):
    if not dibujar_bbox:
        return go.Scatter(
            x=[],
            y=[],
            mode="lines",
            hoverinfo="skip",
            showlegend=False,
        )

    bbox = calcular_bbox_desde_keypoints(
        keypoints,
        conf_threshold,
        modo_visualizacion,
        padding=padding,
    )

    if bbox is None:
        return go.Scatter(
            x=[],
            y=[],
            mode="lines",
            hoverinfo="skip",
            showlegend=False,
        )

    x_min, y_min, x_max, y_max = bbox

    return go.Scatter(
        x=[x_min, x_max, x_max, x_min, x_min],
        y=[y_min, y_min, y_max, y_max, y_min],
        mode="lines",
        line=dict(color=label_color, width=2, dash="dash"),
        hoverinfo="skip",
        name="bbox",
        showlegend=False,
    )


def crear_traces_pose(
    keypoints,
    label,
    conf_threshold=0.0,
    modo_visualizacion="rot90_cw",
    dibujar_bbox=True,
    bbox_padding=12,
    show_skeleton=True,
    show_labels=False,
    point_size=10,
):
    colores_label = obtener_colores_por_label(label)

    puntos_x = []
    puntos_y = []
    textos = []
    hovers = []
    colores = []

    for i, kp_name in enumerate(COCO17_KEYPOINT_NAMES):
        punto = keypoints[i]

        if not punto_valido(punto, conf_threshold):
            continue

        x, y, conf = punto
        x_plot, y_plot = transformar_punto(x, y, modo_visualizacion)

        puntos_x.append(x_plot)
        puntos_y.append(y_plot)
        textos.append(kp_name if show_labels else "")
        hovers.append(f"{kp_name}<br>conf={conf:.3f}<br>label={colores_label['label_text']}")
        colores.append(COCO17_KEYPOINT_COLORS[kp_name])

    keypoints_por_nombre = dict(zip(COCO17_KEYPOINT_NAMES, keypoints))
    lineas_x = []
    lineas_y = []

    if show_skeleton:
        for kp1, kp2 in COCO17_SKELETON:
            punto_1 = keypoints_por_nombre[kp1]
            punto_2 = keypoints_por_nombre[kp2]

            if punto_valido(punto_1, conf_threshold) and punto_valido(punto_2, conf_threshold):
                x1_plot, y1_plot = transformar_punto(
                    punto_1[0],
                    punto_1[1],
                    modo_visualizacion,
                )
                x2_plot, y2_plot = transformar_punto(
                    punto_2[0],
                    punto_2[1],
                    modo_visualizacion,
                )

                lineas_x.extend([x1_plot, x2_plot, None])
                lineas_y.extend([y1_plot, y2_plot, None])

    bbox_trace = crear_trace_bbox(
        keypoints,
        conf_threshold,
        modo_visualizacion,
        colores_label["bbox"],
        dibujar_bbox=dibujar_bbox,
        padding=bbox_padding,
    )

    skeleton_trace = go.Scatter(
        x=lineas_x,
        y=lineas_y,
        mode="lines",
        line=dict(color=colores_label["skeleton"], width=4),
        hoverinfo="skip",
        name=f"Esqueleto: {colores_label['label_text']}",
        showlegend=True,
    )

    point_trace = go.Scatter(
        x=puntos_x,
        y=puntos_y,
        mode="markers+text" if show_labels else "markers",
        text=textos,
        textposition="top center",
        textfont=dict(size=10, color="white"),
        marker=dict(
            size=point_size,
            color=colores,
            line=dict(color="black", width=1),
            opacity=0.95,
        ),
        hovertext=hovers,
        hoverinfo="text",
        name="Keypoints",
        showlegend=False,
    )

    return [bbox_trace, skeleton_trace, point_trace]


# ============================================================
# 5. Figura animada Plotly
# ============================================================

def crear_figura_interactiva(
    json_data,
    json_path,
    split,
    video_id,
    person_id,
    frame_ids,
    labels_df,
    conf_threshold=0.0,
    modo_visualizacion="rot90_cw",
    dibujar_bbox=True,
    bbox_padding=12,
    show_skeleton=True,
    show_labels=False,
    point_size=10,
    animation_duration=250,
):
    first_frame_id = frame_ids[0]
    first_keypoints = extraer_keypoints(json_data, person_id, first_frame_id)
    first_label = obtener_label_frame(split, video_id, first_frame_id, labels_df)

    x_range, y_range = calcular_rango_ejes(
        json_data,
        person_id,
        frame_ids,
        conf_threshold,
        modo_visualizacion,
    )

    frames = []

    for frame_id in frame_ids:
        keypoints = extraer_keypoints(json_data, person_id, frame_id)
        label = obtener_label_frame(split, video_id, frame_id, labels_df)
        colores_label = obtener_colores_por_label(label)

        frames.append(
            go.Frame(
                data=crear_traces_pose(
                    keypoints=keypoints,
                    label=label,
                    conf_threshold=conf_threshold,
                    modo_visualizacion=modo_visualizacion,
                    dibujar_bbox=dibujar_bbox,
                    bbox_padding=bbox_padding,
                    show_skeleton=show_skeleton,
                    show_labels=show_labels,
                    point_size=point_size,
                ),
                name=str(frame_id),
                layout=go.Layout(
                    title_text=(
                        f"{json_path.name}<br>"
                        f"Persona: {person_id} | Frame: {frame_id} | "
                        f"Label: {colores_label['label_text']}"
                    )
                ),
            )
        )

    first_colors = obtener_colores_por_label(first_label)

    fig = go.Figure(
        data=crear_traces_pose(
            keypoints=first_keypoints,
            label=first_label,
            conf_threshold=conf_threshold,
            modo_visualizacion=modo_visualizacion,
            dibujar_bbox=dibujar_bbox,
            bbox_padding=bbox_padding,
            show_skeleton=show_skeleton,
            show_labels=show_labels,
            point_size=point_size,
        ),
        frames=frames,
    )

    fig.update_layout(
        title=(
            f"{json_path.name}<br>"
            f"Persona: {person_id} | Frame: {first_frame_id} | "
            f"Label: {first_colors['label_text']}"
        ),
        xaxis=dict(
            title="Coordenada X",
            range=x_range,
            zeroline=False,
            color="white",
            gridcolor="#374151",
        ),
        yaxis=dict(
            title="Coordenada Y",
            range=y_range,
            scaleanchor="x",
            scaleratio=1,
            zeroline=False,
            color="white",
            gridcolor="#374151",
        ),
        height=750,
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font=dict(color="white"),
        showlegend=True,
        margin=dict(l=60, r=30, t=90, b=60),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": 1.12,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": animation_duration,
                                    "redraw": True,
                                },
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Frame: "},
                "pad": {"t": 45},
                "steps": [
                    {
                        "method": "animate",
                        "label": str(frame_id),
                        "args": [
                            [str(frame_id)],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": True,
                                },
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for frame_id in frame_ids
                ],
            }
        ],
    )

    return fig


# ============================================================
# 6. Interfaz Streamlit
# ============================================================

st.markdown(
    """
    <h1 style='text-align: center;'>Visualización de Keypoints</h1>
    <p style='text-align: center; color: #7F8C8D;'>
    Reproducción interactiva de poses COCO17 con etiqueta visual por frame
    </p>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## Configuración")

split = st.sidebar.radio("Selecciona split:", ["test", "train"], index=0)

try:
    json_files = obtener_json_files(split)
except Exception as e:
    st.error(str(e))
    st.stop()

json_paths = [Path(path) for path in json_files]
json_names = [path.name for path in json_paths]

selected_json_name = st.sidebar.selectbox("Selecciona archivo JSON:", json_names)
json_path = json_paths[json_names.index(selected_json_name)]

json_data = cargar_json(str(json_path))
video_id = obtener_video_id_desde_json(json_path)

persons = obtener_personas(json_data)

if not persons:
    st.error("Este archivo JSON no contiene personas.")
    st.stop()

person_id = st.sidebar.selectbox("Selecciona persona:", persons)

frame_ids_all = obtener_frame_ids(json_data, person_id)

if not frame_ids_all:
    st.error(f"La persona {person_id} no tiene frames disponibles.")
    st.stop()

frame_idx_start, frame_idx_end = st.sidebar.slider(
    "Rango de frames:",
    min_value=0,
    max_value=len(frame_ids_all) - 1,
    value=(0, min(len(frame_ids_all) - 1, 120)),
)

frame_ids = frame_ids_all[frame_idx_start:frame_idx_end + 1]

st.sidebar.markdown("### Opciones visuales")

modo_visualizacion = st.sidebar.selectbox(
    "Modo de visualización:",
    ["rot90_cw", "original", "rot90_ccw", "swap"],
    index=0,
)

conf_threshold = st.sidebar.slider(
    "Confianza mínima:",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.05,
)

show_skeleton = st.sidebar.checkbox("Mostrar esqueleto", value=True)
show_labels = st.sidebar.checkbox("Mostrar nombres de keypoints", value=False)
dibujar_bbox = st.sidebar.checkbox("Mostrar bounding box aproximado", value=True)

point_size = st.sidebar.slider("Tamaño de puntos:", 5, 18, 10)

animation_duration = st.sidebar.slider(
    "Duración por frame en Play (ms):",
    min_value=50,
    max_value=1000,
    value=250,
    step=50,
)

labels_df = cargar_labels_test()

if split == "test" and labels_df is not None:
    labels_video = labels_df[labels_df["video_id"].astype(str) == str(video_id)]

    if labels_video.empty:
        st.warning(
            f"No se encontraron labels para video_id={video_id}. "
            "El esqueleto se mostrará en gris."
        )
    else:
        conteo_labels = labels_video["label"].value_counts().to_dict()
        st.success(
            f"Labels encontrados para video_id={video_id}: "
            f"{conteo_labels}"
        )

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Split", split)

with col2:
    st.metric("Video", video_id)

with col3:
    st.metric("Persona", person_id)

with col4:
    st.metric("Frames seleccionados", len(frame_ids))

if split == "test" and labels_df is None:
    st.warning(
        f"No se pudo cargar el archivo de etiquetas test: {TEST_TABULAR_PATH}. "
        "Se mostrará la pose sin color por label."
    )

if split == "train":
    st.info(
        "El split train no tiene etiquetas label en la estructura inspeccionada. "
        "Por ello, el esqueleto se muestra en gris."
    )

st.markdown(
    """
    **Color del esqueleto según etiqueta:**
    - Blanco: comportamiento normal.
    - Rojo: `shoplifting`.
    - Gris: sin etiqueta disponible, por ejemplo en `train`.
    """
)

try:
    fig = crear_figura_interactiva(
        json_data=json_data,
        json_path=json_path,
        split=split,
        video_id=video_id,
        person_id=person_id,
        frame_ids=frame_ids,
        labels_df=labels_df,
        conf_threshold=conf_threshold,
        modo_visualizacion=modo_visualizacion,
        dibujar_bbox=dibujar_bbox,
        bbox_padding=12,
        show_skeleton=show_skeleton,
        show_labels=show_labels,
        point_size=point_size,
        animation_duration=animation_duration,
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"No se pudo construir la visualización: {e}")
    st.stop()

st.markdown("### Información de depuración")

with st.expander("Ver rutas y datos cargados"):
    st.write("PROJECT_ROOT:", PROJECT_ROOT)
    st.write("JSON path:", json_path)
    st.write("TEST_TABULAR_PATH:", TEST_TABULAR_PATH)
    st.write("Cantidad de personas:", len(persons))
    st.write("Cantidad total de frames para persona:", len(frame_ids_all))
    st.write("Primeros frames:", frame_ids[:10])