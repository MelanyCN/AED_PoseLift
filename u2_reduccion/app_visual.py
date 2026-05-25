from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
EMBEDDING_PATH = PROCESSED_DIR / "embedding_umap.csv"
POSES_PATH = PROCESSED_DIR / "test_limpio_visual.csv"

COCO17_KEYPOINTS = [
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

COCO17_EDGES = [
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

LABEL_COLORS = {
    "0": "#2563eb",
    "1": "#dc2626",
}

LABEL_NAMES = {
    0: "Normal",
    1: "Shoplifting",
    "0": "Normal",
    "1": "Shoplifting",
}



st.set_page_config(
    page_title="Interfaz de analisis visual de trayectorias PoseLift",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp, [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #ffffff;
        color: #111827;
    }
    h1, h2, h3, h4, p, label, span, div {
        color: #111827;
    }
    .block-container {
        padding-top: 2.0rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1500px;
    }
    .selection-help {
        font-size: 0.88rem;
        color: #4b5563;
        margin-top: -0.2rem;
        margin-bottom: 0.5rem;
    }
    .card-normal {
        border-left: 6px solid #2563eb;
        border-top: 1px solid #bfdbfe;
        border-right: 1px solid #bfdbfe;
        border-bottom: 1px solid #bfdbfe;
        border-radius: 18px;
        padding: 1rem 1rem 0.4rem 1rem;
        background-color: #ffffff;
        margin-bottom: 1.2rem;
    }
    .card-shoplifting {
        border-left: 6px solid #dc2626;
        border-top: 1px solid #fecaca;
        border-right: 1px solid #fecaca;
        border-bottom: 1px solid #fecaca;
        border-radius: 18px;
        padding: 1rem 1rem 0.4rem 1rem;
        background-color: #ffffff;
        margin-bottom: 1.2rem;
    }
    .card-title-normal {
        color: #1d4ed8;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.35rem;
    }
    .card-title-shoplifting {
        color: #b91c1c;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.35rem;
    }
    .small-muted {
        color: #6b7280;
        font-size: 0.82rem;
    }
    .stButton button {
        border-radius: 10px;
        border: 1px solid #d1d5db;
        background: #ffffff;
        color: #111827;
    }
    .stButton button:hover {
        border-color: #111827;
        color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_embedding(path: Path = EMBEDDING_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {path}")
    df = pd.read_csv(path)
    required = {
        "video_id",
        "person_id",
        "umap_x",
        "umap_y",
        "label_trayectoria",
        "porcentaje_shoplifting",
        "num_frames",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en embedding_umap.csv: {sorted(missing)}")
    return df


@st.cache_data(show_spinner=False)
def load_poses(path: Path = POSES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {path}")
    df = pd.read_csv(path)
    required = {"video_id", "person_id", "frame_id", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en test_limpio_visual.csv: {sorted(missing)}")
    return df


def get_trajectory(poses: pd.DataFrame, video_id: Any, person_id: Any) -> pd.DataFrame:
    trajectory = poses[
        (poses["video_id"].astype(str) == str(video_id))
        & (poses["person_id"].astype(str) == str(person_id))
    ].copy()
    return trajectory.sort_values("frame_id").reset_index(drop=True)


def trajectory_summary(embedding: pd.DataFrame, video_id: Any, person_id: Any) -> pd.Series | None:
    subset = embedding[
        (embedding["video_id"].astype(str) == str(video_id))
        & (embedding["person_id"].astype(str) == str(person_id))
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def get_label_name(value: Any) -> str:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return str(value)
    return LABEL_NAMES.get(numeric, str(numeric))


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def selection_mode_control() -> str:
    if "selection_mode" not in st.session_state:
        st.session_state["selection_mode"] = "point"

    labels = {
        "point": "Punto",
        "box": "Rectangulo",
        "lasso": "Lazo",
    }

    st.markdown("**Seleccion:**")
    mode = st.radio(
        "Modo de seleccion",
        options=["point", "box", "lasso"],
        format_func=lambda item: labels[item],
        horizontal=True,
        label_visibility="collapsed",
        key="selection_mode",
    )

    active_label = labels[mode]
    st.markdown(
        f"<div class='selection-help'>Modo activo: <b>{active_label}</b>.</div>",
        unsafe_allow_html=True,
    )
    return mode


def extract_selected_points(selection_event: Any) -> list[dict[str, Any]]:
    if selection_event is None:
        return []

    try:
        raw_points = selection_event.selection.points
    except AttributeError:
        try:
            raw_points = selection_event["selection"]["points"]
        except (TypeError, KeyError):
            raw_points = []

    points: list[dict[str, Any]] = []
    for point in raw_points or []:
        if hasattr(point, "to_dict"):
            point = point.to_dict()
        if isinstance(point, dict):
            points.append(point)
    return points


def selected_trajectories_from_points(points: list[dict[str, Any]]) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for point in points:
        customdata = point.get("customdata") or point.get("custom_data")
        if not customdata or len(customdata) < 2:
            continue
        video_id = str(customdata[0])
        person_id = str(customdata[1])
        key = (video_id, person_id)
        if key not in seen:
            selected.append(key)
            seen.add(key)
    return selected


def build_umap_figure(scatter_df: pd.DataFrame, mode: str) -> go.Figure:
    scatter_df = scatter_df.copy()
    scatter_df["label_trayectoria_str"] = scatter_df["label_trayectoria"].astype(str)
    scatter_df["label_nombre"] = scatter_df["label_trayectoria"].map(get_label_name)

    dragmode = {
        "point": "pan",
        "box": "select",
        "lasso": "lasso",
    }.get(mode, "pan")

    fig = px.scatter(
        scatter_df,
        x="umap_x",
        y="umap_y",
        color="label_trayectoria_str",
        color_discrete_map=LABEL_COLORS,
        custom_data=[
            "video_id",
            "person_id",
            "num_frames",
            "porcentaje_shoplifting",
            "label_trayectoria",
            "label_nombre",
        ],
        hover_data={
            "video_id": True,
            "person_id": True,
            "num_frames": True,
            "porcentaje_shoplifting": ":.3f",
            "label_trayectoria": True,
            "label_trayectoria_str": False,
            "umap_x": ":.3f",
            "umap_y": ":.3f",
        },
        labels={
            "umap_x": "UMAP X",
            "umap_y": "UMAP Y",
            "label_trayectoria_str": "Label trayectoria",
        },
    )

    fig.update_traces(
        marker=dict(size=8, opacity=0.9, line=dict(width=0.5, color="#ffffff")),
        selected=dict(marker=dict(size=15, opacity=1.0)),
        unselected=dict(marker=dict(opacity=0.35)),
        hovertemplate=(
            "video_id: %{customdata[0]}<br>"
            "person_id: %{customdata[1]}<br>"
            "frames: %{customdata[2]}<br>"
            "porcentaje shoplifting: %{customdata[3]:.3f}<br>"
            "label trayectoria: %{customdata[5]}<extra></extra>"
        ),
    )

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#111827"),
        legend_title_text="Label trayectoria",
        height=560,
        dragmode=dragmode,
        clickmode="event+select",
        margin=dict(l=20, r=20, t=20, b=20),
        modebar=dict(bgcolor="#ffffff", color="#9ca3af", activecolor="#111827"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    return fig


def get_keypoint_arrays(row: pd.Series) -> tuple[dict[str, tuple[float, float, float, bool]], list[float], list[float]]:
    points: dict[str, tuple[float, float, float, bool]] = {}
    all_x: list[float] = []
    all_y: list[float] = []

    for keypoint in COCO17_KEYPOINTS:
        x_vis = safe_float(row.get(f"{keypoint}_x_vis"))
        y_vis = safe_float(row.get(f"{keypoint}_y_vis"))
        conf = safe_float(row.get(f"{keypoint}_conf"))
        valid = bool(row.get(f"{keypoint}_valid"))

        if np.isnan(x_vis) or np.isnan(y_vis):
            continue

        points[keypoint] = (x_vis, y_vis, conf, valid)
        all_x.append(x_vis)
        all_y.append(y_vis)

    return points, all_x, all_y


def build_bbox_from_row_keypoints(row: pd.Series, padding: float = 12.0):
    """
    Construye un bounding box visual desde los keypoints validos ya transformados
    a coordenadas *_x_vis y *_y_vis. Replica la logica del visor anterior:
    el rectangulo rodea la pose visible, no depende del formato original del bbox.
    """

    bbox_points: list[tuple[float, float]] = []

    for keypoint in COCO17_KEYPOINTS:
        x_vis = safe_float(row.get(f"{keypoint}_x_vis"))
        y_vis = safe_float(row.get(f"{keypoint}_y_vis"))
        valid = bool(row.get(f"{keypoint}_valid", True))

        if np.isnan(x_vis) or np.isnan(y_vis):
            continue

        if not valid:
            continue

        bbox_points.append((float(x_vis), float(y_vis)))

    if not bbox_points:
        return None

    xs = [point[0] for point in bbox_points]
    ys = [point[1] for point in bbox_points]

    x_min = min(xs) - padding
    x_max = max(xs) + padding
    y_min = min(ys) - padding
    y_max = max(ys) + padding

    return x_min, y_min, x_max, y_max


def frame_color(row: pd.Series) -> str:
    """Color principal del esqueleto segun la etiqueta del frame."""
    try:
        label = int(row.get("label"))
    except (TypeError, ValueError):
        label = 0
    return "#dc2626" if label == 1 else "#2563eb"


def build_skeleton_traces(row: pd.Series, show_keypoint_names: bool) -> list[go.Scatter]:
    points, _, _ = get_keypoint_arrays(row)
    main_color = frame_color(row)

    solid_x: list[float | None] = []
    solid_y: list[float | None] = []
    weak_x: list[float | None] = []
    weak_y: list[float | None] = []

    high_x: list[float] = []
    high_y: list[float] = []
    high_names: list[str] = []
    high_hover: list[str] = []

    low_x: list[float] = []
    low_y: list[float] = []
    low_names: list[str] = []
    low_hover: list[str] = []

    for start, end in COCO17_EDGES:
        if start not in points or end not in points:
            continue

        x1, y1, _, valid_1 = points[start]
        x2, y2, _, valid_2 = points[end]

        target_x, target_y = (solid_x, solid_y) if valid_1 and valid_2 else (weak_x, weak_y)
        target_x.extend([x1, x2, None])
        target_y.extend([y1, y2, None])

    for keypoint in COCO17_KEYPOINTS:
        if keypoint not in points:
            continue

        x_vis, y_vis, conf, valid = points[keypoint]
        original_x = safe_float(row.get(f"{keypoint}_x"))
        original_y = safe_float(row.get(f"{keypoint}_y"))
        hover = (
            f"Keypoint: {keypoint}<br>"
            f"x: {original_x:.3f}<br>"
            f"y: {original_y:.3f}<br>"
            f"confidence: {conf:.3f}<br>"
            f"valido: {int(valid)}"
        )

        if valid:
            high_x.append(x_vis)
            high_y.append(y_vis)
            high_names.append(keypoint)
            high_hover.append(hover)
        else:
            low_x.append(x_vis)
            low_y.append(y_vis)
            low_names.append(keypoint)
            low_hover.append(hover)

    text_mode = "+text" if show_keypoint_names else ""

    traces: list[go.Scatter] = [
        go.Scatter(
            x=weak_x,
            y=weak_y,
            mode="lines",
            line=dict(color="#9ca3af", width=2, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
            name="Conexiones baja confianza",
        ),
        go.Scatter(
            x=solid_x,
            y=solid_y,
            mode="lines",
            line=dict(color=main_color, width=3),
            hoverinfo="skip",
            showlegend=False,
            name="Conexiones validas",
        ),
        go.Scatter(
            x=low_x,
            y=low_y,
            mode=f"markers{text_mode}",
            text=low_names if show_keypoint_names else None,
            textposition="top center",
            textfont=dict(size=9, color="#6b7280"),
            marker=dict(size=8, color="#d1d5db", line=dict(color="#6b7280", width=1)),
            hovertext=low_hover,
            hoverinfo="text",
            showlegend=False,
            name="Baja confianza",
        ),
        go.Scatter(
            x=high_x,
            y=high_y,
            mode=f"markers{text_mode}",
            text=high_names if show_keypoint_names else None,
            textposition="top center",
            textfont=dict(size=9, color="#111827"),
            marker=dict(size=9, color=main_color, line=dict(color="#111827", width=1)),
            hovertext=high_hover,
            hoverinfo="text",
            showlegend=False,
            name="Valido",
        ),
    ]

    bbox = build_bbox_from_row_keypoints(row, padding=12.0)
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        traces.append(
            go.Scatter(
                x=[x_min, x_max, x_max, x_min, x_min],
                y=[y_min, y_min, y_max, y_max, y_min],
                mode="lines",
                line=dict(color=main_color, width=2, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
                name="Bounding box",
            )
        )
    else:
        traces.append(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                showlegend=False,
                hoverinfo="skip",
                name="Bounding box",
            )
        )

    return traces

def skeleton_axis_ranges(trajectory: pd.DataFrame) -> tuple[list[float], list[float]]:
    all_x: list[float] = []
    all_y: list[float] = []

    for _, row in trajectory.iterrows():
        _, x_values, y_values = get_keypoint_arrays(row)
        all_x.extend(x_values)
        all_y.extend(y_values)

    if not all_x or not all_y:
        return [-1, 1], [-1, 1]

    x_array = np.asarray(all_x, dtype=float)
    y_array = np.asarray(all_y, dtype=float)
    x_min, x_max = float(np.nanmin(x_array)), float(np.nanmax(x_array))
    y_min, y_max = float(np.nanmin(y_array)), float(np.nanmax(y_array))
    span = max(x_max - x_min, y_max - y_min, 1.0)
    margin = span * 0.20
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    x_range = [x_mid - span / 2 - margin, x_mid + span / 2 + margin]
    y_range = [y_mid - span / 2 - margin, y_mid + span / 2 + margin]
    return x_range, y_range


def frame_title(row: pd.Series, video_id: Any, person_id: Any) -> str:
    frame_id = int(row["frame_id"])
    label_name = get_label_name(row.get("label"))
    return f"Video {video_id} | Persona {person_id} | Frame {frame_id} | Label: {label_name}"


def create_animated_skeleton_figure(
    trajectory: pd.DataFrame,
    video_id: Any,
    person_id: Any,
    show_keypoint_names: bool,
    active_index: int,
    animation_duration: int = 250,
    height: int = 390,
) -> go.Figure:
    trajectory = trajectory.sort_values("frame_id").reset_index(drop=True)
    active_index = min(max(active_index, 0), len(trajectory) - 1)
    x_range, y_range = skeleton_axis_ranges(trajectory)

    first_row = trajectory.iloc[active_index]
    frames = []
    for _, row in trajectory.iterrows():
        frame_id = str(int(row["frame_id"]))
        frames.append(
            go.Frame(
                data=build_skeleton_traces(row, show_keypoint_names),
                name=frame_id,
                layout=go.Layout(title_text=frame_title(row, video_id, person_id)),
            )
        )

    fig = go.Figure(
        data=build_skeleton_traces(first_row, show_keypoint_names),
        frames=frames,
    )

    fig.update_layout(
        title=dict(text=frame_title(first_row, video_id, person_id), font=dict(size=13)),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#111827", size=11),
        height=height,
        margin=dict(l=10, r=10, t=50, b=15),
        showlegend=False,
        xaxis=dict(
            range=x_range,
            showgrid=True,
            gridcolor="#e5e7eb",
            zeroline=False,
            scaleanchor="y",
            scaleratio=1,
            visible=True,
        ),
        yaxis=dict(
            range=y_range,
            showgrid=True,
            gridcolor="#e5e7eb",
            zeroline=False,
            visible=True,
        ),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": 1.18,
                "xanchor": "left",
                "yanchor": "top",
                "direction": "right",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": animation_duration, "redraw": True},
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
                                "frame": {"duration": 0, "redraw": False},
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
                "active": active_index,
                "currentvalue": {"prefix": "Frame: ", "font": {"size": 12}},
                "pad": {"t": 45},
                "len": 0.92,
                "x": 0.04,
                "xanchor": "left",
                "steps": [
                    {
                        "method": "animate",
                        "label": str(int(row["frame_id"])),
                        "args": [
                            [str(int(row["frame_id"]))],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for _, row in trajectory.iterrows()
                ],
            }
        ],
    )
    return fig


def sequence_quality_summary(trajectory: pd.DataFrame) -> str:
    """Resumen de calidad de pose para toda la secuencia."""
    parts: list[str] = []

    if "mean_keypoint_conf" in trajectory.columns:
        conf_values = pd.to_numeric(trajectory["mean_keypoint_conf"], errors="coerce").dropna()
        if not conf_values.empty:
            parts.append(
                f"confianza media {conf_values.mean():.3f} ± {conf_values.std(ddof=0):.3f}"
            )

    if "porcentaje_keypoints_validos" in trajectory.columns:
        valid_values = pd.to_numeric(
            trajectory["porcentaje_keypoints_validos"], errors="coerce"
        ).dropna()
        if not valid_values.empty:
            parts.append(
                f"keypoints validos {valid_values.mean():.3f} ± {valid_values.std(ddof=0):.3f}"
            )

    if "pose_quality" in trajectory.columns:
        counts = trajectory["pose_quality"].dropna().astype(str).value_counts()
        if not counts.empty:
            distribution = ", ".join(f"{name}: {count}" for name, count in counts.items())
            parts.append(f"distribucion {distribution}")

    return "; ".join(parts) if parts else "No disponible"


def render_trajectory_card(
    embedding: pd.DataFrame,
    poses: pd.DataFrame,
    video_id: Any,
    person_id: Any,
    card_index: int,
) -> None:
    summary = trajectory_summary(embedding, video_id, person_id)
    trajectory = get_trajectory(poses, video_id, person_id)

    if summary is None or trajectory.empty:
        st.warning(f"No se encontraron datos para video_id={video_id}, person_id={person_id}.")
        return

    trajectory = trajectory.sort_values("frame_id").reset_index(drop=True)
    row = trajectory.iloc[0]
    frame_index = 0

    label_value = int(summary["label_trayectoria"])
    label_name = get_label_name(label_value)
    title_color = "#b91c1c" if label_value == 1 else "#1d4ed8"
    safe_video = str(video_id).replace(" ", "_").replace("/", "_")
    safe_person = str(person_id).replace(" ", "_").replace("/", "_")
    key_prefix = f"card_{card_index}_{safe_video}_{safe_person}"

    with st.container(border=True):
        st.markdown(
            f'<div style="color:{title_color}; font-weight:700; font-size:1.05rem; margin-bottom:0.25rem;">'
            f'Trayectoria seleccionada {card_index + 1}</div>',
            unsafe_allow_html=True,
        )

        show_names = st.checkbox(
            "Mostrar nombres de keypoints",
            value=False,
            key=f"{key_prefix}_show_names",
        )

        info_col, plot_col = st.columns([0.34, 0.66], vertical_alignment="top")

        with info_col:
            st.write(f"**video_id:** {video_id}")
            st.write(f"**person_id:** {person_id}")
            st.write(f"**Label trayectoria:** {label_name}")
            st.write(f"**Porcentaje shoplifting:** {float(summary['porcentaje_shoplifting']):.3f}")
            st.write(f"**Frames de la trayectoria:** {len(trajectory)}")
            st.write(f"**Frame inicial:** {int(row['frame_id'])}")
            st.write(f"**Etiqueta del frame inicial:** {get_label_name(row.get('label'))}")
            #st.write(f"**Calidad de pose de la secuencia:** {sequence_quality_summary(trajectory)}")

        with plot_col:
            fig = create_animated_skeleton_figure(
                trajectory=trajectory,
                video_id=video_id,
                person_id=person_id,
                show_keypoint_names=show_names,
                active_index=frame_index,
                animation_duration=250,
                height=390,
            )
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

def main() -> None:
    st.title("Interfaz de analisis visual de trayectorias PoseLift")

    try:
        embedding = load_embedding()
        poses = load_poses()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    mode = selection_mode_control()

    left_col, right_col = st.columns([0.52, 0.48], gap="large")

    with left_col:
        fig_umap = build_umap_figure(embedding, mode)
        selection_event = st.plotly_chart(
            fig_umap,
            width="stretch",
            key="umap_selection",
            on_select="rerun",
            selection_mode=("points", "box", "lasso"),
            config={
                "displaylogo": False,
                "modeBarButtonsToAdd": ["select2d", "lasso2d"],
                "modeBarButtonsToRemove": ["autoScale2d", "toggleSpikelines"],
            },
        )

    selected_points = extract_selected_points(selection_event)
    selected_trajectories = selected_trajectories_from_points(selected_points)

    with right_col:
        if not selected_trajectories:
            st.markdown(
                '<div class="small-muted">Seleccione uno o varios puntos del mapa para visualizar sus secuencias.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.subheader("Secuencias seleccionadas")
            st.caption(f"Trayectorias seleccionadas: {len(selected_trajectories)}")
            for index, (video_id, person_id) in enumerate(selected_trajectories):
                render_trajectory_card(embedding, poses, video_id, person_id, index)

    # Mejoras futuras:
    # - agregar graficos temporales de velocidad de munecas y codos
    # - agregar distancias codo-cadera y muneca-cadera
    # - agregar comparacion cuantitativa entre dos trayectorias
    # - agregar t-SNE como alternativa a UMAP
    # - agregar busqueda automatica de vecinos cercanos


if __name__ == "__main__":
    main()