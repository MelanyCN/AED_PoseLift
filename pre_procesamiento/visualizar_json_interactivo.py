from pathlib import Path
import json

import numpy as np
import plotly.graph_objects as go


BASE_DIR = Path(__file__).resolve().parents[1]
JSON_DIRS = {
    "train": BASE_DIR / "Json_files" / "data" / "PoseLift" / "pose" / "train",
    "test": BASE_DIR / "Json_files" / "data" / "PoseLift" / "pose" / "test",
}

# Cambia estos valores si quieres visualizar otra persona o archivo.
JSON_SPLIT = "test"
JSON_FILE_INDEX = 10
PERSON_ID = None
CONF_THRESHOLD = 0.3
MODO_VISUALIZACION = "rot90_cw"
DIBUJAR_BBOX = True
BBOX_PADDING = 12
OUTPUT_HTML = Path(__file__).resolve().with_name("visualizacion_json_interactiva.html")
SHOW_FIGURE = True


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


def ordenar_id(valor):
    try:
        return 0, int(valor)
    except ValueError:
        return 1, str(valor)


def obtener_json_files(split=JSON_SPLIT):
    json_dir = JSON_DIRS[split]
    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No se encontraron archivos JSON en: {json_dir}")

    return json_files


def cargar_json(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def seleccionar_persona(json_data, person_id=None):
    if person_id is None:
        person_id = sorted(json_data.keys(), key=ordenar_id)[0]

    person_id = str(person_id)

    if person_id not in json_data:
        raise KeyError(f"No existe person_id={person_id} en este JSON.")

    return person_id


def obtener_frame_ids(json_data, person_id):
    frame_ids = sorted(json_data[str(person_id)].keys(), key=ordenar_id)

    if not frame_ids:
        raise ValueError(f"La persona {person_id} no tiene frames disponibles.")

    return frame_ids


def extraer_keypoints(json_data, person_id, frame_id):
    frame_content = json_data[str(person_id)][str(frame_id)]
    return np.asarray(frame_content["keypoints"], dtype=float).reshape(17, 3)


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
    Transforma coordenadas solo para visualizacion.
    No modifica los datos originales.

    modos:
    - "original": usa x, y tal como vienen
    - "rot90_cw": rota 90 grados en sentido horario
    - "rot90_ccw": rota 90 grados en sentido antihorario
    - "swap": intercambia x e y
    """
    if modo == "original":
        return x, y

    if modo == "rot90_cw":
        return y, -x

    if modo == "rot90_ccw":
        return -y, x

    if modo == "swap":
        return y, x

    raise ValueError(f"Modo de transformacion no reconocido: {modo}")


def calcular_bbox_keypoints(
    keypoints,
    conf_threshold=0.0,
    modo_visualizacion="original",
    padding=12,
):
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


def crear_trace_bbox(
    keypoints,
    conf_threshold=0.0,
    modo_visualizacion="original",
    dibujar_bbox=True,
    bbox_padding=12,
):
    bbox = None

    if dibujar_bbox:
        bbox = calcular_bbox_keypoints(
            keypoints,
            conf_threshold,
            modo_visualizacion,
            bbox_padding,
        )

    if bbox is None:
        return go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="#ef4444", width=2, dash="dash"),
            hoverinfo="skip",
            name="bbox",
        )

    x_min, y_min, x_max, y_max = bbox

    return go.Scatter(
        x=[x_min, x_max, x_max, x_min, x_min],
        y=[y_min, y_min, y_max, y_max, y_min],
        mode="lines",
        line=dict(color="#ef4444", width=2, dash="dash"),
        hoverinfo="skip",
        name="bbox",
    )


def crear_traces_pose(
    keypoints,
    conf_threshold=0.0,
    modo_visualizacion="original",
    dibujar_bbox=True,
    bbox_padding=12,
):
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
        textos.append(kp_name)
        hovers.append(f"{kp_name}<br>conf={conf:.3f}")
        colores.append(COCO17_KEYPOINT_COLORS[kp_name])

    keypoints_por_nombre = dict(zip(COCO17_KEYPOINT_NAMES, keypoints))
    lineas_x = []
    lineas_y = []

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

    skeleton_trace = go.Scatter(
        x=lineas_x,
        y=lineas_y,
        mode="lines",
        line=dict(color="black", width=3),
        hoverinfo="skip",
        name="esqueleto",
    )

    point_trace = go.Scatter(
        x=puntos_x,
        y=puntos_y,
        mode="markers+text",
        text=textos,
        textposition="top center",
        textfont=dict(size=11, color="#111827"),
        marker=dict(
            size=10,
            color=colores,
            line=dict(color="black", width=1),
        ),
        hovertext=hovers,
        hoverinfo="text",
        name="keypoints",
    )

    bbox_trace = crear_trace_bbox(
        keypoints,
        conf_threshold,
        modo_visualizacion,
        dibujar_bbox,
        bbox_padding,
    )

    return [bbox_trace, skeleton_trace, point_trace]


def calcular_rango_ejes(
    json_data,
    person_id,
    frame_ids,
    conf_threshold,
    modo_visualizacion="original",
):
    puntos_visibles = []
    puntos_finitos = []

    for frame_id in frame_ids:
        keypoints = extraer_keypoints(json_data, person_id, frame_id)

        for punto in keypoints:
            x, y, _ = punto

            if np.isfinite(x) and np.isfinite(y):
                x_plot, y_plot = transformar_punto(x, y, modo_visualizacion)
                puntos_finitos.append([x_plot, y_plot])

            if punto_valido(punto, conf_threshold):
                x_plot, y_plot = transformar_punto(x, y, modo_visualizacion)
                puntos_visibles.append([x_plot, y_plot])

    puntos = np.asarray(puntos_visibles or puntos_finitos, dtype=float)

    if puntos.size == 0:
        raise ValueError(f"No hay keypoints validos para person_id={person_id}.")

    x_min, y_min = puntos.min(axis=0)
    x_max, y_max = puntos.max(axis=0)

    ancho = max(x_max - x_min, 1.0)
    alto = max(y_max - y_min, 1.0)
    margen = max(ancho, alto) * 0.08 + 20

    return [x_min - margen, x_max + margen], [y_min - margen, y_max + margen]


def construir_frames(
    json_data,
    json_path,
    person_id,
    frame_ids,
    conf_threshold,
    modo_visualizacion="original",
    dibujar_bbox=True,
    bbox_padding=12,
):
    frames = []

    for frame_id in frame_ids:
        keypoints = extraer_keypoints(json_data, person_id, frame_id)

        frames.append(
            go.Frame(
                data=crear_traces_pose(
                    keypoints,
                    conf_threshold,
                    modo_visualizacion,
                    dibujar_bbox,
                    bbox_padding,
                ),
                name=str(frame_id),
                layout=go.Layout(
                    title_text=(
                        f"{json_path.name}<br>Persona: {person_id} | "
                        f"Frame: {frame_id} | Vista: {modo_visualizacion}"
                    )
                ),
            )
        )

    return frames


def crear_figura_interactiva(
    json_data,
    json_path,
    person_id,
    conf_threshold=0.0,
    modo_visualizacion="original",
    dibujar_bbox=True,
    bbox_padding=12,
):
    frame_ids = obtener_frame_ids(json_data, person_id)
    first_frame_id = frame_ids[0]
    first_keypoints = extraer_keypoints(json_data, person_id, first_frame_id)
    x_range, y_range = calcular_rango_ejes(
        json_data,
        person_id,
        frame_ids,
        conf_threshold,
        modo_visualizacion,
    )

    fig = go.Figure(
        data=crear_traces_pose(
            first_keypoints,
            conf_threshold,
            modo_visualizacion,
            dibujar_bbox,
            bbox_padding,
        ),
        frames=construir_frames(
            json_data,
            json_path,
            person_id,
            frame_ids,
            conf_threshold,
            modo_visualizacion,
            dibujar_bbox,
            bbox_padding,
        ),
    )

    fig.update_layout(
        title=(
            f"{json_path.name}<br>Persona: {person_id} | "
            f"Frame: {first_frame_id} | Vista: {modo_visualizacion}"
        ),
        xaxis=dict(
            title="Coordenada X",
            range=x_range,
            zeroline=False,
        ),
        yaxis=dict(
            title="Coordenada Y",
            range=y_range,
            scaleanchor="x",
            scaleratio=1,
            zeroline=False,
        ),
        width=900,
        height=800,
        plot_bgcolor="white",
        showlegend=True,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": 1.08,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 250, "redraw": True},
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
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for frame_id in frame_ids
                ],
            }
        ],
        margin=dict(l=60, r=30, t=90, b=60),
    )

    return fig


def visualizar_persona_json_interactivo(
    json_path,
    person_id=None,
    conf_threshold=0.0,
    output_html=None,
    show_figure=True,
    modo_visualizacion="original",
    dibujar_bbox=True,
    bbox_padding=12,
):
    json_path = Path(json_path)
    json_data = cargar_json(json_path)
    person_id = seleccionar_persona(json_data, person_id)
    fig = crear_figura_interactiva(
        json_data,
        json_path,
        person_id,
        conf_threshold,
        modo_visualizacion,
        dibujar_bbox,
        bbox_padding,
    )

    if output_html is None:
        output_html = Path(__file__).resolve().with_name(
            f"pose_json_{json_path.stem}_person_{person_id}.html"
        )

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html))

    print(f"Archivo JSON: {json_path}")
    print(f"Persona: {person_id}")
    print(f"Vista: {modo_visualizacion}")
    print(f"Bounding box: {'activado' if dibujar_bbox else 'desactivado'}")
    print(f"Visualizacion guardada en: {output_html}")

    if show_figure:
        fig.show()

    return fig


def main():
    json_files = obtener_json_files(JSON_SPLIT)
    json_path = json_files[JSON_FILE_INDEX]

    visualizar_persona_json_interactivo(
        json_path=json_path,
        person_id=PERSON_ID,
        conf_threshold=CONF_THRESHOLD,
        output_html=OUTPUT_HTML,
        show_figure=SHOW_FIGURE,
        modo_visualizacion=MODO_VISUALIZACION,
        dibujar_bbox=DIBUJAR_BBOX,
        bbox_padding=BBOX_PADDING,
    )


if __name__ == "__main__":
    main()
