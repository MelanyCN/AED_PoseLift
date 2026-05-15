from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
JSON_DIRS = {
    "train": BASE_DIR / "Json_files" / "data" / "PoseLift" / "pose" / "train",
    "test": BASE_DIR / "Json_files" / "data" / "PoseLift" / "pose" / "test",
}

# Cambia estos valores si quieres visualizar otra persona o frame.
JSON_SPLIT = "test"
JSON_FILE_INDEX = 0
PERSON_ID = None
FRAME_ID = None
CONF_THRESHOLD = 0.0

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
    name: plt.get_cmap("tab20")(i)
    for i, name in enumerate(COCO17_KEYPOINT_NAMES)
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


def seleccionar_persona_y_frame(json_data, person_id=None, frame_id=None):
    if person_id is None:
        person_id = sorted(json_data.keys(), key=ordenar_id)[0]

    person_id = str(person_id)

    if person_id not in json_data:
        raise KeyError(f"No existe person_id={person_id} en este JSON.")

    person_data = json_data[person_id]

    if frame_id is None:
        frame_id = sorted(person_data.keys(), key=ordenar_id)[0]

    frame_id = str(frame_id)

    if frame_id not in person_data:
        raise KeyError(f"No existe frame_id={frame_id} para person_id={person_id}.")

    return person_id, frame_id


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


def configurar_grafica(ax, titulo):
    ax.set_title(titulo)
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")


def dibujar_keypoints(ax, keypoints, conf_threshold):
    for i, kp_name in enumerate(COCO17_KEYPOINT_NAMES):
        punto = keypoints[i]

        if not punto_valido(punto, conf_threshold):
            continue

        x, y, _ = punto
        color = COCO17_KEYPOINT_COLORS[kp_name]

        ax.scatter(
            x,
            y,
            s=70,
            color=color,
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )
        ax.text(
            x + 3,
            y + 3,
            kp_name,
            fontsize=8,
            color=color,
            weight="bold",
            zorder=4,
        )


def graficar_keypoints_json(json_data, person_id=None, frame_id=None, conf_threshold=0.0):
    person_id, frame_id = seleccionar_persona_y_frame(json_data, person_id, frame_id)
    keypoints = extraer_keypoints(json_data, person_id, frame_id)

    fig, ax = plt.subplots(figsize=(7, 8))
    dibujar_keypoints(ax, keypoints, conf_threshold)
    configurar_grafica(
        ax,
        f"Keypoints COCO17\nPersona: {person_id} | Frame: {frame_id}",
    )
    plt.tight_layout()
    plt.show()


def graficar_esqueleto_json(json_data, person_id=None, frame_id=None, conf_threshold=0.0):
    person_id, frame_id = seleccionar_persona_y_frame(json_data, person_id, frame_id)
    keypoints = extraer_keypoints(json_data, person_id, frame_id)
    keypoints_por_nombre = dict(zip(COCO17_KEYPOINT_NAMES, keypoints))

    fig, ax = plt.subplots(figsize=(7, 8))

    for kp1, kp2 in COCO17_SKELETON:
        punto_1 = keypoints_por_nombre[kp1]
        punto_2 = keypoints_por_nombre[kp2]

        if punto_valido(punto_1, conf_threshold) and punto_valido(punto_2, conf_threshold):
            ax.plot(
                [punto_1[0], punto_2[0]],
                [punto_1[1], punto_2[1]],
                color="black",
                linewidth=2,
                alpha=0.65,
                zorder=2,
            )

    dibujar_keypoints(ax, keypoints, conf_threshold)
    configurar_grafica(
        ax,
        f"Esqueleto COCO17\nPersona: {person_id} | Frame: {frame_id}",
    )
    plt.tight_layout()
    plt.show()


def main():
    json_files = obtener_json_files(JSON_SPLIT)
    json_path = json_files[JSON_FILE_INDEX]
    json_data = cargar_json(json_path)
    person_id, frame_id = seleccionar_persona_y_frame(json_data, PERSON_ID, FRAME_ID)

    print(f"Archivo JSON: {json_path}")
    print(f"Persona: {person_id} | Frame: {frame_id}")

    graficar_keypoints_json(json_data, person_id, frame_id, CONF_THRESHOLD)
    graficar_esqueleto_json(json_data, person_id, frame_id, CONF_THRESHOLD)


if __name__ == "__main__":
    main()
