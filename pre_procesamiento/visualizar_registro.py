import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("outputs/test_tabular.csv")

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
def graficar_puntos_con_bbox_corregido(df, row_index=0, conf_threshold=0.0):
    row = df.iloc[row_index]

    plt.figure(figsize=(7, 8))

    # Según lo observado en el pkl:
    # bbox original = [y1, x1, y2, x2]
    y1 = row["bbox_x1"]
    x1 = row["bbox_y1"]
    y2 = row["bbox_x2"]
    x2 = row["bbox_y2"]

    # Dibujar bbox
    plt.plot(
        [x1, x2, x2, x1, x1],
        [y1, y1, y2, y2, y1],
        linewidth=2,
        label="bbox corregido"
    )

    # Dibujar keypoints
    for kp in COCO17_KEYPOINT_NAMES:
        x = row[f"{kp}_x"]
        y = row[f"{kp}_y"]
        conf = row[f"{kp}_conf"]

        if pd.notna(x) and pd.notna(y) and pd.notna(conf):
            if conf >= conf_threshold:
                plt.scatter(x, y, s=50)
                plt.text(x + 3, y + 3, kp, fontsize=8)

    title = (
        f"Pose COCO17 con bbox corregido\n"
        f"video_id={row['video_id']} | frame_id={row['frame_id']} | person_id={row['person_id']}"
    )

    if "label" in df.columns:
        label = int(row["label"])
        label_name = "normal" if label == 0 else "shoplifting"
        title += f" | label={label} ({label_name})"

    plt.title(title)
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()


def graficar_puntos_pose(df, row_index=0, conf_threshold=0.0):
    row = df.iloc[row_index]

    plt.figure(figsize=(7, 8))

    for kp in COCO17_KEYPOINT_NAMES:
        x = row[f"{kp}_x"]
        y = row[f"{kp}_y"]
        conf = row[f"{kp}_conf"]

        if pd.notna(x) and pd.notna(y) and pd.notna(conf):
            if conf >= conf_threshold:
                plt.scatter(x, y, s=50)
                plt.text(x + 3, y + 3, kp, fontsize=8)

    title = (
        f"Puntos COCO17\n"
        f"video_id={row['video_id']} | frame_id={row['frame_id']} | person_id={row['person_id']}"
    )

    if "label" in df.columns:
        label = int(row["label"])
        label_name = "normal" if label == 0 else "shoplifting"
        title += f" | label={label} ({label_name})"

    plt.title(title)
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

graficar_puntos_pose(df, row_index=0, conf_threshold=0.0)
graficar_puntos_con_bbox_corregido(df, row_index=0, conf_threshold=0.0)

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

def graficar_esqueleto_pose(df, row_index=0, conf_threshold=0.0):
    row = df.iloc[row_index]

    keypoints = {}

    for kp in COCO17_KEYPOINT_NAMES:
        keypoints[kp] = {
            "x": row[f"{kp}_x"],
            "y": row[f"{kp}_y"],
            "conf": row[f"{kp}_conf"],
        }

    plt.figure(figsize=(7, 8))

    # bbox observado como [y1, x1, y2, x2]
    y1 = row["bbox_x1"]
    x1 = row["bbox_y1"]
    y2 = row["bbox_x2"]
    x2 = row["bbox_y2"]

    plt.plot(
        [x1, x2, x2, x1, x1],
        [y1, y1, y2, y2, y1],
        linewidth=2,
        label="bbox corregido"
    )

    # Dibujar conexiones
    for kp1, kp2 in COCO17_SKELETON:
        p1 = keypoints[kp1]
        p2 = keypoints[kp2]

        if (
            pd.notna(p1["x"]) and pd.notna(p1["y"]) and pd.notna(p1["conf"]) and
            pd.notna(p2["x"]) and pd.notna(p2["y"]) and pd.notna(p2["conf"]) and
            p1["conf"] >= conf_threshold and
            p2["conf"] >= conf_threshold
        ):
            plt.plot(
                [p1["x"], p2["x"]],
                [p1["y"], p2["y"]],
                linewidth=2
            )

    # Dibujar puntos y nombres
    for kp, p in keypoints.items():
        if pd.notna(p["x"]) and pd.notna(p["y"]) and pd.notna(p["conf"]):
            if p["conf"] >= conf_threshold:
                plt.scatter(p["x"], p["y"], s=50)
                plt.text(p["x"] + 3, p["y"] + 3, kp, fontsize=8)

    title = (
        f"Esqueleto COCO17 reconstruido\n"
        f"video_id={row['video_id']} | frame_id={row['frame_id']} | person_id={row['person_id']}"
    )

    if "label" in df.columns:
        label = int(row["label"])
        label_name = "normal" if label == 0 else "shoplifting"
        title += f" | label={label} ({label_name})"

    plt.title(title)
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()

graficar_esqueleto_pose(df, row_index=0, conf_threshold=0.0)

