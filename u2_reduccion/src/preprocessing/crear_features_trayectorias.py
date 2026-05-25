from pathlib import Path

import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = MODULE_ROOT / "data" / "processed" / "test_limpio_visual.csv"
OUTPUT_PATH = MODULE_ROOT / "data" / "processed" / "features_trayectorias.csv"

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


def predominant_quality(values: pd.Series) -> str:
    counts = values.value_counts()
    if counts.empty:
        return "baja"
    priority = {"alta": 2, "media": 1, "baja": 0}
    return sorted(counts.index, key=lambda item: (counts[item], priority.get(item, -1)), reverse=True)[0]


def build_trajectory_features(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required = {"video_id", "person_id", "frame_id", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    aggregations = {
        "frame_id": ["count", "min", "max"],
        "label": ["sum", "mean"],
        "num_keypoints_validos": "mean",
        "porcentaje_keypoints_validos": "mean",
        "mean_keypoint_conf": "mean",
        "pose_quality": predominant_quality,
    }

    for keypoint in COCO17_KEYPOINTS:
        aggregations[f"{keypoint}_x_norm"] = ["mean", "std"]
        aggregations[f"{keypoint}_y_norm"] = ["mean", "std"]
        aggregations[f"{keypoint}_conf"] = "mean"

    grouped = df.groupby(["video_id", "person_id"], as_index=False).agg(aggregations)
    grouped.columns = [
        "_".join([part for part in col if part]).strip("_")
        if isinstance(col, tuple)
        else col
        for col in grouped.columns
    ]

    grouped = grouped.rename(
        columns={
            "frame_id_count": "num_frames",
            "frame_id_min": "frame_min",
            "frame_id_max": "frame_max",
            "label_sum": "label_sum",
            "label_mean": "porcentaje_shoplifting",
            "num_keypoints_validos_mean": "num_keypoints_validos_mean",
            "porcentaje_keypoints_validos_mean": "porcentaje_keypoints_validos_mean",
            "mean_keypoint_conf_mean": "mean_keypoint_conf_mean",
            "pose_quality_predominant_quality": "pose_quality_predominante",
        }
    )
    grouped["label_trayectoria"] = (grouped["porcentaje_shoplifting"] > 0).astype(int)

    first_columns = [
        "video_id",
        "person_id",
        "num_frames",
        "frame_min",
        "frame_max",
        "label_sum",
        "porcentaje_shoplifting",
        "label_trayectoria",
        "pose_quality_predominante",
        "num_keypoints_validos_mean",
        "porcentaje_keypoints_validos_mean",
        "mean_keypoint_conf_mean",
    ]
    remaining_columns = [col for col in grouped.columns if col not in first_columns]
    grouped = grouped[first_columns + remaining_columns]

    expected_rows = df[["video_id", "person_id"]].drop_duplicates().shape[0]
    if len(grouped) != expected_rows:
        raise RuntimeError("La tabla de features no tiene una fila por trayectoria unica.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, index=False)
    return grouped


if __name__ == "__main__":
    result = build_trajectory_features()
    print(f"Archivo generado: {OUTPUT_PATH}")
    print(f"Trayectorias: {len(result)}")
