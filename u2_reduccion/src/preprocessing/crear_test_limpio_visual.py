from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "pre_procesamiento" / "outputs" / "test_tabular.csv"
OUTPUT_PATH = MODULE_ROOT / "data" / "processed" / "test_limpio_visual.csv"

CONF_THRESHOLD = 0.3

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


def transform_visual_coordinates(x_values: pd.Series, y_values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Transformacion visual reutilizada del dashboard: rot90_cw => x_vis=y, y_vis=-x."""
    return y_values, -x_values


def build_clean_visual_dataset(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    original_rows = len(df)

    required_columns = {
        "video_id",
        "frame_id",
        "person_id",
        "label",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    df["bbox_w"] = df["bbox_x2"] - df["bbox_x1"]
    df["bbox_h"] = df["bbox_y2"] - df["bbox_y1"]

    valid_columns = []
    conf_columns = []

    for keypoint in COCO17_KEYPOINTS:
        x_col = f"{keypoint}_x"
        y_col = f"{keypoint}_y"
        conf_col = f"{keypoint}_conf"

        for col in (x_col, y_col, conf_col):
            if col not in df.columns:
                raise ValueError(f"Falta la columna de keypoint: {col}")

        width = df["bbox_w"].replace(0, np.nan)
        height = df["bbox_h"].replace(0, np.nan)
        df[f"{keypoint}_x_norm"] = (df[x_col] - df["bbox_x1"]) / width
        df[f"{keypoint}_y_norm"] = (df[y_col] - df["bbox_y1"]) / height

        df[f"{keypoint}_x_vis"], df[f"{keypoint}_y_vis"] = transform_visual_coordinates(
            df[x_col], df[y_col]
        )

        valid_col = f"{keypoint}_valid"
        df[valid_col] = (
            df[x_col].notna()
            & df[y_col].notna()
            & df[conf_col].notna()
            & (df[conf_col] >= CONF_THRESHOLD)
        )
        valid_columns.append(valid_col)
        conf_columns.append(conf_col)

    df["num_keypoints_validos"] = df[valid_columns].sum(axis=1)
    df["porcentaje_keypoints_validos"] = df["num_keypoints_validos"] / len(COCO17_KEYPOINTS)
    df["mean_keypoint_conf"] = df[conf_columns].mean(axis=1, skipna=True)

    conditions = [
        df["porcentaje_keypoints_validos"] >= 0.75,
        df["porcentaje_keypoints_validos"] >= 0.50,
    ]
    df["pose_quality"] = np.select(conditions, ["alta", "media"], default="baja")

    if len(df) != original_rows:
        raise RuntimeError("El preprocesamiento elimino registros, lo cual no esta permitido.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    result = build_clean_visual_dataset()
    print(f"Archivo generado: {OUTPUT_PATH}")
    print(f"Filas conservadas: {len(result)}")
