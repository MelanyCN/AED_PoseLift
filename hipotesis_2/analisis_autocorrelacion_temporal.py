from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. Configuracion
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "pre_procesamiento" / "outputs" / "test_tabular.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "autocorrelacion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COLS = ["video_id", "frame_id", "person_id"]
LABEL_COL = "label"

MAX_LAG = 10
MIN_SEGMENT_LENGTH = 8

KEYPOINTS = [
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

GROUPS = {
    "Cabeza": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
    "Hombros": ["left_shoulder", "right_shoulder"],
    "Codos": ["left_elbow", "right_elbow"],
    "Muñecas": ["left_wrist", "right_wrist"],
    "Caderas": ["left_hip", "right_hip"],
    "Rodillas": ["left_knee", "right_knee"],
    "Tobillos": ["left_ankle", "right_ankle"],
}


# ============================================================
# 2. Carga y preparacion
# ============================================================

def cargar_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = ID_COLS + [LABEL_COL, "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    return df


def normalizar_keypoints_por_bbox(df):
    df = df.copy()

    bbox_w = df["bbox_x2"] - df["bbox_x1"]
    bbox_h = df["bbox_y2"] - df["bbox_y1"]

    bbox_w = bbox_w.replace(0, np.nan)
    bbox_h = bbox_h.replace(0, np.nan)

    for kp in KEYPOINTS:
        x_col = f"{kp}_x"
        y_col = f"{kp}_y"

        if x_col not in df.columns or y_col not in df.columns:
            print(f"Advertencia: no se encontraron columnas para {kp}")
            continue

        df[f"{kp}_x_norm"] = (df[x_col] - df["bbox_x1"]) / bbox_w
        df[f"{kp}_y_norm"] = (df[y_col] - df["bbox_y1"]) / bbox_h

    return df


def calcular_velocidades_normalizadas(df):
    df = df.copy()
    df = df.sort_values(["video_id", "person_id", "frame_id"])

    group_cols = ["video_id", "person_id"]

    for kp in KEYPOINTS:
        x_norm = f"{kp}_x_norm"
        y_norm = f"{kp}_y_norm"

        if x_norm not in df.columns or y_norm not in df.columns:
            continue

        dx = df.groupby(group_cols)[x_norm].diff()
        dy = df.groupby(group_cols)[y_norm].diff()
        dframe = df.groupby(group_cols)["frame_id"].diff()

        # Solo se consideran frames consecutivos o temporalmente ordenados.
        # Si dframe es mayor que 0, se calcula velocidad por unidad de frame.
        velocidad = np.sqrt(dx**2 + dy**2) / dframe.replace(0, np.nan)

        # Evitar velocidades entre cambios de etiqueta.
        prev_label = df.groupby(group_cols)[LABEL_COL].shift(1)
        velocidad = velocidad.where(prev_label == df[LABEL_COL])

        # Evitar velocidades entre saltos temporales negativos o nulos.
        velocidad = velocidad.where(dframe > 0)

        df[f"vel_{kp}"] = velocidad

    return df


def crear_segmentos_continuos(df):
    df = df.copy()
    df = df.sort_values(["video_id", "person_id", "frame_id"])

    group_cols = ["video_id", "person_id"]

    prev_frame = df.groupby(group_cols)["frame_id"].shift(1)
    prev_label = df.groupby(group_cols)[LABEL_COL].shift(1)

    cambio_frame = (df["frame_id"] - prev_frame) != 1
    cambio_label = df[LABEL_COL] != prev_label

    inicio_segmento = cambio_frame | cambio_label | prev_frame.isna()
    df["segment_id"] = inicio_segmento.groupby(
        [df["video_id"], df["person_id"]]
    ).cumsum()

    return df


# ============================================================
# 3. Autocorrelacion
# ============================================================

def autocorr_lag(series, lag):
    series = pd.to_numeric(series, errors="coerce").dropna()

    if len(series) <= lag + 2:
        return np.nan

    x = series.iloc[:-lag].to_numpy()
    y = series.iloc[lag:].to_numpy()

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return np.corrcoef(x, y)[0, 1]


def calcular_acf_por_segmento(df):
    registros = []
    vel_cols = [f"vel_{kp}" for kp in KEYPOINTS if f"vel_{kp}" in df.columns]

    segment_cols = ["video_id", "person_id", "segment_id", LABEL_COL]

    for keys, segment in df.groupby(segment_cols):
        video_id, person_id, segment_id, label = keys

        if len(segment) < MIN_SEGMENT_LENGTH:
            continue

        for vel_col in vel_cols:
            kp = vel_col.replace("vel_", "")

            for lag in range(1, MAX_LAG + 1):
                acf_value = autocorr_lag(segment[vel_col], lag)

                registros.append({
                    "video_id": video_id,
                    "person_id": person_id,
                    "segment_id": segment_id,
                    "label": int(label),
                    "keypoint": kp,
                    "variable": vel_col,
                    "lag": lag,
                    "acf": acf_value,
                    "segment_length": len(segment),
                })

    acf_df = pd.DataFrame(registros)
    return acf_df


def agregar_acf_por_keypoint(acf_df):
    resumen = (
        acf_df
        .dropna(subset=["acf"])
        .groupby(["label", "keypoint", "lag"], as_index=False)
        .agg(
            acf_promedio=("acf", "mean"),
            acf_mediana=("acf", "median"),
            n_segmentos=("acf", "count"),
        )
    )

    return resumen


def agregar_acf_por_grupo(acf_keypoint_df):
    rows = []

    for group_name, keypoints in GROUPS.items():
        subset = acf_keypoint_df[acf_keypoint_df["keypoint"].isin(keypoints)]

        if subset.empty:
            continue

        grouped = (
            subset
            .groupby(["label", "lag"], as_index=False)
            .agg(
                acf_promedio_grupo=("acf_promedio", "mean"),
                acf_mediana_grupo=("acf_mediana", "mean"),
                n_keypoints=("keypoint", "nunique"),
            )
        )

        grouped["grupo_corporal"] = group_name
        rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# ============================================================
# 4. Graficos
# ============================================================

def guardar_heatmap_acf_keypoints(acf_keypoint_df, label_value, filename):
    subset = acf_keypoint_df[acf_keypoint_df["label"] == label_value]

    pivot = subset.pivot(
        index="keypoint",
        columns="lag",
        values="acf_promedio",
    )

    pivot = pivot.reindex(KEYPOINTS)

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(pivot.values, aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    class_name = "normal" if label_value == 0 else "shoplifting"
    ax.set_title(f"Autocorrelacion promedio de velocidad por keypoint - {class_name}")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Keypoint")

    fig.colorbar(im, ax=ax, label="ACF promedio")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def guardar_barras_lag1_por_grupo(acf_group_df):
    lag1 = acf_group_df[acf_group_df["lag"] == 1].copy()

    if lag1.empty:
        print("No hay datos para graficar lag 1 por grupo.")
        return

    pivot = lag1.pivot(
        index="grupo_corporal",
        columns="label",
        values="acf_promedio_grupo",
    )

    pivot = pivot.reindex(list(GROUPS.keys()))

    x = np.arange(len(pivot.index))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    normal_values = pivot.get(0, pd.Series(index=pivot.index, dtype=float))
    shop_values = pivot.get(1, pd.Series(index=pivot.index, dtype=float))

    ax.bar(x - width / 2, normal_values, width, label="Normal")
    ax.bar(x + width / 2, shop_values, width, label="Shoplifting")

    ax.set_title("Autocorrelacion lag 1 de velocidad normalizada por grupo corporal")
    ax.set_xlabel("Grupo corporal")
    ax.set_ylabel("ACF promedio en lag 1")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "acf_lag1_grupos_corporales_por_clase.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def guardar_lineas_acf_por_grupo(acf_group_df):
    grupos = list(GROUPS.keys())

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.ravel()

    for ax, grupo in zip(axes, grupos):
        subset = acf_group_df[acf_group_df["grupo_corporal"] == grupo]

        for label_value, class_name in [(0, "Normal"), (1, "Shoplifting")]:
            data = subset[subset["label"] == label_value].sort_values("lag")

            if data.empty:
                continue

            ax.plot(
                data["lag"],
                data["acf_promedio_grupo"],
                marker="o",
                label=class_name,
            )

        ax.set_title(grupo)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF promedio")
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes[len(grupos):]:
        ax.axis("off")

    fig.suptitle("Curvas de autocorrelacion por grupo corporal y clase", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "curvas_acf_grupos_corporales_por_clase.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5. Tablas resumen
# ============================================================

def crear_tabla_lag1_grupos(acf_group_df):
    lag1 = acf_group_df[acf_group_df["lag"] == 1].copy()

    pivot = lag1.pivot(
        index="grupo_corporal",
        columns="label",
        values="acf_promedio_grupo",
    )

    pivot = pivot.rename(columns={
        0: "acf_lag1_normal",
        1: "acf_lag1_shoplifting",
    })

    if "acf_lag1_normal" not in pivot.columns:
        pivot["acf_lag1_normal"] = np.nan

    if "acf_lag1_shoplifting" not in pivot.columns:
        pivot["acf_lag1_shoplifting"] = np.nan

    pivot["diferencia_abs"] = (
        pivot["acf_lag1_shoplifting"] - pivot["acf_lag1_normal"]
    ).abs()

    pivot = pivot.reindex(list(GROUPS.keys()))
    pivot = pivot.reset_index()

    return pivot


def escribir_conclusion(tabla_lag1):
    conclusion_path = OUTPUT_DIR / "conclusion_autocorrelacion.txt"

    tabla_ordenada = tabla_lag1.sort_values("diferencia_abs", ascending=False)

    with open(conclusion_path, "w", encoding="utf-8") as f:
        f.write("Analisis de autocorrelacion temporal\n")
        f.write("=" * 80 + "\n\n")

        f.write("Metodo:\n")
        f.write(
            "Se normalizaron las coordenadas de cada keypoint respecto al bounding box "
            "de la persona. Luego se calculo la velocidad normalizada entre frames "
            "consecutivos para los 17 keypoints COCO. La autocorrelacion se calculo "
            "por segmentos continuos de una misma persona, dentro de un mismo video "
            "y manteniendo la misma etiqueta de clase.\n\n"
        )

        f.write("Tabla ACF lag 1 por grupo corporal:\n")
        f.write(tabla_lag1.to_string(index=False))
        f.write("\n\n")

        f.write("Grupos con mayor diferencia absoluta entre clases:\n")
        f.write(tabla_ordenada.head(5).to_string(index=False))
        f.write("\n\n")

        f.write("Texto breve para informe:\n")
        f.write(
            "La autocorrelacion temporal se evaluo sobre la velocidad normalizada de "
            "los 17 keypoints COCO17, agrupados por region corporal. Este analisis "
            "permite medir si el movimiento actual conserva dependencia con frames "
            "anteriores. Los resultados se compararon entre comportamiento normal y "
            "shoplifting para identificar si alguna region corporal presenta distinta "
            "persistencia temporal del movimiento.\n"
        )


# ============================================================
# 6. Main
# ============================================================

def main():
    print(f"Cargando dataset desde: {DATA_PATH}")
    df = cargar_dataset()
    print("Shape original:", df.shape)

    df = normalizar_keypoints_por_bbox(df)
    df = calcular_velocidades_normalizadas(df)
    df = crear_segmentos_continuos(df)

    acf_segmentos = calcular_acf_por_segmento(df)
    acf_keypoints = agregar_acf_por_keypoint(acf_segmentos)
    acf_grupos = agregar_acf_por_grupo(acf_keypoints)
    tabla_lag1 = crear_tabla_lag1_grupos(acf_grupos)

    acf_segmentos.to_csv(OUTPUT_DIR / "acf_por_segmento_keypoint.csv", index=False)
    acf_keypoints.to_csv(OUTPUT_DIR / "acf_promedio_por_keypoint.csv", index=False)
    acf_grupos.to_csv(OUTPUT_DIR / "acf_promedio_por_grupo_corporal.csv", index=False)
    tabla_lag1.to_csv(OUTPUT_DIR / "acf_lag1_por_grupo_corporal.csv", index=False)

    guardar_heatmap_acf_keypoints(
        acf_keypoints,
        label_value=0,
        filename="heatmap_acf_keypoints_normal.png",
    )

    guardar_heatmap_acf_keypoints(
        acf_keypoints,
        label_value=1,
        filename="heatmap_acf_keypoints_shoplifting.png",
    )

    guardar_barras_lag1_por_grupo(acf_grupos)
    guardar_lineas_acf_por_grupo(acf_grupos)
    escribir_conclusion(tabla_lag1)

    print("\nResultados guardados en:", OUTPUT_DIR)
    print("\nTabla ACF lag 1 por grupo corporal:")
    print(tabla_lag1)


if __name__ == "__main__":
    main()