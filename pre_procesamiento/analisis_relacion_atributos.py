from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. Configuracion
# ============================================================

PREPROCESAMIENTO_DIR = Path(__file__).resolve().parent
DATA_PATH = PREPROCESAMIENTO_DIR / "outputs" / "test_tabular.csv"
OUTPUT_DIR = PREPROCESAMIENTO_DIR / "outputs" / "relacion_atributos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORR_GENERAL_CSV = OUTPUT_DIR / "matriz_correlacion_general_test.csv"
CORR_LABEL_CSV = OUTPUT_DIR / "correlacion_con_label.csv"
TOP_LABEL_CSV = OUTPUT_DIR / "top20_correlacion_label.csv"
CORR_BBOX_KEYPOINTS_CSV = OUTPUT_DIR / "correlacion_bbox_keypoints.csv"
CORR_GROUPS_CSV = OUTPUT_DIR / "correlacion_label_por_grupo_corporal.csv"
RESUMEN_TXT = OUTPUT_DIR / "resumen_relacion_atributos.txt"

TOP_N = 20
MOSTRAR_GRAFICOS = True
GUARDAR_GRAFICOS = True

ID_COLS = ["video_id", "frame_id", "person_id"]
BBOX_COLS = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
LABEL_COL = "label"

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

BODY_GROUPS = {
    "cabeza": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
    "hombros": ["left_shoulder", "right_shoulder"],
    "codos": ["left_elbow", "right_elbow"],
    "munecas": ["left_wrist", "right_wrist"],
    "caderas": ["left_hip", "right_hip"],
    "rodillas": ["left_knee", "right_knee"],
    "tobillos": ["left_ankle", "right_ankle"],
}

UPPER_BODY_COLS = [
    "left_shoulder_x",
    "left_shoulder_y",
    "left_shoulder_conf",
    "right_shoulder_x",
    "right_shoulder_y",
    "right_shoulder_conf",
    "left_elbow_x",
    "left_elbow_y",
    "left_elbow_conf",
    "right_elbow_x",
    "right_elbow_y",
    "right_elbow_conf",
    "left_wrist_x",
    "left_wrist_y",
    "left_wrist_conf",
    "right_wrist_x",
    "right_wrist_y",
    "right_wrist_conf",
    LABEL_COL,
]


# ============================================================
# 2. Utilidades
# ============================================================

def cargar_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe el CSV de test: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if LABEL_COL not in df.columns:
        raise ValueError(f"El dataset debe contener la columna {LABEL_COL}.")

    return df


def obtener_keypoint_cols(df):
    excluded_cols = set(ID_COLS + BBOX_COLS + [LABEL_COL])
    return [col for col in df.columns if col not in excluded_cols]


def seleccionar_numeric_cols(df):
    keypoint_cols = obtener_keypoint_cols(df)
    numeric_cols = BBOX_COLS + keypoint_cols + [LABEL_COL]
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    return numeric_cols, keypoint_cols


def guardar_o_mostrar(fig, filename):
    if GUARDAR_GRAFICOS:
        fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")

    if MOSTRAR_GRAFICOS:
        plt.show()
    else:
        plt.close(fig)


def plot_heatmap(
    matrix,
    title,
    filename,
    figsize=(12, 10),
    xtick_fontsize=7,
    ytick_fontsize=7,
):
    values = matrix.to_numpy(dtype=float)
    max_abs = np.nanmax(np.abs(values))

    if max_abs == 0 or np.isnan(max_abs):
        max_abs = 1

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(values, cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto")

    ax.set_title(title)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=xtick_fontsize)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=ytick_fontsize)

    fig.colorbar(im, ax=ax, label="Correlacion de Pearson")
    fig.tight_layout()
    guardar_o_mostrar(fig, filename)


# ============================================================
# 3. Calculos de correlacion
# ============================================================

def calcular_correlacion_general(df, numeric_cols):
    df_numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df_numeric.corr(method="pearson")


def calcular_correlacion_con_label(corr):
    corr_label = corr[LABEL_COL].drop(LABEL_COL)
    corr_label = corr_label.dropna()
    corr_label = corr_label.reindex(corr_label.abs().sort_values(ascending=False).index)

    return corr_label.reset_index().rename(
        columns={
            "index": "variable",
            LABEL_COL: "corr_label",
        }
    )


def calcular_correlacion_bbox_keypoints(corr, keypoint_cols):
    corr_bbox_keypoints = corr.loc[BBOX_COLS, keypoint_cols]
    rows = []

    for bbox_col in corr_bbox_keypoints.index:
        for variable in corr_bbox_keypoints.columns:
            corr_value = corr_bbox_keypoints.loc[bbox_col, variable]
            rows.append({
                "bbox_variable": bbox_col,
                "keypoint_variable": variable,
                "corr": corr_value,
                "abs_corr": abs(corr_value),
            })

    return (
        pd.DataFrame(rows)
        .sort_values("abs_corr", ascending=False)
        .reset_index(drop=True)
    )


def calcular_correlacion_grupos_label(df):
    group_results = []

    for group_name, keypoints in BODY_GROUPS.items():
        cols = []

        for kp in keypoints:
            cols.extend([f"{kp}_x", f"{kp}_y", f"{kp}_conf"])

        cols = [col for col in cols if col in df.columns]

        if not cols:
            continue

        corr_values = df[cols + [LABEL_COL]].corr(method="pearson")[LABEL_COL]
        corr_values = corr_values.drop(LABEL_COL).dropna()

        if corr_values.empty:
            continue

        variable_mayor_corr = corr_values.abs().idxmax()

        group_results.append({
            "grupo": group_name,
            "variables": len(corr_values),
            "max_abs_corr": corr_values.abs().max(),
            "mean_abs_corr": corr_values.abs().mean(),
            "variable_mayor_corr": variable_mayor_corr,
            "corr_variable_mayor": corr_values[variable_mayor_corr],
        })

    return (
        pd.DataFrame(group_results)
        .sort_values("max_abs_corr", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# 4. Graficos especificos
# ============================================================

def graficar_correlacion_general(corr):
    plot_heatmap(
        corr,
        "Matriz de correlacion entre atributos numericos del dataset test",
        "correlacion_general_test.png",
        figsize=(18, 16),
        xtick_fontsize=5,
        ytick_fontsize=5,
    )


def graficar_top_correlacion_label(corr_label_df):
    top_corr = corr_label_df.head(TOP_N).copy()
    top_corr = top_corr.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_corr["variable"], top_corr["corr_label"], color="#4c78a8")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"Top {TOP_N} variables con mayor correlacion con label")
    ax.set_xlabel("Correlacion de Pearson con label")
    ax.set_ylabel("Variable")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    guardar_o_mostrar(fig, "top20_correlacion_label.png")


def graficar_correlacion_extremidades_superiores(df):
    cols = [col for col in UPPER_BODY_COLS if col in df.columns]
    corr_upper = df[cols].corr(method="pearson")

    corr_upper.to_csv(OUTPUT_DIR / "matriz_correlacion_extremidades_superiores.csv")

    plot_heatmap(
        corr_upper,
        "Correlacion entre extremidades superiores y label",
        "correlacion_extremidades_superiores.png",
        figsize=(11, 9),
        xtick_fontsize=7,
        ytick_fontsize=7,
    )


def graficar_correlacion_label_por_grupo(group_corr_df):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(group_corr_df["grupo"], group_corr_df["mean_abs_corr"], color="#f58518")
    ax.set_title("Correlacion promedio absoluta con label por grupo corporal")
    ax.set_xlabel("Grupo corporal")
    ax.set_ylabel("Correlacion absoluta promedio con label")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    guardar_o_mostrar(fig, "correlacion_label_por_grupo_corporal.png")


def graficar_correlacion_bbox_keypoints(corr, keypoint_cols):
    corr_bbox_keypoints = corr.loc[BBOX_COLS, keypoint_cols]

    plot_heatmap(
        corr_bbox_keypoints,
        "Correlacion entre bounding box y keypoints",
        "correlacion_bbox_keypoints.png",
        figsize=(16, 4),
        xtick_fontsize=5,
        ytick_fontsize=8,
    )


# ============================================================
# 5. Resumen para informe
# ============================================================

def escribir_resumen(df, corr_label_df, group_corr_df, bbox_keypoints_df):
    top_label = corr_label_df.head(10)
    top_group = group_corr_df.head(1).iloc[0]
    top_bbox = bbox_keypoints_df.head(10)

    with open(RESUMEN_TXT, "w", encoding="utf-8") as file:
        file.write("Relacion entre atributos\n")
        file.write("=" * 80 + "\n\n")
        file.write(f"Dataset: {DATA_PATH}\n")
        file.write(f"Shape: {df.shape}\n")
        file.write("Distribucion de label:\n")
        file.write(df[LABEL_COL].value_counts(dropna=False).sort_index().to_string())
        file.write("\n\n")

        file.write("Top 10 variables por correlacion absoluta con label:\n")
        file.write(top_label.to_string(index=False))
        file.write("\n\n")

        file.write("Grupo con mayor correlacion maxima absoluta con label:\n")
        file.write(top_group.to_string())
        file.write("\n\n")

        file.write("Top 10 relaciones bbox-keypoint por correlacion absoluta:\n")
        file.write(top_bbox.to_string(index=False))
        file.write("\n\n")

        file.write("Texto sugerido para informe:\n")
        file.write(
            "Antes de evaluar las hipotesis, se realizo un analisis de "
            "correlacion entre los atributos numericos del dataset de prueba. "
            "Este analisis tuvo como objetivo identificar relaciones lineales "
            "entre coordenadas de keypoints, bounding boxes, valores de "
            "confianza y la variable label. La correlacion se calculo sobre "
            "test_tabular.csv, ya que este es el dataset que contiene etiquetas "
            "binarias. Una correlacion baja con label no implica ausencia de "
            "relacion, sino ausencia de una relacion lineal simple entre una "
            "variable individual y la clase.\n\n"
        )
        file.write(
            "El analisis permite observar correlaciones altas entre keypoints "
            "anatomicamente relacionados y entre algunos keypoints y el bounding "
            "box. Esto es esperable, porque las partes del cuerpo no se mueven "
            "de forma independiente y la posicion absoluta depende de la ubicacion "
            "de la persona en la escena. Si las correlaciones individuales con "
            "label son bajas, se justifica el analisis posterior por hipotesis, "
            "enfocado en grupos corporales, normalizacion espacial y patrones "
            "temporales.\n"
        )


# ============================================================
# 6. Main
# ============================================================

def main():
    df = cargar_dataset()
    numeric_cols, keypoint_cols = seleccionar_numeric_cols(df)

    print("Dataset:", df.shape)
    print("Distribucion de label:")
    print(df[LABEL_COL].value_counts(dropna=False).sort_index())
    print(f"Columnas numericas analizadas: {len(numeric_cols)}")

    corr = calcular_correlacion_general(df, numeric_cols)
    corr_label_df = calcular_correlacion_con_label(corr)
    bbox_keypoints_df = calcular_correlacion_bbox_keypoints(corr, keypoint_cols)
    group_corr_df = calcular_correlacion_grupos_label(df)

    corr.to_csv(CORR_GENERAL_CSV)
    corr_label_df.to_csv(CORR_LABEL_CSV, index=False)
    corr_label_df.head(TOP_N).to_csv(TOP_LABEL_CSV, index=False)
    bbox_keypoints_df.to_csv(CORR_BBOX_KEYPOINTS_CSV, index=False)
    group_corr_df.to_csv(CORR_GROUPS_CSV, index=False)

    print("\nTop 20 variables con mayor correlacion absoluta con label:")
    print(corr_label_df.head(TOP_N))

    print("\nCorrelacion con label por grupo corporal:")
    print(group_corr_df)

    print("\nTop 10 relaciones bbox-keypoint:")
    print(bbox_keypoints_df.head(10))

    graficar_correlacion_general(corr)
    graficar_top_correlacion_label(corr_label_df)
    graficar_correlacion_extremidades_superiores(df)
    graficar_correlacion_label_por_grupo(group_corr_df)
    graficar_correlacion_bbox_keypoints(corr, keypoint_cols)
    escribir_resumen(df, corr_label_df, group_corr_df, bbox_keypoints_df)

    print(f"\nResultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
