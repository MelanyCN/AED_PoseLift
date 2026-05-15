from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr


# ============================================================
# 1. Configuración
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "pre_procesamiento" / "outputs" / "test_tabular.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.30

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

GRUPOS_CORPORALES = {
    "cabeza": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
    "hombros": ["left_shoulder", "right_shoulder"],
    "codos": ["left_elbow", "right_elbow"],
    "munecas": ["left_wrist", "right_wrist"],
    "caderas": ["left_hip", "right_hip"],
    "rodillas": ["left_knee", "right_knee"],
    "tobillos": ["left_ankle", "right_ankle"],
}

GRUPO_POR_KEYPOINT = {
    kp: grupo
    for grupo, keypoints in GRUPOS_CORPORALES.items()
    for kp in keypoints
}


# ============================================================
# 2. Carga y normalización
# ============================================================

def cargar_dataset():
    df = pd.read_csv(DATA_PATH)

    if "label" not in df.columns:
        raise ValueError("Este análisis requiere test_tabular.csv con columna label.")

    return df


def normalizar_por_bbox(df):
    df = df.copy()

    # En el dataset inspeccionado, el bbox crudo corresponde a:
    # [y1, x1, y2, x2]
    # aunque quedó guardado con nombres bbox_x1, bbox_y1, bbox_x2, bbox_y2.
    df["bbox_real_y1"] = df["bbox_x1"]
    df["bbox_real_x1"] = df["bbox_y1"]
    df["bbox_real_y2"] = df["bbox_x2"]
    df["bbox_real_x2"] = df["bbox_y2"]

    df["bbox_width"] = df["bbox_real_x2"] - df["bbox_real_x1"]
    df["bbox_height"] = df["bbox_real_y2"] - df["bbox_real_y1"]

    df = df[(df["bbox_width"] > 0) & (df["bbox_height"] > 0)].copy()

    for kp in COCO17_KEYPOINTS:
        x_col = f"{kp}_x"
        y_col = f"{kp}_y"
        conf_col = f"{kp}_conf"

        x_norm_col = f"{kp}_x_norm"
        y_norm_col = f"{kp}_y_norm"

        df[x_norm_col] = (df[x_col] - df["bbox_real_x1"]) / df["bbox_width"]
        df[y_norm_col] = (df[y_col] - df["bbox_real_y1"]) / df["bbox_height"]

        valido = (
            df[x_col].notna()
            & df[y_col].notna()
            & df[conf_col].notna()
            & (df[conf_col] >= CONF_THRESHOLD)
        )

        df.loc[~valido, [x_norm_col, y_norm_col]] = np.nan

    return df


# ============================================================
# 3. Métricas por keypoint
# ============================================================

def rank_biserial_from_mannwhitney(x0, x1):
    """
    Tamaño de efecto no paramétrico.
    x0 = valores label 0
    x1 = valores label 1

    Retorna una magnitud entre -1 y 1 aproximadamente.
    Valores cercanos a 0 indican poca separación.
    """
    x0 = pd.Series(x0).dropna()
    x1 = pd.Series(x1).dropna()

    n0 = len(x0)
    n1 = len(x1)

    if n0 == 0 or n1 == 0:
        return np.nan, np.nan

    stat, p_value = mannwhitneyu(x1, x0, alternative="two-sided")

    # Rank-biserial correlation:
    # r_rb = 2U / (n1*n0) - 1
    effect = (2 * stat) / (n1 * n0) - 1

    return effect, p_value


def calcular_metricas_keypoints(df):
    rows = []

    for kp in COCO17_KEYPOINTS:
        grupo = GRUPO_POR_KEYPOINT[kp]

        x_col = f"{kp}_x_norm"
        y_col = f"{kp}_y_norm"

        normal_x = df.loc[df["label"] == 0, x_col].dropna()
        shop_x = df.loc[df["label"] == 1, x_col].dropna()

        normal_y = df.loc[df["label"] == 0, y_col].dropna()
        shop_y = df.loc[df["label"] == 1, y_col].dropna()

        median_x_0 = normal_x.median()
        median_x_1 = shop_x.median()
        median_y_0 = normal_y.median()
        median_y_1 = shop_y.median()

        diff_x = median_x_1 - median_x_0
        diff_y = median_y_1 - median_y_0
        diff_total = np.sqrt(diff_x**2 + diff_y**2)

        effect_x, p_x = rank_biserial_from_mannwhitney(normal_x, shop_x)
        effect_y, p_y = rank_biserial_from_mannwhitney(normal_y, shop_y)

        effect_total = np.sqrt(effect_x**2 + effect_y**2)

        corr_x, corr_x_p = spearmanr(
            df[x_col],
            df["label"],
            nan_policy="omit"
        )

        corr_y, corr_y_p = spearmanr(
            df[y_col],
            df["label"],
            nan_policy="omit"
        )

        rows.append({
            "keypoint": kp,
            "grupo": grupo,

            "n_normal_x": len(normal_x),
            "n_shoplifting_x": len(shop_x),
            "n_normal_y": len(normal_y),
            "n_shoplifting_y": len(shop_y),

            "median_x_normal": median_x_0,
            "median_x_shoplifting": median_x_1,
            "diff_x_median": diff_x,

            "median_y_normal": median_y_0,
            "median_y_shoplifting": median_y_1,
            "diff_y_median": diff_y,

            "diff_total_median": diff_total,

            "rank_biserial_x": effect_x,
            "rank_biserial_y": effect_y,
            "effect_total": effect_total,

            "mannwhitney_p_x": p_x,
            "mannwhitney_p_y": p_y,

            "spearman_x_label": corr_x,
            "spearman_y_label": corr_y,
        })

    return pd.DataFrame(rows).sort_values("effect_total", ascending=False)


# ============================================================
# 4. Métricas por grupo corporal
# ============================================================

def calcular_metricas_grupos(df_keypoints):
    df_grupos = (
        df_keypoints
        .groupby("grupo")
        .agg(
            keypoints=("keypoint", "count"),
            diff_total_mediana_grupo=("diff_total_median", "median"),
            diff_total_promedio_grupo=("diff_total_median", "mean"),
            effect_total_mediana_grupo=("effect_total", "median"),
            effect_total_promedio_grupo=("effect_total", "mean"),
            abs_spearman_y_promedio=("spearman_y_label", lambda x: x.abs().mean()),
            abs_spearman_x_promedio=("spearman_x_label", lambda x: x.abs().mean()),
        )
        .reset_index()
        .sort_values("effect_total_mediana_grupo", ascending=False)
    )

    return df_grupos


# ============================================================
# 5. Gráficos
# ============================================================

def graficar_grupos(df_grupos):
    df_plot = df_grupos.sort_values("effect_total_mediana_grupo", ascending=True)

    plt.figure(figsize=(9, 5))
    plt.barh(df_plot["grupo"], df_plot["effect_total_mediana_grupo"])
    plt.xlabel("Tamaño de efecto mediano del grupo")
    plt.ylabel("Grupo corporal")
    plt.title("Hipótesis 1: grupos corporales más distintivos entre normal y shoplifting")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "h1_ranking_grupos_effect_total.png", dpi=300)
    plt.show()


def graficar_keypoints(df_keypoints):
    df_plot = df_keypoints.sort_values("effect_total", ascending=True)

    colores = df_plot["grupo"].map({
        "cabeza": "#4c78a8",
        "hombros": "#72b7b2",
        "codos": "#f58518",
        "munecas": "#e45756",
        "caderas": "#54a24b",
        "rodillas": "#b279a2",
        "tobillos": "#ff9da6",
    })

    plt.figure(figsize=(10, 7))
    plt.barh(df_plot["keypoint"], df_plot["effect_total"], color=colores)
    plt.xlabel("Tamaño de efecto total")
    plt.ylabel("Keypoint")
    plt.title("Hipótesis 1: keypoints más distintivos entre normal y shoplifting")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "h1_ranking_keypoints_effect_total.png", dpi=300)
    plt.show()


def graficar_heatmap_grupo_metricas(df_grupos):
    metricas = [
        "diff_total_mediana_grupo",
        "effect_total_mediana_grupo",
        "abs_spearman_y_promedio",
        "abs_spearman_x_promedio",
    ]

    df_plot = df_grupos.set_index("grupo")[metricas]

    matriz = df_plot.to_numpy()
    matriz_norm = matriz / np.nanmax(matriz, axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matriz_norm, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(np.arange(len(metricas)))
    ax.set_xticklabels(
        [
            "Diff. mediana",
            "Efecto mediano",
            "|Spearman Y|",
            "|Spearman X|",
        ],
        rotation=30,
        ha="right"
    )

    ax.set_yticks(np.arange(len(df_plot.index)))
    ax.set_yticklabels(df_plot.index)

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(
                j,
                i,
                f"{matriz[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8
            )

    ax.set_title("Hipótesis 1: comparación de métricas por grupo corporal")
    fig.colorbar(im, ax=ax, label="Valor normalizado por métrica")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "h1_heatmap_metricas_grupo.png", dpi=300)
    plt.show()


# ============================================================
# 6. Main
# ============================================================

def main():
    df = cargar_dataset()

    print("Dataset original:", df.shape)
    print("Distribución de label:")
    print(df["label"].value_counts().sort_index())

    df_norm = normalizar_por_bbox(df)

    df_keypoints = calcular_metricas_keypoints(df_norm)
    df_grupos = calcular_metricas_grupos(df_keypoints)

    df_norm.to_csv(OUTPUT_DIR / "test_tabular_h1_normalizado.csv", index=False)
    df_keypoints.to_csv(OUTPUT_DIR / "h1_metricas_keypoints.csv", index=False)
    df_grupos.to_csv(OUTPUT_DIR / "h1_metricas_grupos.csv", index=False)

    print("\nRanking por keypoint:")
    print(df_keypoints[[
        "keypoint",
        "grupo",
        "diff_total_median",
        "effect_total",
        "spearman_y_label",
        "spearman_x_label",
    ]])

    print("\nRanking por grupo corporal:")
    print(df_grupos)

    graficar_grupos(df_grupos)
    graficar_keypoints(df_keypoints)
    graficar_heatmap_grupo_metricas(df_grupos)

    print(f"\nResultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
