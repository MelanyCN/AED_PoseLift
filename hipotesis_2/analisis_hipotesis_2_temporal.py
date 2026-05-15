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

TEMPORAL_FEATURES = [
    "left_wrist_speed",
    "right_wrist_speed",
    "left_elbow_speed",
    "right_elbow_speed",
    "left_wrist_to_left_hip_dist",
    "right_wrist_to_right_hip_dist",
    "left_elbow_to_left_hip_dist",
    "right_elbow_to_right_hip_dist",
    "delta_left_wrist_to_left_hip_dist",
    "delta_right_wrist_to_right_hip_dist",
    "delta_left_elbow_to_left_hip_dist",
    "delta_right_elbow_to_right_hip_dist",
]


# ============================================================
# 2. Carga y normalización
# ============================================================

def cargar_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if "label" not in df.columns:
        raise ValueError("Este análisis requiere test_tabular.csv con columna label.")

    return df


def normalizar_por_bbox(df):
    df = df.copy()

    # En la inspección del dataset, el bbox crudo corresponde a:
    # [y1, x1, y2, x2]
    # aunque en el CSV quedó nombrado como bbox_x1, bbox_y1, bbox_x2, bbox_y2.
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
# 3. Features temporales y relacionales
# ============================================================

def distancia_2d(df, kp1, kp2):
    dx = df[f"{kp1}_x_norm"] - df[f"{kp2}_x_norm"]
    dy = df[f"{kp1}_y_norm"] - df[f"{kp2}_y_norm"]
    return np.sqrt(dx**2 + dy**2)


def calcular_velocidad_por_keypoint(df, kp):
    """
    Calcula velocidad frame a frame para un keypoint normalizado.
    La velocidad es distancia euclidiana entre la posición actual y la anterior
    dentro de la misma trayectoria (video_id, person_id).
    """

    x_col = f"{kp}_x_norm"
    y_col = f"{kp}_y_norm"

    dx = df.groupby(["video_id", "person_id"])[x_col].diff()
    dy = df.groupby(["video_id", "person_id"])[y_col].diff()

    return np.sqrt(dx**2 + dy**2)


def agregar_features_temporales(df):
    df = df.copy()
    df = df.sort_values(["video_id", "person_id", "frame_id"]).copy()

    # Velocidad de muñecas y codos
    for kp in ["left_wrist", "right_wrist", "left_elbow", "right_elbow"]:
        df[f"{kp}_speed"] = calcular_velocidad_por_keypoint(df, kp)

    # Distancias relativas respecto a cadera
    df["left_wrist_to_left_hip_dist"] = distancia_2d(df, "left_wrist", "left_hip")
    df["right_wrist_to_right_hip_dist"] = distancia_2d(df, "right_wrist", "right_hip")

    df["left_elbow_to_left_hip_dist"] = distancia_2d(df, "left_elbow", "left_hip")
    df["right_elbow_to_right_hip_dist"] = distancia_2d(df, "right_elbow", "right_hip")

    # Cambios temporales de esas distancias
    for feature in [
        "left_wrist_to_left_hip_dist",
        "right_wrist_to_right_hip_dist",
        "left_elbow_to_left_hip_dist",
        "right_elbow_to_right_hip_dist",
    ]:
        df[f"delta_{feature}"] = df.groupby(["video_id", "person_id"])[feature].diff()

    return df


# ============================================================
# 4. Métricas estadísticas
# ============================================================

def rank_biserial_from_mannwhitney(x0, x1):
    """
    Tamaño de efecto no paramétrico.
    x0 = valores label 0
    x1 = valores label 1

    Retorna:
    - rank-biserial correlation
    - p-value Mann-Whitney
    """
    x0 = pd.Series(x0).dropna()
    x1 = pd.Series(x1).dropna()

    n0 = len(x0)
    n1 = len(x1)

    if n0 == 0 or n1 == 0:
        return np.nan, np.nan

    stat, p_value = mannwhitneyu(x1, x0, alternative="two-sided")

    effect = (2 * stat) / (n1 * n0) - 1

    return effect, p_value


def calcular_metricas_temporales(df):
    rows = []

    for feature in TEMPORAL_FEATURES:
        normal = df.loc[df["label"] == 0, feature].dropna()
        shop = df.loc[df["label"] == 1, feature].dropna()

        median_normal = normal.median()
        median_shop = shop.median()

        mean_normal = normal.mean()
        mean_shop = shop.mean()

        diff_median = median_shop - median_normal
        abs_diff_median = abs(diff_median)

        effect, p_value = rank_biserial_from_mannwhitney(normal, shop)

        corr, corr_p = spearmanr(
            df[feature],
            df["label"],
            nan_policy="omit"
        )

        rows.append({
            "feature": feature,
            "n_normal": len(normal),
            "n_shoplifting": len(shop),
            "median_normal": median_normal,
            "median_shoplifting": median_shop,
            "diff_median": diff_median,
            "abs_diff_median": abs_diff_median,
            "mean_normal": mean_normal,
            "mean_shoplifting": mean_shop,
            "diff_mean": mean_shop - mean_normal,
            "rank_biserial": effect,
            "abs_rank_biserial": abs(effect),
            "mannwhitney_p": p_value,
            "spearman_label": corr,
            "abs_spearman_label": abs(corr),
        })

    return pd.DataFrame(rows).sort_values("abs_rank_biserial", ascending=False)


# ============================================================
# 5. Gráficos
# ============================================================

def graficar_boxplot_velocidad(df):
    features = [
        "left_wrist_speed",
        "right_wrist_speed",
        "left_elbow_speed",
        "right_elbow_speed",
    ]

    data = []
    positions = []
    labels_plot = []
    colors = []
    pos = 1

    for feature in features:
        for label in [0, 1]:
            valores = df.loc[df["label"] == label, feature].dropna().values
            data.append(valores)
            positions.append(pos)
            labels_plot.append(f"{feature}\nlabel={label}")
            colors.append("#4c78a8" if label == 0 else "#f58518")
            pos += 1

        pos += 0.5

    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(
        data,
        positions=positions,
        showfliers=False,
        patch_artist=True,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_plot, rotation=45, ha="right")
    ax.set_ylabel("Velocidad normalizada")
    ax.set_title("Hipótesis 2: velocidad de muñecas y codos por clase")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "h2_boxplot_velocidad_extremidades.png", dpi=300)
    plt.show()


def graficar_ranking_features_temporales(df_metrics):
    df_plot = df_metrics.sort_values("abs_rank_biserial", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_plot["feature"], df_plot["abs_rank_biserial"], color="#4c78a8")
    ax.set_xlabel("Tamaño de efecto absoluto")
    ax.set_ylabel("Feature temporal / relacional")
    ax.set_title("Hipótesis 2: ranking de variables temporales más distintivas")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "h2_ranking_features_temporales.png", dpi=300)
    plt.show()


def seleccionar_trayectoria_con_shoplifting(df):
    """
    Selecciona una trayectoria con suficientes frames normales y shoplifting.
    Esto evita elegir casos donde casi toda la secuencia pertenece a una sola clase.
    """
    resumen = (
        df.groupby(["video_id", "person_id"])
        .agg(
            n_frames=("frame_id", "count"),
            n_shop=("label", "sum"),
            n_normal=("label", lambda x: (x == 0).sum()),
        )
        .reset_index()
    )

    # Exigir presencia real de ambas clases
    resumen = resumen[
        (resumen["n_shop"] >= 10)
        & (resumen["n_normal"] >= 10)
        & (resumen["n_frames"] >= 30)
    ].copy()

    if len(resumen) == 0:
        raise ValueError(
            "No se encontró una trayectoria con al menos 10 frames normales "
            "y 10 frames shoplifting."
        )

    # Buscar trayectorias más balanceadas
    resumen["balance"] = abs(resumen["n_shop"] - resumen["n_normal"])
    resumen = resumen.sort_values(["balance", "n_frames"], ascending=[True, False])

    elegido = resumen.iloc[0]

    sub = df[
        (df["video_id"] == elegido["video_id"])
        & (df["person_id"] == elegido["person_id"])
    ].copy()

    sub = sub.sort_values("frame_id").copy()

    print("\nTrayectoria seleccionada para serie temporal:")
    print(elegido)

    return sub

def graficar_serie_temporal_extremidades(df):
    sub = seleccionar_trayectoria_con_shoplifting(df)

    fig, ax1 = plt.subplots(figsize=(13, 5))

    variables = [
        "right_wrist_speed",
        "right_elbow_speed",
        "right_wrist_to_right_hip_dist",
        "right_elbow_to_right_hip_dist",
    ]

    for var in variables:
        ax1.plot(
            sub["frame_id"],
            sub[var],
            marker="o",
            linewidth=1.5,
            label=var,
        )

    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Valor normalizado")
    ax1.set_title(
        "Hipótesis 2: evolución temporal de extremidades superiores\n"
        f"video_id={sub['video_id'].iloc[0]} | person_id={sub['person_id'].iloc[0]}"
    )
    ax1.grid(True, alpha=0.3)

    # Sombrear segmentos label=1
    frames_shop = sub.loc[sub["label"] == 1, "frame_id"].values

    if len(frames_shop) > 0:
        ax1.axvspan(
            frames_shop.min(),
            frames_shop.max(),
            color="red",
            alpha=0.12,
            label="segmento shoplifting"
        )

    ax2 = ax1.twinx()
    ax2.step(
        sub["frame_id"],
        sub["label"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="label",
        where="mid",
    )
    ax2.set_ylabel("Label")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["normal", "shoplifting"])

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "h2_serie_temporal_extremidades.png", dpi=300)
    plt.show()


def graficar_heatmap_temporal(df):
    sub = seleccionar_trayectoria_con_shoplifting(df)

    features = [
        "left_wrist_speed",
        "right_wrist_speed",
        "left_elbow_speed",
        "right_elbow_speed",
        "left_wrist_to_left_hip_dist",
        "right_wrist_to_right_hip_dist",
        "left_elbow_to_left_hip_dist",
        "right_elbow_to_right_hip_dist",
    ]

    matrix = sub[features].T.copy()

    # Normalización por fila para visualizar patrones relativos
    matrix_norm = matrix.copy()
    for idx in matrix_norm.index:
        row = matrix_norm.loc[idx]
        min_v = row.min(skipna=True)
        max_v = row.max(skipna=True)

        if pd.notna(min_v) and pd.notna(max_v) and max_v > min_v:
            matrix_norm.loc[idx] = (row - min_v) / (max_v - min_v)

    fig, ax = plt.subplots(figsize=(13, 5))

    im = ax.imshow(
        matrix_norm,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )

    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features)

    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(sub["frame_id"].astype(int).astype(str), rotation=90, fontsize=7)

    ax.set_title(
        "Hipótesis 2: mapa temporal de variables dinámicas\n"
        f"video_id={sub['video_id'].iloc[0]} | person_id={sub['person_id'].iloc[0]}"
    )

    ax.set_xlabel("Frame")
    fig.colorbar(im, ax=ax, label="Valor normalizado por variable")

    # Marcar frames label=1
    for j, label in enumerate(sub["label"].values):
        if label == 1:
            ax.axvline(j, color="blue", linewidth=0.8, alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "h2_heatmap_temporal_extremidades.png", dpi=300)
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
    df_temporal = agregar_features_temporales(df_norm)
    df_metrics = calcular_metricas_temporales(df_temporal)

    df_temporal.to_csv(OUTPUT_DIR / "test_tabular_h2_temporal.csv", index=False)
    df_metrics.to_csv(OUTPUT_DIR / "h2_metricas_temporales.csv", index=False)

    print("\nDataset con features temporales:", df_temporal.shape)

    print("\nRanking de variables temporales:")
    print(df_metrics[[
        "feature",
        "median_normal",
        "median_shoplifting",
        "diff_median",
        "rank_biserial",
        "spearman_label",
        "mannwhitney_p",
    ]])

    graficar_boxplot_velocidad(df_temporal)
    graficar_ranking_features_temporales(df_metrics)
    graficar_serie_temporal_extremidades(df_temporal)
    graficar_heatmap_temporal(df_temporal)

    print(f"\nResultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
