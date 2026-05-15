from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. Configuracion
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "pre_procesamiento" / "outputs" / "test_tabular.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "label"
ID_COLS = ["video_id", "frame_id", "person_id"]

WINDOW_BEFORE = 15
WINDOW_AFTER = 20

# Si quieres forzar una trayectoria especifica, coloca valores aqui.
# Por ejemplo:
# FORCE_VIDEO_ID = "1_242"
# FORCE_PERSON_ID = 4
FORCE_VIDEO_ID = None
FORCE_PERSON_ID = None


# ============================================================
# 2. Carga y validacion
# ============================================================

def cargar_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_cols = [
        "video_id", "frame_id", "person_id", LABEL_COL,
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "left_wrist_x", "left_wrist_y",
        "right_wrist_x", "right_wrist_y",
        "left_hip_x", "left_hip_y",
        "right_hip_x", "right_hip_y",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas necesarias: {missing}")

    df = df.copy()
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce")
    df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce")
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")

    df = df.dropna(subset=["frame_id", "person_id", LABEL_COL])
    df["frame_id"] = df["frame_id"].astype(int)
    df["person_id"] = df["person_id"].astype(int)
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    return df


# ============================================================
# 3. Variables relativas al bounding box y cadera
# ============================================================

def agregar_variables_relativas(df):
    df = df.copy()

    bbox_w = df["bbox_x2"] - df["bbox_x1"]
    bbox_h = df["bbox_y2"] - df["bbox_y1"]

    bbox_w = bbox_w.replace(0, np.nan)
    bbox_h = bbox_h.replace(0, np.nan)

    keypoints = [
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
    ]

    for kp in keypoints:
        df[f"{kp}_x_norm"] = (df[f"{kp}_x"] - df["bbox_x1"]) / bbox_w
        df[f"{kp}_y_norm"] = (df[f"{kp}_y"] - df["bbox_y1"]) / bbox_h

    # Centro de cadera normalizado
    df["hip_center_x_norm"] = (
        df["left_hip_x_norm"] + df["right_hip_x_norm"]
    ) / 2

    df["hip_center_y_norm"] = (
        df["left_hip_y_norm"] + df["right_hip_y_norm"]
    ) / 2

    # Centro de muñecas normalizado
    df["wrist_center_x_norm"] = (
        df["left_wrist_x_norm"] + df["right_wrist_x_norm"]
    ) / 2

    df["wrist_center_y_norm"] = (
        df["left_wrist_y_norm"] + df["right_wrist_y_norm"]
    ) / 2

    # Diferencia vertical muñeca-cadera.
    # En coordenadas de imagen, valores y mayores suelen estar mas abajo.
    df["left_wrist_hip_vertical"] = (
        df["left_wrist_y_norm"] - df["left_hip_y_norm"]
    )

    df["right_wrist_hip_vertical"] = (
        df["right_wrist_y_norm"] - df["right_hip_y_norm"]
    )

    df["wrist_center_hip_vertical"] = (
        df["wrist_center_y_norm"] - df["hip_center_y_norm"]
    )

    # Distancia euclidiana muñeca-cadera
    df["left_wrist_hip_dist"] = np.sqrt(
        (df["left_wrist_x_norm"] - df["left_hip_x_norm"]) ** 2
        + (df["left_wrist_y_norm"] - df["left_hip_y_norm"]) ** 2
    )

    df["right_wrist_hip_dist"] = np.sqrt(
        (df["right_wrist_x_norm"] - df["right_hip_x_norm"]) ** 2
        + (df["right_wrist_y_norm"] - df["right_hip_y_norm"]) ** 2
    )

    df["wrist_center_hip_dist"] = np.sqrt(
        (df["wrist_center_x_norm"] - df["hip_center_x_norm"]) ** 2
        + (df["wrist_center_y_norm"] - df["hip_center_y_norm"]) ** 2
    )

    return df


def agregar_deltas_temporales(df):
    df = df.copy()
    df = df.sort_values(["video_id", "person_id", "frame_id"])

    group_cols = ["video_id", "person_id"]

    variables = [
        "left_wrist_y_norm",
        "right_wrist_y_norm",
        "wrist_center_y_norm",
        "left_wrist_hip_vertical",
        "right_wrist_hip_vertical",
        "wrist_center_hip_vertical",
        "left_wrist_hip_dist",
        "right_wrist_hip_dist",
        "wrist_center_hip_dist",
    ]

    for var in variables:
        df[f"delta_{var}"] = df.groupby(group_cols)[var].diff()

    return df


# ============================================================
# 4. Identificacion de transiciones normal -> shoplifting
# ============================================================

def detectar_transiciones(df):
    df = df.copy()
    df = df.sort_values(["video_id", "person_id", "frame_id"])

    group_cols = ["video_id", "person_id"]
    df["prev_label"] = df.groupby(group_cols)[LABEL_COL].shift(1)

    transiciones = df[
        (df["prev_label"] == 0)
        & (df[LABEL_COL] == 1)
    ].copy()

    transiciones = transiciones[
        ["video_id", "person_id", "frame_id"]
    ].rename(columns={"frame_id": "transition_frame"})

    return transiciones


def extraer_ventanas_transicion(df, transiciones):
    ventanas = []

    for _, row in transiciones.iterrows():
        video_id = row["video_id"]
        person_id = int(row["person_id"])
        transition_frame = int(row["transition_frame"])

        trayectoria = df[
            (df["video_id"] == video_id)
            & (df["person_id"] == person_id)
        ].copy()

        trayectoria = trayectoria.sort_values("frame_id")

        start = transition_frame - WINDOW_BEFORE
        end = transition_frame + WINDOW_AFTER

        ventana = trayectoria[
            (trayectoria["frame_id"] >= start)
            & (trayectoria["frame_id"] <= end)
        ].copy()

        if ventana.empty:
            continue

        ventana["transition_frame"] = transition_frame
        ventana["relative_frame"] = ventana["frame_id"] - transition_frame

        ventanas.append(ventana)

    if not ventanas:
        return pd.DataFrame()

    return pd.concat(ventanas, ignore_index=True)


def resumir_transiciones(ventanas):
    variables = [
        "left_wrist_y_norm",
        "right_wrist_y_norm",
        "wrist_center_y_norm",
        "left_wrist_hip_vertical",
        "right_wrist_hip_vertical",
        "wrist_center_hip_vertical",
        "left_wrist_hip_dist",
        "right_wrist_hip_dist",
        "wrist_center_hip_dist",
    ]

    resumen = []

    for var in variables:
        before = ventanas[ventanas["relative_frame"] < 0][var].dropna()
        onset = ventanas[
            (ventanas["relative_frame"] >= 0)
            & (ventanas["relative_frame"] <= 5)
        ][var].dropna()
        after = ventanas[ventanas["relative_frame"] > 5][var].dropna()

        resumen.append({
            "variable": var,
            "media_antes": before.mean(),
            "media_inicio": onset.mean(),
            "media_despues": after.mean(),
            "mediana_antes": before.median(),
            "mediana_inicio": onset.median(),
            "mediana_despues": after.median(),
            "delta_inicio_vs_antes": onset.median() - before.median(),
            "delta_despues_vs_antes": after.median() - before.median(),
            "n_antes": len(before),
            "n_inicio": len(onset),
            "n_despues": len(after),
        })

    return pd.DataFrame(resumen)


# ============================================================
# 5. Seleccion de trayectoria representativa
# ============================================================

def seleccionar_trayectoria_representativa(df, transiciones):
    if FORCE_VIDEO_ID is not None and FORCE_PERSON_ID is not None:
        selected = transiciones[
            (transiciones["video_id"].astype(str) == str(FORCE_VIDEO_ID))
            & (transiciones["person_id"].astype(int) == int(FORCE_PERSON_ID))
        ]

        if not selected.empty:
            return selected.iloc[0]

        print("Advertencia: no se encontro la trayectoria forzada. Se seleccionara automaticamente.")

    candidatos = []

    for _, row in transiciones.iterrows():
        video_id = row["video_id"]
        person_id = int(row["person_id"])
        transition_frame = int(row["transition_frame"])

        trayectoria = df[
            (df["video_id"] == video_id)
            & (df["person_id"] == person_id)
        ].copy()

        before = trayectoria[
            (trayectoria["frame_id"] < transition_frame)
            & (trayectoria["frame_id"] >= transition_frame - WINDOW_BEFORE)
        ]

        after = trayectoria[
            (trayectoria["frame_id"] >= transition_frame)
            & (trayectoria["frame_id"] <= transition_frame + WINDOW_AFTER)
        ]

        if len(before) >= 5 and len(after) >= 5:
            candidatos.append({
                "video_id": video_id,
                "person_id": person_id,
                "transition_frame": transition_frame,
                "n_before": len(before),
                "n_after": len(after),
                "total_window": len(before) + len(after),
            })

    if not candidatos:
        raise ValueError("No se encontraron trayectorias con suficientes frames antes y despues de la transicion.")

    candidatos_df = pd.DataFrame(candidatos)
    candidatos_df = candidatos_df.sort_values(
        ["total_window", "n_before", "n_after"],
        ascending=False,
    )

    return candidatos_df.iloc[0]


# ============================================================
# 6. Graficos
# ============================================================

def graficar_serie_trayectoria(df, selected):
    video_id = selected["video_id"]
    person_id = int(selected["person_id"])
    transition_frame = int(selected["transition_frame"])

    trayectoria = df[
        (df["video_id"] == video_id)
        & (df["person_id"] == person_id)
    ].copy()

    trayectoria = trayectoria[
        (trayectoria["frame_id"] >= transition_frame - WINDOW_BEFORE)
        & (trayectoria["frame_id"] <= transition_frame + WINDOW_AFTER)
    ].copy()

    trayectoria["relative_frame"] = trayectoria["frame_id"] - transition_frame
    trayectoria = trayectoria.sort_values("relative_frame")

    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.plot(
        trayectoria["relative_frame"],
        trayectoria["left_wrist_hip_vertical"],
        marker="o",
        label="Muñeca izq. - cadera izq.",
    )

    ax1.plot(
        trayectoria["relative_frame"],
        trayectoria["right_wrist_hip_vertical"],
        marker="o",
        label="Muñeca der. - cadera der.",
    )

    ax1.plot(
        trayectoria["relative_frame"],
        trayectoria["wrist_center_hip_vertical"],
        marker="o",
        label="Centro muñecas - centro cadera",
    )

    ax1.axvline(0, linestyle="--", linewidth=2, label="Inicio shoplifting")
    ax1.axhline(0, linestyle=":", linewidth=1)

    ax1.set_title(
        f"H3: posicion relativa muñeca-cadera alrededor del inicio anomalo\n"
        f"video_id={video_id}, person_id={person_id}, frame_inicio={transition_frame}"
    )

    ax1.set_xlabel("Frame relativo al inicio de shoplifting")
    ax1.set_ylabel("Diferencia vertical normalizada")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "h3_serie_muñeca_cadera_inicio_anomalia.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def graficar_boxplot_ventanas(ventanas):
    ventanas = ventanas.copy()

    def periodo(rel_frame):
        if rel_frame < 0:
            return "Antes"
        if rel_frame <= 5:
            return "Inicio"
        return "Despues"

    ventanas["periodo"] = ventanas["relative_frame"].apply(periodo)

    variables = [
        "left_wrist_hip_vertical",
        "right_wrist_hip_vertical",
        "wrist_center_hip_vertical",
        "left_wrist_hip_dist",
        "right_wrist_hip_dist",
        "wrist_center_hip_dist",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    orden = ["Antes", "Inicio", "Despues"]

    for ax, var in zip(axes, variables):
        data = [
            ventanas.loc[ventanas["periodo"] == p, var].dropna().to_numpy()
            for p in orden
        ]

        ax.boxplot(data, labels=orden, showfliers=False)
        ax.set_title(var)
        ax.set_ylabel("Valor normalizado")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("H3: comparacion de posicion relativa antes, durante y despues del inicio anomalo")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "h3_boxplot_ventanas_inicio_anomalia.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def graficar_perfil_promedio_transiciones(ventanas):
    variables = [
        "left_wrist_hip_vertical",
        "right_wrist_hip_vertical",
        "wrist_center_hip_vertical",
    ]

    agg = (
        ventanas
        .groupby("relative_frame")[variables]
        .median()
        .reset_index()
        .sort_values("relative_frame")
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    for var in variables:
        ax.plot(
            agg["relative_frame"],
            agg[var],
            marker="o",
            label=var,
        )

    ax.axvline(0, linestyle="--", linewidth=2, label="Inicio shoplifting")
    ax.axhline(0, linestyle=":", linewidth=1)

    ax.set_title("H3: perfil mediano de posicion muñeca-cadera alrededor de transiciones")
    ax.set_xlabel("Frame relativo al inicio de shoplifting")
    ax.set_ylabel("Diferencia vertical normalizada mediana")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "h3_perfil_promedio_transiciones.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def graficar_heatmap_trayectoria(df, selected):
    video_id = selected["video_id"]
    person_id = int(selected["person_id"])
    transition_frame = int(selected["transition_frame"])

    trayectoria = df[
        (df["video_id"] == video_id)
        & (df["person_id"] == person_id)
    ].copy()

    trayectoria = trayectoria[
        (trayectoria["frame_id"] >= transition_frame - WINDOW_BEFORE)
        & (trayectoria["frame_id"] <= transition_frame + WINDOW_AFTER)
    ].copy()

    trayectoria["relative_frame"] = trayectoria["frame_id"] - transition_frame
    trayectoria = trayectoria.sort_values("relative_frame")

    vars_heatmap = [
        "left_wrist_y_norm",
        "right_wrist_y_norm",
        "left_hip_y_norm",
        "right_hip_y_norm",
        "left_wrist_hip_vertical",
        "right_wrist_hip_vertical",
        "wrist_center_hip_vertical",
        "left_wrist_hip_dist",
        "right_wrist_hip_dist",
        "wrist_center_hip_dist",
    ]

    matrix = trayectoria[vars_heatmap].copy()

    # Normalizacion por fila para visualizacion
    matrix_t = matrix.T
    matrix_norm = matrix_t.sub(matrix_t.min(axis=1), axis=0)
    denom = matrix_t.max(axis=1) - matrix_t.min(axis=1)
    matrix_norm = matrix_norm.div(denom.replace(0, np.nan), axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix_norm.values, aspect="auto", vmin=0, vmax=1)

    ax.set_yticks(np.arange(len(vars_heatmap)))
    ax.set_yticklabels(vars_heatmap)

    ax.set_xticks(np.arange(len(trayectoria)))
    ax.set_xticklabels(trayectoria["relative_frame"].astype(int), rotation=90)

    transition_pos = np.where(trayectoria["relative_frame"].to_numpy() == 0)[0]
    if len(transition_pos) > 0:
        ax.axvline(transition_pos[0], color="red", linestyle="--", linewidth=2)

    ax.set_title(
        f"H3: mapa temporal alrededor del inicio anomalo\n"
        f"video_id={video_id}, person_id={person_id}"
    )
    ax.set_xlabel("Frame relativo al inicio de shoplifting")
    ax.set_ylabel("Variable")

    fig.colorbar(im, ax=ax, label="Valor normalizado por variable")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "h3_heatmap_inicio_anomalia.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# 7. Conclusion automatica
# ============================================================

def escribir_conclusion(resumen, selected, n_transiciones):
    path = OUTPUT_DIR / "h3_conclusion_inicio_anomalia.txt"

    variables_clave = [
        "left_wrist_hip_vertical",
        "right_wrist_hip_vertical",
        "wrist_center_hip_vertical",
        "left_wrist_hip_dist",
        "right_wrist_hip_dist",
        "wrist_center_hip_dist",
    ]

    resumen_clave = resumen[resumen["variable"].isin(variables_clave)]

    with open(path, "w", encoding="utf-8") as f:
        f.write("Hipotesis 3: Momento del comportamiento anomalalo\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Transiciones normal -> shoplifting detectadas: {n_transiciones}\n")
        f.write(
            f"Trayectoria representativa: video_id={selected['video_id']}, "
            f"person_id={int(selected['person_id'])}, "
            f"frame_inicio={int(selected['transition_frame'])}\n\n"
        )

        f.write("Resumen de variables antes, inicio y despues:\n")
        f.write(resumen_clave.to_string(index=False))
        f.write("\n\n")

        f.write("Texto breve para informe:\n")
        f.write(
            "Para evaluar la Hipotesis 3 se analizaron ventanas temporales alrededor "
            "del primer frame etiquetado como shoplifting dentro de trayectorias que "
            "presentaban una transicion desde comportamiento normal. Las coordenadas "
            "de muñecas y caderas fueron normalizadas respecto al bounding box de la "
            "persona, y se calcularon diferencias verticales y distancias relativas "
            "entre muñecas y caderas. Este analisis permite observar si el inicio del "
            "comportamiento anomalalo coincide con un cambio en la posicion relativa "
            "de las muñecas respecto a la zona de la cadera.\n"
        )


# ============================================================
# 8. Main
# ============================================================

def main():
    print(f"Cargando dataset desde: {DATA_PATH}")
    df = cargar_dataset()
    print("Shape original:", df.shape)

    df = agregar_variables_relativas(df)
    df = agregar_deltas_temporales(df)

    transiciones = detectar_transiciones(df)
    print("Transiciones normal -> shoplifting detectadas:", len(transiciones))

    if transiciones.empty:
        raise ValueError("No se encontraron transiciones normal -> shoplifting.")

    ventanas = extraer_ventanas_transicion(df, transiciones)

    if ventanas.empty:
        raise ValueError("No se pudieron extraer ventanas alrededor de transiciones.")

    resumen = resumir_transiciones(ventanas)
    selected = seleccionar_trayectoria_representativa(df, transiciones)

    df.to_csv(OUTPUT_DIR / "h3_dataset_variables_relativas.csv", index=False)
    transiciones.to_csv(OUTPUT_DIR / "h3_transiciones_detectadas.csv", index=False)
    ventanas.to_csv(OUTPUT_DIR / "h3_ventanas_transicion.csv", index=False)
    resumen.to_csv(OUTPUT_DIR / "h3_resumen_ventanas.csv", index=False)

    graficar_serie_trayectoria(df, selected)
    graficar_boxplot_ventanas(ventanas)
    graficar_perfil_promedio_transiciones(ventanas)
    graficar_heatmap_trayectoria(df, selected)

    escribir_conclusion(resumen, selected, len(transiciones))

    print("\nTrayectoria representativa seleccionada:")
    print(selected)

    print("\nResumen de ventanas:")
    print(resumen)

    print("\nResultados guardados en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()