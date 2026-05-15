from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import normaltest, shapiro
from sklearn.feature_selection import mutual_info_classif


# ============================================================
# 1. Configuracion
# ============================================================

PREPROCESAMIENTO_DIR = Path(__file__).resolve().parent
DATA_PATH = PREPROCESAMIENTO_DIR / "outputs" / "test_tabular.csv"
OUTPUT_DIR = PREPROCESAMIENTO_DIR / "outputs" / "evaluar_normalidad"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NORMALITY_CSV = OUTPUT_DIR / "normalidad_todas_variables.csv"
NORMALITY_SUMMARY_CSV = OUTPUT_DIR / "resumen_normalidad.csv"
PEARSON_SPEARMAN_CSV = OUTPUT_DIR / "comparacion_pearson_spearman_label.csv"
MUTUAL_INFO_CSV = OUTPUT_DIR / "informacion_mutua_label.csv"
CONCLUSION_TXT = OUTPUT_DIR / "conclusion_metodo_correlacion.txt"

SAMPLE_SIZE = 5000
RANDOM_STATE = 42
TOP_N = 20

MOSTRAR_GRAFICOS = False
GUARDAR_GRAFICOS = True

ID_COLS = ["video_id", "frame_id", "person_id"]
LABEL_COL = "label"

EXCLUDE_NORMALITY_COLS = ID_COLS + [LABEL_COL]

PAIRS_TO_CHECK = [
    ("left_wrist_x", "right_wrist_x"),
    ("left_wrist_y", "right_wrist_y"),
    ("left_wrist_y", "left_hip_y"),
    ("right_wrist_y", "right_hip_y"),
    ("left_elbow_y", "left_wrist_y"),
    ("right_elbow_y", "right_wrist_y"),
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


def seleccionar_muestra(data):
    data = pd.to_numeric(data, errors="coerce").dropna()

    if len(data) > SAMPLE_SIZE:
        return data.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

    return data


def guardar_o_mostrar(fig, filename):
    if GUARDAR_GRAFICOS:
        fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")

    if MOSTRAR_GRAFICOS:
        plt.show()
    else:
        plt.close(fig)


def obtener_columnas_numericas(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols


def obtener_variables_para_normalidad(df):
    numeric_cols = obtener_columnas_numericas(df)

    vars_to_check = [
        col for col in numeric_cols
        if col not in EXCLUDE_NORMALITY_COLS
    ]

    return vars_to_check


def dividir_en_bloques(lista, tamano_bloque):
    for i in range(0, len(lista), tamano_bloque):
        yield lista[i:i + tamano_bloque]


# ============================================================
# 3. Normalidad
# ============================================================

def evaluar_normalidad(df, vars_to_check):
    resultados = []

    for col in vars_to_check:
        data = pd.to_numeric(df[col], errors="coerce").dropna()
        sample = seleccionar_muestra(data)

        if len(sample) < 8:
            shapiro_stat = np.nan
            shapiro_p = np.nan
            dagostino_stat = np.nan
            dagostino_p = np.nan
        else:
            shapiro_stat, shapiro_p = shapiro(sample)
            dagostino_stat, dagostino_p = normaltest(sample)

        resultados.append({
            "variable": col,
            "n": len(data),
            "n_muestra": len(sample),
            "n_nulos": df[col].isna().sum(),
            "porcentaje_nulos": df[col].isna().mean() * 100,
            "media": data.mean(),
            "mediana": data.median(),
            "desv_std": data.std(ddof=1),
            "min": data.min(),
            "q1": data.quantile(0.25),
            "q3": data.quantile(0.75),
            "max": data.max(),
            "asimetria": stats.skew(data, nan_policy="omit"),
            "curtosis": stats.kurtosis(data, nan_policy="omit"),
            "shapiro_stat": shapiro_stat,
            "shapiro_p": shapiro_p,
            "dagostino_stat": dagostino_stat,
            "dagostino_p": dagostino_p,
            "normal_shapiro_005": shapiro_p > 0.05 if not np.isnan(shapiro_p) else False,
            "normal_dagostino_005": dagostino_p > 0.05 if not np.isnan(dagostino_p) else False,
        })

    normality_df = pd.DataFrame(resultados)

    normality_df["conclusion_normalidad"] = np.where(
        normality_df["normal_shapiro_005"] & normality_df["normal_dagostino_005"],
        "aprox. normal",
        "no normal",
    )

    return normality_df


def resumir_normalidad(normality_df):
    total = len(normality_df)
    aprox_normal = (normality_df["conclusion_normalidad"] == "aprox. normal").sum()
    no_normal = (normality_df["conclusion_normalidad"] == "no normal").sum()

    return pd.DataFrame([{
        "variables_evaluadas": total,
        "aprox_normales": aprox_normal,
        "no_normales": no_normal,
        "porcentaje_no_normales": no_normal / total if total else np.nan,
    }])


# ============================================================
# 4. Graficos de distribucion y normalidad
# ============================================================

def graficar_histogramas(df, vars_to_check):
    n_cols = 4
    n_rows = 4
    vars_por_figura = n_cols * n_rows

    for idx, bloque in enumerate(dividir_en_bloques(vars_to_check, vars_por_figura), start=1):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
        axes = axes.ravel()

        for ax, col in zip(axes, bloque):
            data = pd.to_numeric(df[col], errors="coerce").dropna()

            ax.hist(data, bins=40, alpha=0.85)
            ax.set_title(col, fontsize=9)
            ax.grid(True, alpha=0.3)

        for ax in axes[len(bloque):]:
            ax.axis("off")

        fig.suptitle(
            f"Histogramas de variables numericas - bloque {idx}",
            fontsize=14,
        )
        fig.tight_layout()

        filename = f"histogramas_variables_{idx:02d}.png"
        guardar_o_mostrar(fig, filename)


def graficar_qqplots(df, vars_to_check):
    n_cols = 4
    n_rows = 4
    vars_por_figura = n_cols * n_rows

    for idx, bloque in enumerate(dividir_en_bloques(vars_to_check, vars_por_figura), start=1):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
        axes = axes.ravel()

        for ax, col in zip(axes, bloque):
            data = seleccionar_muestra(df[col])

            if len(data) >= 8:
                stats.probplot(data, dist="norm", plot=ax)
                ax.set_title(col, fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.set_title(f"{col}\nDatos insuficientes", fontsize=9)
                ax.axis("off")

        for ax in axes[len(bloque):]:
            ax.axis("off")

        fig.suptitle(
            f"Q-Q plots de variables numericas - bloque {idx}",
            fontsize=14,
        )
        fig.tight_layout()

        filename = f"qqplots_variables_{idx:02d}.png"
        guardar_o_mostrar(fig, filename)


def graficar_distribucion_label(df):
    conteo = df[LABEL_COL].value_counts().sort_index()
    porcentajes = conteo / conteo.sum() * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(conteo.index.astype(str), conteo.values)

    for bar, pct in zip(bars, porcentajes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}\n{pct:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("Distribucion de la variable label")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Cantidad de registros")
    ax.set_xticklabels(["0: normal", "1: shoplifting"])
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    guardar_o_mostrar(fig, "distribucion_label.png")


def graficar_nulos(df, vars_to_check):
    nulos = df[vars_to_check].isna().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=True)

    if nulos.empty:
        print("No se encontraron valores nulos en las variables evaluadas.")
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(nulos) * 0.25)))
    ax.barh(nulos.index, nulos.values)
    ax.set_title("Valores nulos por variable numerica")
    ax.set_xlabel("Cantidad de valores nulos")
    ax.set_ylabel("Variable")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    guardar_o_mostrar(fig, "nulos_por_variable.png")


def graficar_scatter_linealidad(df):
    pares_existentes = [
        (x, y) for x, y in PAIRS_TO_CHECK
        if x in df.columns and y in df.columns
    ]

    if not pares_existentes:
        print("No hay pares disponibles para graficar linealidad.")
        return

    n_cols = 3
    n_rows = int(np.ceil(len(pares_existentes) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    axes = np.array(axes).ravel()

    for ax, (x_col, y_col) in zip(axes, pares_existentes):
        data_pair = df[[x_col, y_col, LABEL_COL]].dropna()

        if len(data_pair) == 0:
            ax.set_title(f"{x_col} vs {y_col}\nSin datos")
            ax.axis("off")
            continue

        sample = data_pair.sample(
            min(len(data_pair), SAMPLE_SIZE),
            random_state=RANDOM_STATE,
        )

        colors = np.where(sample[LABEL_COL].to_numpy() == 1, "#f58518", "#4c78a8")
        ax.scatter(sample[x_col], sample[y_col], alpha=0.35, s=10, c=colors)
        ax.set_title(f"{x_col} vs {y_col}", fontsize=9)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(pares_existentes):]:
        ax.axis("off")

    fig.suptitle("Diagramas de dispersion para evaluar linealidad", fontsize=14)
    fig.tight_layout()
    guardar_o_mostrar(fig, "scatter_linealidad_pares_representativos.png")


# ============================================================
# 5. Pearson, Spearman e informacion mutua
# ============================================================

def comparar_pearson_spearman(df):
    numeric_cols = obtener_columnas_numericas(df)

    pearson_corr = df[numeric_cols].corr(method="pearson")
    spearman_corr = df[numeric_cols].corr(method="spearman")

    pearson_label = pearson_corr[LABEL_COL].drop(LABEL_COL)
    spearman_label = spearman_corr[LABEL_COL].drop(LABEL_COL)

    comparison = pd.DataFrame({
        "variable": pearson_label.index,
        "pearson": pearson_label.values,
        "spearman": spearman_label.reindex(pearson_label.index).values,
    })

    comparison["abs_pearson"] = comparison["pearson"].abs()
    comparison["abs_spearman"] = comparison["spearman"].abs()
    comparison["abs_difference"] = (
        comparison["pearson"] - comparison["spearman"]
    ).abs()

    return comparison.sort_values("abs_difference", ascending=False)


def calcular_informacion_mutua_label(df):
    numeric_cols = obtener_columnas_numericas(df)
    feature_cols = [
        col for col in numeric_cols
        if col not in ID_COLS + [LABEL_COL]
    ]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df[LABEL_COL].astype(int)

    X = X.fillna(X.median(numeric_only=True))

    mi_values = mutual_info_classif(
        X,
        y,
        discrete_features=False,
        random_state=RANDOM_STATE,
    )

    mi_df = pd.DataFrame({
        "variable": feature_cols,
        "mutual_info_label": mi_values,
    })

    return mi_df.sort_values("mutual_info_label", ascending=False)


def graficar_comparacion_pearson_spearman(comparison_df):
    top = comparison_df.head(TOP_N).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["variable"], top["abs_difference"])
    ax.set_title(f"Top {TOP_N}: diferencia absoluta entre Pearson y Spearman")
    ax.set_xlabel("|Pearson - Spearman| respecto a label")
    ax.set_ylabel("Variable")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    guardar_o_mostrar(fig, "top20_diferencia_pearson_spearman_label.png")


def graficar_top_informacion_mutua(mi_df):
    top = mi_df.head(TOP_N).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["variable"], top["mutual_info_label"])
    ax.set_title(f"Top {TOP_N}: informacion mutua con label")
    ax.set_xlabel("Informacion mutua con label")
    ax.set_ylabel("Variable")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    guardar_o_mostrar(fig, "top20_informacion_mutua_label.png")


def graficar_heatmap_spearman(df, vars_to_check):
    cols = vars_to_check + [LABEL_COL]
    corr = df[cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(18, 15))
    im = ax.imshow(corr.values, aspect="auto")

    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Matriz de correlacion de Spearman - variables numericas")

    fig.tight_layout()
    guardar_o_mostrar(fig, "heatmap_spearman_todas_variables.png")


# ============================================================
# 6. Conclusion automatica
# ============================================================

def decidir_metodo(normality_df, comparison_df):
    porcentaje_no_normales = (
        normality_df["conclusion_normalidad"].eq("no normal").mean()
        if len(normality_df)
        else np.nan
    )

    diferencia_media = comparison_df["abs_difference"].mean()
    diferencia_maxima = comparison_df["abs_difference"].max()

    if porcentaje_no_normales >= 0.5:
        metodo = "Spearman como correlacion exploratoria principal"
    elif diferencia_media > 0.05 or diferencia_maxima > 0.15:
        metodo = "Spearman como correlacion exploratoria principal"
    else:
        metodo = "Pearson y Spearman como referencias similares"

    return metodo, porcentaje_no_normales, diferencia_media, diferencia_maxima


def escribir_conclusion(normality_df, comparison_df, mi_df, vars_to_check):
    metodo, pct_no_normales, diff_media, diff_maxima = decidir_metodo(
        normality_df,
        comparison_df,
    )

    top_mi = mi_df.head(10)
    top_spearman = comparison_df.sort_values("abs_spearman", ascending=False).head(10)

    with open(CONCLUSION_TXT, "w", encoding="utf-8") as file:
        file.write("Evaluacion de normalidad y decision de correlacion\n")
        file.write("=" * 80 + "\n\n")
        file.write(f"Dataset: {DATA_PATH}\n")
        file.write(f"Variables numericas evaluadas para normalidad: {len(vars_to_check)}\n")
        file.write(f"Porcentaje de variables no normales: {pct_no_normales:.3f}\n")
        file.write(f"Diferencia media |Pearson - Spearman|: {diff_media:.3f}\n")
        file.write(f"Diferencia maxima |Pearson - Spearman|: {diff_maxima:.3f}\n")
        file.write(f"Decision practica: {metodo}\n\n")

        file.write("Resumen de normalidad:\n")
        file.write(
            normality_df[[
                "variable",
                "n",
                "n_muestra",
                "n_nulos",
                "shapiro_p",
                "dagostino_p",
                "conclusion_normalidad",
            ]].to_string(index=False)
        )
        file.write("\n\n")

        file.write("Top 10 por correlacion absoluta de Spearman con label:\n")
        file.write(
            top_spearman[[
                "variable",
                "pearson",
                "spearman",
                "abs_difference",
            ]].to_string(index=False)
        )
        file.write("\n\n")

        file.write("Top 10 por informacion mutua con label:\n")
        file.write(top_mi.to_string(index=False))
        file.write("\n\n")

        file.write("Texto breve para informe:\n")
        file.write(
            "Se evaluo la normalidad de todas las variables numericas del dataset, "
            "excluyendo los identificadores y la variable label. Para ello se usaron "
            "histogramas, graficos Q-Q y las pruebas de Shapiro-Wilk y D'Agostino. "
            "Debido a que la mayoria de variables no presento normalidad aproximada "
            "y las relaciones entre keypoints no son necesariamente lineales, se "
            "utilizo Spearman como medida principal de correlacion exploratoria. "
            "Pearson se empleo solo como referencia, e informacion mutua se uso "
            "para complementar el analisis de asociacion con la variable label.\n"
        )


# ============================================================
# 7. Main
# ============================================================

def main():
    df = cargar_dataset()

    vars_to_check = obtener_variables_para_normalidad(df)

    print("Dataset:", df.shape)
    print("Variables numericas evaluadas para normalidad:", len(vars_to_check))
    print("Variables evaluadas:")
    print(vars_to_check)

    print("\nDistribucion de label:")
    print(df[LABEL_COL].value_counts(dropna=False).sort_index())

    normality_df = evaluar_normalidad(df, vars_to_check)
    normality_summary_df = resumir_normalidad(normality_df)
    comparison_df = comparar_pearson_spearman(df)
    mi_df = calcular_informacion_mutua_label(df)

    normality_df.to_csv(NORMALITY_CSV, index=False)
    normality_summary_df.to_csv(NORMALITY_SUMMARY_CSV, index=False)
    comparison_df.to_csv(PEARSON_SPEARMAN_CSV, index=False)
    mi_df.to_csv(MUTUAL_INFO_CSV, index=False)

    print("\nNormalidad de todas las variables numericas:")
    print(normality_df[[
        "variable",
        "n",
        "n_muestra",
        "n_nulos",
        "shapiro_p",
        "dagostino_p",
        "conclusion_normalidad",
    ]])

    print("\nResumen normalidad:")
    print(normality_summary_df)

    print("\nTop 20 diferencias Pearson vs Spearman respecto a label:")
    print(comparison_df.head(TOP_N))

    print("\nTop 20 informacion mutua con label:")
    print(mi_df.head(TOP_N))

    graficar_distribucion_label(df)
    graficar_nulos(df, vars_to_check)
    graficar_histogramas(df, vars_to_check)
    graficar_qqplots(df, vars_to_check)
    graficar_scatter_linealidad(df)
    graficar_comparacion_pearson_spearman(comparison_df)
    graficar_top_informacion_mutua(mi_df)
    graficar_heatmap_spearman(df, vars_to_check)

    escribir_conclusion(normality_df, comparison_df, mi_df, vars_to_check)

    print(f"\nResultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()