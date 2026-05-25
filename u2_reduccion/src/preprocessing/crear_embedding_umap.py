import os
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


MODULE_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = MODULE_ROOT / "data" / "processed" / "features_trayectorias.csv"
OUTPUT_PATH = MODULE_ROOT / "data" / "processed" / "embedding_umap.csv"

EXCLUDED_COLUMNS = {
    "video_id",
    "person_id",
    "label_sum",
    "porcentaje_shoplifting",
    "label_trayectoria",
    "frame_min",
    "frame_max",
}


def build_umap_embedding(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    # Evita fallos de cache/JIT de numba en algunos entornos locales al importar umap-learn.
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

    try:
        import numba

        def without_cache(decorator):
            def wrapper(*args, **kwargs):
                kwargs["cache"] = False
                return decorator(*args, **kwargs)

            return wrapper

        numba.jit = without_cache(numba.jit)
        numba.njit = without_cache(numba.njit)
        numba.vectorize = without_cache(numba.vectorize)
        numba.guvectorize = without_cache(numba.guvectorize)

        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "Falta instalar umap-learn. Ejecuta: pip install umap-learn"
        ) from exc

    df = pd.read_csv(input_path)
    numeric_columns = [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in EXCLUDED_COLUMNS
    ]
    if not numeric_columns:
        raise ValueError("No hay columnas numericas disponibles para UMAP.")

    values = df[numeric_columns]
    values = SimpleImputer(strategy="median").fit_transform(values)
    values = StandardScaler().fit_transform(values)

    embedding = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    ).fit_transform(values)

    result = df[
        [
            "video_id",
            "person_id",
            "label_trayectoria",
            "porcentaje_shoplifting",
            "num_frames",
            "pose_quality_predominante",
        ]
    ].copy()
    result.insert(2, "umap_x", embedding[:, 0])
    result.insert(3, "umap_y", embedding[:, 1])
    result = result.rename(columns={"pose_quality_predominante": "pose_quality"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    result = build_umap_embedding()
    print(f"Archivo generado: {OUTPUT_PATH}")
    print(f"Trayectorias embebidas: {len(result)}")
