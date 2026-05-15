from pathlib import Path
import pickle
import numpy as np
import pandas as pd


# ============================================================
# 1. Rutas
# ============================================================

BASE_DIR = Path("..").resolve()

PKL_TRAIN = BASE_DIR / "Pickle_files" / "Train"
PKL_TEST = BASE_DIR / "Pickle_files" / "Test"
PKL_GT = BASE_DIR / "Pickle_files" / "GT"

OUTPUT_DIR = Path(".").resolve() / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_CSV = OUTPUT_DIR / "train_tabular.csv"
TEST_CSV = OUTPUT_DIR / "test_tabular.csv"
TRAIN_SUMMARY_CSV = OUTPUT_DIR / "train_summary.csv"
TEST_SUMMARY_CSV = OUTPUT_DIR / "test_summary.csv"


# ============================================================
# 2. Nombres COCO17
# ============================================================

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

keypoint_columns = []
for kp_name in COCO17_KEYPOINT_NAMES:
    keypoint_columns.extend([
        f"{kp_name}_x",
        f"{kp_name}_y",
        f"{kp_name}_conf",
    ])

base_columns = [
    "video_id",
    "frame_id",
    "person_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
]

train_output_columns = base_columns + keypoint_columns
test_output_columns = base_columns + keypoint_columns + ["label"]


# ============================================================
# 3. Utilidades
# ============================================================

def listar_archivos(ruta: Path, extension: str):
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la ruta: {ruta}")
    return sorted(ruta.glob(f"*{extension}"))


def cargar_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def cargar_npy(path: Path):
    return np.load(path, allow_pickle=True)


def ordenar_frame_ids(frame_ids):
    """
    Ordena los frame_id respetando su valor numérico cuando sea posible.
    """
    try:
        return sorted(frame_ids, key=lambda x: int(x))
    except Exception:
        return sorted(frame_ids)


def validar_person_info(person_info, pkl_file: Path, frame_id, person_id):
    """
    Verifica que cada persona tenga la estructura esperada:
    [bbox, keypoints]
    """
    if not isinstance(person_info, (list, tuple)):
        raise ValueError(
            f"Estructura inválida en {pkl_file.name}, frame {frame_id}, person {person_id}: "
            f"person_info no es lista/tupla."
        )

    if len(person_info) < 2:
        raise ValueError(
            f"Estructura inválida en {pkl_file.name}, frame {frame_id}, person {person_id}: "
            f"person_info tiene menos de 2 elementos."
        )

    bbox = person_info[0]
    keypoints = np.asarray(person_info[1])

    if len(bbox) != 4:
        raise ValueError(
            f"BBox inválido en {pkl_file.name}, frame {frame_id}, person {person_id}: "
            f"se esperaban 4 valores, se obtuvo {len(bbox)}."
        )

    if keypoints.shape != (17, 3):
        raise ValueError(
            f"Keypoints inválidos en {pkl_file.name}, frame {frame_id}, person {person_id}: "
            f"se esperaba shape (17, 3), se obtuvo {keypoints.shape}."
        )

    return bbox, keypoints


def construir_fila_base(video_id, frame_id, person_id, bbox):
    return {
        "video_id": video_id,
        "frame_id": frame_id,
        "person_id": person_id,
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
    }


def agregar_keypoints_a_fila(row_data, keypoints):
    """
    Separa los 17 keypoints COCO17 en columnas:
    keypoint_x, keypoint_y, keypoint_conf.
    No modifica los valores.
    """
    for i, kp_name in enumerate(COCO17_KEYPOINT_NAMES):
        row_data[f"{kp_name}_x"] = keypoints[i, 0]
        row_data[f"{kp_name}_y"] = keypoints[i, 1]
        row_data[f"{kp_name}_conf"] = keypoints[i, 2]
    return row_data


# ============================================================
# 4. Procesamiento Train sin label
# ============================================================

def process_pkl_for_train(pkl_file_path: Path):
    video_id = pkl_file_path.stem
    pkl_data = cargar_pkl(pkl_file_path)

    rows = []
    total_frames = len(pkl_data)
    frames_with_persons = 0
    empty_frames = 0

    for frame_id in ordenar_frame_ids(pkl_data.keys()):
        frame_data = pkl_data[frame_id]

        if isinstance(frame_data, dict) and len(frame_data) > 0:
            frames_with_persons += 1

            for person_id, person_info in frame_data.items():
                bbox, keypoints = validar_person_info(
                    person_info,
                    pkl_file_path,
                    frame_id,
                    person_id
                )

                row_data = construir_fila_base(video_id, frame_id, person_id, bbox)
                row_data = agregar_keypoints_a_fila(row_data, keypoints)
                rows.append(row_data)

        else:
            empty_frames += 1

    summary = {
        "video_id": video_id,
        "total_frames": total_frames,
        "frames_with_persons": frames_with_persons,
        "empty_frames": empty_frames,
        "rows_generated": len(rows),
    }

    return rows, summary


# ============================================================
# 5. Procesamiento Test con label desde .npy
# ============================================================

def process_pkl_for_test(pkl_file_path: Path, npy_file_path: Path):
    video_id = pkl_file_path.stem
    pkl_data = cargar_pkl(pkl_file_path)
    labels = cargar_npy(npy_file_path)

    rows = []

    total_frames_pkl = len(pkl_data)
    total_frames_npy = len(labels)

    frame_ids_ordenados = ordenar_frame_ids(pkl_data.keys())

    if len(frame_ids_ordenados) > 0:
        max_frame_id_pkl = max(int(fid) for fid in frame_ids_ordenados)
    else:
        max_frame_id_pkl = -1

    if max_frame_id_pkl >= len(labels):
        raise IndexError(
            f"El PKL contiene un frame_id fuera del rango del NPY para {video_id}: "
            f"max_frame_id_pkl={max_frame_id_pkl}, len(labels)={len(labels)}"
        )

    frames_with_persons = 0
    empty_frames = 0

    label_0_frames_pkl = 0
    label_1_frames_pkl = 0
    other_label_frames_pkl = 0

    for frame_id in frame_ids_ordenados:
        frame_data = pkl_data[frame_id]

        # La etiqueta esta a nivel de frame.
        # Se respeta la alineacion original por frame_id.
        frame_index = int(frame_id)
        label = labels[frame_index]

        if label == 0:
            label_0_frames_pkl += 1
        elif label == 1:
            label_1_frames_pkl += 1
        else:
            other_label_frames_pkl += 1

        if isinstance(frame_data, dict) and len(frame_data) > 0:
            frames_with_persons += 1

            for person_id, person_info in frame_data.items():
                bbox, keypoints = validar_person_info(
                    person_info,
                    pkl_file_path,
                    frame_id,
                    person_id
                )

                row_data = construir_fila_base(video_id, frame_id, person_id, bbox)
                row_data = agregar_keypoints_a_fila(row_data, keypoints)

                # La etiqueta se replica para cada persona detectada en el mismo frame.
                # No significa que la etiqueta sea individual por person_id.
                row_data["label"] = label

                rows.append(row_data)

        else:
            empty_frames += 1

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_distribution_npy = dict(zip(unique_labels.tolist(), label_counts.tolist()))

    summary = {
        "video_id": video_id,
        "total_frames_pkl": total_frames_pkl,
        "total_frames_npy": total_frames_npy,
        "frame_count_difference_npy_minus_pkl": total_frames_npy - total_frames_pkl,
        "max_frame_id_pkl": max_frame_id_pkl,
        "frames_with_persons": frames_with_persons,
        "empty_frames": empty_frames,
        "rows_generated": len(rows),
        "label_0_frames_in_pkl": label_0_frames_pkl,
        "label_1_frames_in_pkl": label_1_frames_pkl,
        "other_label_frames_in_pkl": other_label_frames_pkl,
        "label_0_frames_in_npy": label_distribution_npy.get(0, 0),
        "label_1_frames_in_npy": label_distribution_npy.get(1, 0),
    }

    return rows, summary

# ============================================================
# 6. Construccion de datasets
# ============================================================

def construir_train():
    pkl_train_files = listar_archivos(PKL_TRAIN, ".pkl")

    all_rows = []
    summaries = []

    print("Procesando TRAIN...")
    print(f"Ruta: {PKL_TRAIN}")
    print(f"Archivos .pkl encontrados: {len(pkl_train_files)}")

    for i, pkl_file in enumerate(pkl_train_files, start=1):
        rows, summary = process_pkl_for_train(pkl_file)
        all_rows.extend(rows)
        summaries.append(summary)

        if i % 10 == 0 or i == len(pkl_train_files):
            print(f"  Procesados {i}/{len(pkl_train_files)} archivos")

    train_df = pd.DataFrame(all_rows, columns=train_output_columns)
    train_summary_df = pd.DataFrame(summaries)

    train_df.to_csv(TRAIN_CSV, index=False)
    train_summary_df.to_csv(TRAIN_SUMMARY_CSV, index=False)

    return train_df, train_summary_df


def construir_test():
    pkl_test_files = listar_archivos(PKL_TEST, ".pkl")
    npy_files = listar_archivos(PKL_GT, ".npy")

    npy_by_stem = {path.stem: path for path in npy_files}

    all_rows = []
    summaries = []

    print("\nProcesando TEST...")
    print(f"Ruta PKL: {PKL_TEST}")
    print(f"Ruta GT: {PKL_GT}")
    print(f"Archivos .pkl encontrados: {len(pkl_test_files)}")
    print(f"Archivos .npy encontrados: {len(npy_files)}")

    for i, pkl_file in enumerate(pkl_test_files, start=1):
        video_id = pkl_file.stem

        if video_id not in npy_by_stem:
            raise FileNotFoundError(
                f"No se encontro archivo .npy para {pkl_file.name}. "
                f"Se esperaba: {video_id}.npy"
            )

        npy_file = npy_by_stem[video_id]

        rows, summary = process_pkl_for_test(pkl_file, npy_file)
        all_rows.extend(rows)
        summaries.append(summary)

        if i % 10 == 0 or i == len(pkl_test_files):
            print(f"  Procesados {i}/{len(pkl_test_files)} archivos")

    test_df = pd.DataFrame(all_rows, columns=test_output_columns)
    test_summary_df = pd.DataFrame(summaries)

    test_df.to_csv(TEST_CSV, index=False)
    test_summary_df.to_csv(TEST_SUMMARY_CSV, index=False)

    return test_df, test_summary_df


# ============================================================
# 7. Reporte de validacion
# ============================================================

def imprimir_resumen(train_df, train_summary_df, test_df, test_summary_df):
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)

    print("\nTRAIN")
    print("-" * 80)
    print(f"CSV guardado en: {TRAIN_CSV}")
    print(f"Resumen guardado en: {TRAIN_SUMMARY_CSV}")
    print(f"Shape train: {train_df.shape}")
    print(f"Columnas train: {len(train_df.columns)}")
    print(f"Tiene label?: {'label' in train_df.columns}")
    print(f"Archivos procesados: {len(train_summary_df)}")
    print(f"Total frames: {train_summary_df['total_frames'].sum()}")
    print(f"Frames con personas: {train_summary_df['frames_with_persons'].sum()}")
    print(f"Frames vacios: {train_summary_df['empty_frames'].sum()}")
    print(f"Filas generadas: {train_summary_df['rows_generated'].sum()}")

    print("\nPrimeras filas TRAIN:")
    print(train_df.head())

    print("\nTEST")
    print("-" * 80)
    print(f"CSV guardado en: {TEST_CSV}")
    print(f"Resumen guardado en: {TEST_SUMMARY_CSV}")
    print(f"Shape test: {test_df.shape}")
    print(f"Columnas test: {len(test_df.columns)}")
    print(f"Tiene label?: {'label' in test_df.columns}")
    print(f"Archivos procesados: {len(test_summary_df)}")
    print(f"Total frames en PKL: {test_summary_df['total_frames_pkl'].sum()}")
    print(f"Total frames en NPY: {test_summary_df['total_frames_npy'].sum()}")
    print(f"Diferencia total NPY - PKL: {test_summary_df['frame_count_difference_npy_minus_pkl'].sum()}")
    print(f"Frames con personas: {test_summary_df['frames_with_persons'].sum()}")
    print(f"Frames vacios en PKL: {test_summary_df['empty_frames'].sum()}")
    print(f"Filas generadas: {test_summary_df['rows_generated'].sum()}")

    print("\nDistribucion de labels considerando solo frames presentes en PKL:")
    print(f"Label 0 frames en PKL: {test_summary_df['label_0_frames_in_pkl'].sum()}")
    print(f"Label 1 frames en PKL: {test_summary_df['label_1_frames_in_pkl'].sum()}")
    print(f"Otros labels en PKL: {test_summary_df['other_label_frames_in_pkl'].sum()}")

    print("\nDistribucion de labels considerando todo el NPY:")
    print(f"Label 0 frames en NPY: {test_summary_df['label_0_frames_in_npy'].sum()}")
    print(f"Label 1 frames en NPY: {test_summary_df['label_1_frames_in_npy'].sum()}")

    print("\nDistribucion de label por filas en TEST:")
    print(test_df["label"].value_counts(dropna=False).sort_index())

    print("\nPrimeras filas TEST:")
    print(test_df.head())

    print("\nValidacion de columnas esperadas")
    print("-" * 80)
    print("Train esperado: 58 columnas = 3 identificadores + 4 bbox + 51 keypoints")
    print("Test esperado: 59 columnas = 3 identificadores + 4 bbox + 51 keypoints + label")
    print(f"Train correcto?: {train_df.shape[1] == 58}")
    print(f"Test correcto?: {test_df.shape[1] == 59}")


# ============================================================
# 8. Main
# ============================================================

if __name__ == "__main__":
    print("BASE_DIR:", BASE_DIR)
    print("PKL_TRAIN existe:", PKL_TRAIN.exists())
    print("PKL_TEST existe:", PKL_TEST.exists())
    print("PKL_GT existe:", PKL_GT.exists())

    train_df, train_summary_df = construir_train()
    test_df, test_summary_df = construir_test()

    imprimir_resumen(train_df, train_summary_df, test_df, test_summary_df)