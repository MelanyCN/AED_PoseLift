"""
Data loaders para JSON, CSV y pickle files del dataset PoseLift.
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from config import (
    JSON_DIR, PICKLE_DIR, PREPROCESSING_OUTPUTS,
    SPLIT_DIRS, COCO17_KEYPOINTS
)


@st.cache_data
def load_json_alphaposes(
    split: str = 'test',
    video_id: str = None
) -> Dict:
    """
    Carga archivo JSON AlphaPose para un video específico.
    
    Args:
        split: 'train' o 'test'
        video_id: ID del video (ej: '01_0222')
    
    Returns:
        Dict {person_id: {frame_id: {'keypoints': array}}}
    """
    if split not in SPLIT_DIRS:
        raise ValueError(f"Split debe ser 'train' o 'test', got {split}")
    
    if video_id is None:
        # Cargar primer video disponible
        video_files = list(SPLIT_DIRS[split].glob('*_alphapose_tracked_person.json'))
        if not video_files:
            raise FileNotFoundError(f"No JSON files found in {SPLIT_DIRS[split]}")
        json_path = video_files[0]
    else:
        # Buscar archivo con video_id
        pattern = f"{video_id}*_alphapose_tracked_person.json"
        matching_files = list(SPLIT_DIRS[split].glob(pattern))
        
        if not matching_files:
            raise FileNotFoundError(
                f"No JSON file found matching pattern {pattern} in {SPLIT_DIRS[split]}"
            )
        json_path = matching_files[0]
    
    # Cargar JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convertir keypoints strings a arrays
    converted_data = {}
    for person_id_str, frames_dict in data.items():
        converted_data[person_id_str] = {}
        for frame_id_str, frame_data in frames_dict.items():
            kpts = frame_data.get('keypoints', [])
            kpts_array = np.array(kpts, dtype=np.float32)
            
            # Si los keypoints son flat (51 elementos = 17*3), reshape a (17, 3)
            if len(kpts_array) == 51:
                kpts_array = kpts_array.reshape(17, 3)
            elif len(kpts_array) != 17 and len(kpts_array) > 0:
                # Si no es 51 ni 17, asumir que ya está en el formato correcto
                pass
            
            converted_data[person_id_str][frame_id_str] = {
                'keypoints': kpts_array
            }
    
    return converted_data


@st.cache_data
def load_tabular_data(
    split: str = 'test',
    csv_type: str = 'raw'
) -> pd.DataFrame:
    """
    Carga datos tabulares (CSV) del preprocessamiento.
    
    Args:
        split: 'train' o 'test'
        csv_type: 'raw' (tabular.csv) o 'normalized' (normalizado.csv)
    
    Returns:
        DataFrame con datos
    """
    if split == 'test':
        if csv_type == 'raw':
            csv_path = PREPROCESSING_OUTPUTS / 'test_tabular.csv'
        else:
            csv_path = PREPROCESSING_OUTPUTS / 'test_tabular_h1_normalizado.csv'
    elif split == 'train':
        if csv_type == 'raw':
            csv_path = PREPROCESSING_OUTPUTS / 'train_tabular.csv'
        else:
            # Asumir que existe versión normalizada
            csv_path = PREPROCESSING_OUTPUTS / 'train_tabular.csv'
    else:
        raise ValueError(f"Split debe ser 'train' o 'test', got {split}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


@st.cache_data
def get_available_videos(split: str = 'test') -> List[str]:
    """
    Obtiene lista de videos disponibles en un split.
    
    Args:
        split: 'train' o 'test'
    
    Returns:
        Lista de video_ids (ej: ['01_0222', '02_0310', ...])
    """
    if split not in SPLIT_DIRS:
        raise ValueError(f"Split debe ser 'train' o 'test', got {split}")
    
    json_files = list(SPLIT_DIRS[split].glob('*_alphapose_tracked_person.json'))
    
    # Extraer video_id del nombre del archivo
    video_ids = []
    for json_file in json_files:
        filename = json_file.stem  # sin .json
        # Formato esperado: {numero}_{numero}_alphapose_tracked_person
        parts = filename.split('_')
        if len(parts) >= 2:
            video_id = f"{parts[0]}_{parts[1]}"
            video_ids.append(video_id)
    
    return sorted(list(set(video_ids)))


@st.cache_data
def get_available_persons(
    split: str = 'test',
    video_id: str = None
) -> List[int]:
    """
    Obtiene lista de persons disponibles en un video.
    
    Args:
        split: 'train' o 'test'
        video_id: ID del video
    
    Returns:
        Lista de person_ids
    """
    data = load_json_alphaposes(split, video_id)
    person_ids = sorted([int(pid) for pid in data.keys()])
    return person_ids


@st.cache_data
def get_frame_count(
    split: str = 'test',
    video_id: str = None,
    person_id: int = None
) -> int:
    """
    Obtiene número de frames para una persona en un video.
    
    Args:
        split: 'train' o 'test'
        video_id: ID del video
        person_id: ID de la persona
    
    Returns:
        Número de frames
    """
    data = load_json_alphaposes(split, video_id)
    
    if person_id is None:
        person_id = list(data.keys())[0]
    else:
        person_id = str(person_id)
    
    if person_id not in data:
        return 0
    
    return len(data[person_id])


def get_keypoints_for_person(
    split: str = 'test',
    video_id: str = None,
    person_id: int = None,
    frame_range: Tuple[int, int] = None
) -> Dict[int, np.ndarray]:
    """
    Obtiene keypoints para una persona en un rango de frames.
    
    Args:
        split: 'train' o 'test'
        video_id: ID del video
        person_id: ID de la persona
        frame_range: Tupla (start_frame, end_frame) inclusiva
    
    Returns:
        Dict {frame_id: keypoints_array (17, 3)}
    """
    data = load_json_alphaposes(split, video_id)
    
    if person_id is None:
        person_id = list(data.keys())[0]
    else:
        person_id = str(person_id)
    
    if person_id not in data:
        return {}
    
    person_frames = data[person_id]
    
    # Convertir frame_ids a integers
    frame_ids = sorted([int(fid) for fid in person_frames.keys()])
    
    if frame_range is None:
        frame_range = (frame_ids[0], frame_ids[-1])
    
    start_frame, end_frame = frame_range
    
    result = {}
    for fid in frame_ids:
        if start_frame <= fid <= end_frame:
            kpts = person_frames[str(fid)]['keypoints']
            # Verificar que sea COCO17 (shape (17, 3))
            if hasattr(kpts, 'shape') and len(kpts.shape) == 2 and kpts.shape[0] == 17:
                result[fid] = kpts
            elif len(kpts) == 17:
                result[fid] = kpts
    
    return result


@st.cache_data
def load_pickle_data(file_path: str) -> np.ndarray:
    """
    Carga archivo .npy (numpy pickle).
    
    Args:
        file_path: Ruta al archivo .npy
    
    Returns:
        Array numpy
    """
    full_path = PICKLE_DIR / file_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {full_path}")
    
    return np.load(full_path)


def get_person_summary_stats(
    split: str = 'test',
    video_id: str = None,
    person_id: int = None
) -> Dict:
    """
    Obtiene estadísticas resumen para una persona.
    
    Args:
        split: 'train' o 'test'
        video_id: ID del video
        person_id: ID de la persona
    
    Returns:
        Dict con estadísticas: frame_count, bbox_mean, etc.
    """
    kpts_dict = get_keypoints_for_person(split, video_id, person_id)
    
    if not kpts_dict:
        return {'error': 'No keypoints found'}
    
    # Agrupar todos los keypoints
    all_kpts = np.vstack([kpts for kpts in kpts_dict.values()])
    
    return {
        'frame_count': len(kpts_dict),
        'mean_x': float(np.mean(all_kpts[:, 0])),
        'mean_y': float(np.mean(all_kpts[:, 1])),
        'mean_confidence': float(np.mean(all_kpts[:, 2])),
        'std_x': float(np.std(all_kpts[:, 0])),
        'std_y': float(np.std(all_kpts[:, 1])),
        'std_confidence': float(np.std(all_kpts[:, 2])),
        'min_confidence': float(np.min(all_kpts[:, 2])),
        'max_confidence': float(np.max(all_kpts[:, 2]))
    }


def extract_keypoint_columns(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Extrae columnas de keypoints de un DataFrame tabular.
    Espera columnas con formato: {keypoint_name}_{x|y|conf}
    
    Args:
        df: DataFrame con datos tabulares
    
    Returns:
        Dict {keypoint_name: Series de valores X/Y/conf concatenados}
    """
    keypoint_data = {}
    
    for kpt_name in COCO17_KEYPOINTS:
        x_col = f'{kpt_name}_x'
        y_col = f'{kpt_name}_y'
        conf_col = f'{kpt_name}_conf'
        
        if x_col in df.columns and y_col in df.columns:
            keypoint_data[kpt_name] = {
                'x': df[x_col] if x_col in df.columns else None,
                'y': df[y_col] if y_col in df.columns else None,
                'conf': df[conf_col] if conf_col in df.columns else None
            }
    
    return keypoint_data
