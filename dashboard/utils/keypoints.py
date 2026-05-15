"""
Funciones utilitarias para manipulación de keypoints.
Normalización, cálculo de distancias, velocidades, etc.
"""

import numpy as np
from typing import Dict, List, Tuple
from config import COCO17_KEYPOINTS, KEYPOINT_COLORS, BODY_GROUPS, MIN_VELOCITY_THRESHOLD


def normalize_keypoints_by_bbox(
    keypoints: np.ndarray,
    bbox: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Normaliza keypoints por bounding box.
    
    Args:
        keypoints: Array de shape (N, 3) o (N, 2) con [x, y, conf] o [x, y]
        bbox: Tuple (x1, y1, x2, y2) de bounding box
    
    Returns:
        Array normalizado de mismo shape
    """
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    if bbox_width <= 0 or bbox_height <= 0:
        return keypoints.copy()
    
    keypoints_norm = keypoints.copy().astype(float)
    
    # Normalizar coordenadas X, Y
    keypoints_norm[:, 0] = (keypoints_norm[:, 0] - x1) / bbox_width
    keypoints_norm[:, 1] = (keypoints_norm[:, 1] - y1) / bbox_height
    
    # Clamp a [0, 1]
    keypoints_norm[:, :2] = np.clip(keypoints_norm[:, :2], 0, 1)
    
    return keypoints_norm


def get_keypoint_color(keypoint_idx: int) -> str:
    """
    Obtiene color hexadecimal para un keypoint.
    
    Args:
        keypoint_idx: Índice del keypoint (0-16)
    
    Returns:
        Color en formato hex (#RRGGBB)
    """
    if keypoint_idx not in KEYPOINT_COLORS:
        return '#808080'  # gray por defecto
    
    r, g, b = KEYPOINT_COLORS[keypoint_idx]
    return f'#{r:02x}{g:02x}{b:02x}'


def get_body_group_indices(group_name: str) -> List[int]:
    """
    Obtiene índices de keypoints para un grupo corporal.
    
    Args:
        group_name: Nombre del grupo ('Cabeza', 'Hombros', etc.)
    
    Returns:
        Lista de índices de keypoints
    """
    return BODY_GROUPS.get(group_name, [])


def get_keypoint_name(idx: int) -> str:
    """
    Obtiene el nombre de un keypoint por su índice.
    
    Args:
        idx: Índice del keypoint (0-16)
    
    Returns:
        Nombre del keypoint
    """
    if 0 <= idx < len(COCO17_KEYPOINTS):
        return COCO17_KEYPOINTS[idx]
    return f'Unknown_{idx}'


def calculate_distances(
    kpts_frame1: np.ndarray,
    kpts_frame2: np.ndarray,
    confidence_threshold: float = 0.1
) -> np.ndarray:
    """
    Calcula distancias euclideas entre keypoints en dos frames.
    
    Args:
        kpts_frame1: Array (N, 3) [x, y, conf] del frame 1
        kpts_frame2: Array (N, 3) [x, y, conf] del frame 2
        confidence_threshold: Confianza mínima para considerar keypoint válido
    
    Returns:
        Array (N,) con distancias euclideas (0 si confianza < threshold)
    """
    distances = np.zeros(len(kpts_frame1))
    
    for i in range(len(kpts_frame1)):
        conf1, conf2 = kpts_frame1[i, 2], kpts_frame2[i, 2]
        
        if conf1 >= confidence_threshold and conf2 >= confidence_threshold:
            dx = kpts_frame2[i, 0] - kpts_frame1[i, 0]
            dy = kpts_frame2[i, 1] - kpts_frame1[i, 1]
            distances[i] = np.sqrt(dx**2 + dy**2)
        else:
            distances[i] = 0
    
    return distances


def calculate_velocities(
    frames_keypoints: Dict[int, np.ndarray],
    confidence_threshold: float = 0.1
) -> Dict[int, np.ndarray]:
    """
    Calcula velocidades de keypoints a través de frames.
    
    Args:
        frames_keypoints: Dict {frame_id: keypoints_array (17, 3)}
        confidence_threshold: Confianza mínima
    
    Returns:
        Dict {frame_id: velocities_array (17,)}
        - frame_id=0 tendrá velocidad=0 (sin frame anterior)
    """
    velocities = {}
    sorted_frames = sorted(frames_keypoints.keys())
    
    velocities[sorted_frames[0]] = np.zeros(17)  # Primer frame sin velocidad
    
    for i in range(1, len(sorted_frames)):
        frame_prev = sorted_frames[i - 1]
        frame_curr = sorted_frames[i]
        
        kpts_prev = frames_keypoints[frame_prev]
        kpts_curr = frames_keypoints[frame_curr]
        
        # Distancias entre frames
        distances = calculate_distances(kpts_prev, kpts_curr, confidence_threshold)
        
        # Normalizar por frame delta (generalmente 1)
        frame_delta = frame_curr - frame_prev
        velocities[frame_curr] = distances / max(frame_delta, 1)
    
    return velocities


def extract_features_per_frame(
    frames_keypoints: Dict[int, np.ndarray],
    normalize: bool = False,
    bbox: Tuple[float, float, float, float] = None
) -> Dict[str, np.ndarray]:
    """
    Extrae features de keypoints agregados por frame.
    
    Args:
        frames_keypoints: Dict {frame_id: keypoints_array (17, 3)}
        normalize: Si aplicar normalización por bbox
        bbox: Bounding box para normalización (si normalize=True)
    
    Returns:
        Dict con features agregadas: {
            'mean_x': array(N_frames,),
            'mean_y': array(N_frames,),
            'mean_confidence': array(N_frames,),
            'velocity': array(N_frames,),
            ...
        }
    """
    sorted_frames = sorted(frames_keypoints.keys())
    n_frames = len(sorted_frames)
    
    features = {
        'frame_ids': np.array(sorted_frames),
        'mean_x': np.zeros(n_frames),
        'mean_y': np.zeros(n_frames),
        'mean_confidence': np.zeros(n_frames),
        'std_x': np.zeros(n_frames),
        'std_y': np.zeros(n_frames),
        'velocity': np.zeros(n_frames)
    }
    
    # Calcular velocidades
    velocities = calculate_velocities(frames_keypoints)
    
    for idx, frame_id in enumerate(sorted_frames):
        kpts = frames_keypoints[frame_id].copy()
        
        # Normalizar si es necesario
        if normalize and bbox is not None:
            kpts = normalize_keypoints_by_bbox(kpts, bbox)
        
        features['mean_x'][idx] = np.mean(kpts[:, 0])
        features['mean_y'][idx] = np.mean(kpts[:, 1])
        features['mean_confidence'][idx] = np.mean(kpts[:, 2])
        features['std_x'][idx] = np.std(kpts[:, 0])
        features['std_y'][idx] = np.std(kpts[:, 1])
        features['velocity'][idx] = np.mean(velocities[frame_id])
    
    return features


def get_valid_keypoints_mask(
    keypoints: np.ndarray,
    confidence_threshold: float = 0.1
) -> np.ndarray:
    """
    Obtiene máscara de keypoints válidos (confianza > threshold).
    
    Args:
        keypoints: Array (N, 3) [x, y, conf]
        confidence_threshold: Confianza mínima
    
    Returns:
        Array boolean (N,)
    """
    return keypoints[:, 2] >= confidence_threshold
