"""
Configuración global del dashboard PoseLift.
Contiene rutas, constantes, y mapeos de keypoints.
"""

from pathlib import Path

# ==================== RUTAS (ROBUSTAS CON PATHLIB) ====================
# PROJECT_ROOT se define como la carpeta padre del directorio del dashboard
# Esto garantiza que funcione desde cualquier ubicación de ejecución
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Directorios de datos
JSON_DIR = PROJECT_ROOT / "Json_files" / "data" / "PoseLift" / "pose"
PICKLE_DIR = PROJECT_ROOT / "Pickle_files"
PREPROCESSING_OUTPUTS = PROJECT_ROOT / "pre_procesamiento" / "outputs"
PREPROCESSING_AED = PREPROCESSING_OUTPUTS / "evaluar_normalidad"

# Directorios de hipótesis
H1_DIR = PROJECT_ROOT / "hipotesis_1" / "outputs"
H2_DIR = PROJECT_ROOT / "hipotesis_2" / "outputs"
H2_ACF_DIR = H2_DIR / "autocorrelacion"
H3_DIR = PROJECT_ROOT / "hipotesis_3" / "outputs"

# Splits de entrenamiento
SPLIT_DIRS = {
    'train': JSON_DIR / 'train',
    'test': JSON_DIR / 'test'
}

# ==================== KEYPOINTS COCO17 ====================
# Orden: [nose, left_eye, right_eye, left_ear, right_ear,
# left_shoulder, right_shoulder, left_elbow, right_elbow,
# left_wrist, right_wrist, left_hip, right_hip,
# left_knee, right_knee, left_ankle, right_ankle]

COCO17_KEYPOINTS = [
 'nose',
 'left_eye', 'right_eye', 'left_ear', 'right_ear',
 'left_shoulder', 'right_shoulder',
 'left_elbow', 'right_elbow',
 'left_wrist', 'right_wrist',
 'left_hip', 'right_hip',
 'left_knee', 'right_knee',
 'left_ankle', 'right_ankle'
]

assert len(COCO17_KEYPOINTS) == 17, "COCO17 debe tener exactamente 17 keypoints"

# ==================== GRUPOS CORPORALES ====================
# Mapeo: grupo corporal → indices de keypoints en COCO17
BODY_GROUPS = {
 'Cabeza': [0, 1, 2, 3, 4], # nose, eyes, ears
 'Hombros': [5, 6], # shoulders
 'Codos': [7, 8], # elbows
 'Muñecas': [9, 10], # wrists
 'Caderas': [11, 12], # hips
 'Rodillas': [13, 14], # knees
 'Tobillos': [15, 16] # ankles
}

# ==================== CONEXIONES ESQUELETO ====================
# Pares de índices de keypoints que deben conectarse
SKELETON_EDGES = [
 # Cabeza
 (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
 # Brazos
 (5, 7), (7, 9), (6, 8), (8, 10),
 # Torso
 (5, 6), (5, 11), (6, 12), (11, 12),
 # Piernas
 (11, 13), (13, 15), (12, 14), (14, 16)
]

# ==================== PALETA DE COLORES ====================
# Colores RGB (0-255) para cada keypoint
KEYPOINT_COLORS = {
 0: (255, 0, 0), # nose - rojo
 1: (0, 255, 0), # left_eye - verde
 2: (0, 0, 255), # right_eye - azul
 3: (255, 255, 0), # left_ear - amarillo
 4: (255, 0, 255), # right_ear - magenta
 5: (0, 255, 255), # left_shoulder - cyan
 6: (128, 0, 128), # right_shoulder - púrpura
 7: (255, 165, 0), # left_elbow - naranja
 8: (165, 42, 42), # right_elbow - marrón
 9: (0, 128, 128), # left_wrist - teal
 10: (128, 128, 0), # right_wrist - olive
 11: (255, 192, 203), # left_hip - rosa
 12: (173, 255, 47), # right_hip - greenish
 13: (70, 130, 180), # left_knee - steel blue
 14: (240, 128, 128), # right_knee - light coral
 15: (144, 238, 144), # left_ankle - light green
 16: (100, 149, 237) # right_ankle - cornflower blue
}

# ==================== CONSTANTES DE ANÁLISIS ====================
# Test de normalidad
SHAPIRO_ALPHA = 0.05

# Detección de outliers (IQR method)
IQR_MULTIPLIER = 1.5

# Velocidad mínima para considerar movimiento (píxeles/frame)
MIN_VELOCITY_THRESHOLD = 0.5

# Confianza mínima para keypoints válidos
MIN_CONFIDENCE = 0.1

# ==================== ETIQUETAS ====================
LABEL_MAPPING = {
 0: 'Normal',
 1: 'Shoplifting'
}

LABEL_COLORS = {
 'Normal': '#2ECC71',
 'Shoplifting': '#E74C3C'
}

# ==================== STREAMLIT CONFIG ====================
PAGE_CONFIG = {
 'page_title': 'Dashboard PoseLift',
 'page_icon': '',
 'layout': 'wide',
 'initial_sidebar_state': 'expanded'
}
