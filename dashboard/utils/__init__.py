"""
Utilidades para el dashboard PoseLift.
"""

from .data_loader import (
    load_json_alphaposes,
    load_tabular_data,
    get_available_videos,
    get_available_persons,
    get_frame_count,
    get_keypoints_for_person,
    get_person_summary_stats,
    extract_keypoint_columns,
    load_pickle_data
)
from .keypoints import (
    normalize_keypoints_by_bbox,
    get_keypoint_color,
    get_body_group_indices,
    get_keypoint_name,
    calculate_distances,
    calculate_velocities,
    extract_features_per_frame,
    get_valid_keypoints_mask
)
from .stats_utils import (
    check_normality_shapiro,
    detect_outliers_iqr,
    calculate_descriptive_stats,
    calculate_mann_whitney_effect_size,
    calculate_spearman_correlation,
    create_summary_dataframe
)
from .display_utils import (
    show_image,
    show_csv,
    show_text_file,
    check_path_exists,
    list_directory_files
)

__all__ = [
    # data_loader
    'load_json_alphaposes',
    'load_tabular_data',
    'get_available_videos',
    'get_available_persons',
    'get_frame_count',
    'get_keypoints_for_person',
    'get_person_summary_stats',
    'extract_keypoint_columns',
    'load_pickle_data',
    # keypoints
    'normalize_keypoints_by_bbox',
    'get_keypoint_color',
    'get_body_group_indices',
    'get_keypoint_name',
    'calculate_distances',
    'calculate_velocities',
    'extract_features_per_frame',
    'get_valid_keypoints_mask',
    # stats_utils
    'check_normality_shapiro',
    'detect_outliers_iqr',
    'calculate_descriptive_stats',
    'calculate_mann_whitney_effect_size',
    'calculate_spearman_correlation',
    'create_summary_dataframe',
    # display_utils
    'show_image',
    'show_csv',
    'show_text_file',
    'check_path_exists',
    'list_directory_files'
]

