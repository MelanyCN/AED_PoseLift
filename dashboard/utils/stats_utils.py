"""
Funciones utilitarias para análisis estadístico.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List
from config import SHAPIRO_ALPHA, IQR_MULTIPLIER


def check_normality_shapiro(
    data: np.ndarray,
    alpha: float = SHAPIRO_ALPHA
) -> Dict[str, float]:
    """
    Aplica test de Shapiro-Wilk para normalidad.
    
    Args:
        data: Array 1D con datos
        alpha: Nivel de significancia
    
    Returns:
        Dict con {
            'statistic': valor del test,
            'p_value': p-value,
            'is_normal': boolean (p_value > alpha)
        }
    """
    # Eliminar NaN y infinitos
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) < 3:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'is_normal': False,
            'reason': 'Muestra muy pequeña'
        }
    
    statistic, p_value = stats.shapiro(data_clean)
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'is_normal': p_value > alpha,
        'alpha': alpha
    }


def detect_outliers_iqr(
    data: np.ndarray,
    multiplier: float = IQR_MULTIPLIER
) -> Dict[str, any]:
    """
    Detecta outliers usando método IQR (Interquartile Range).
    
    Args:
        data: Array 1D con datos
        multiplier: Multiplicador del IQR (1.5 estándar, 3.0 extremos)
    
    Returns:
        Dict con {
            'indices': indices de outliers,
            'values': valores de outliers,
            'lower_bound': límite inferior,
            'upper_bound': límite superior,
            'count': número de outliers
        }
    """
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) < 2:
        return {
            'indices': np.array([], dtype=int),
            'values': np.array([]),
            'lower_bound': np.nan,
            'upper_bound': np.nan,
            'count': 0
        }
    
    Q1 = np.percentile(data_clean, 25)
    Q3 = np.percentile(data_clean, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Encontrar outliers en data original (con NaN)
    mask = (data < lower_bound) | (data > upper_bound) | np.isnan(data)
    indices = np.where(mask)[0]
    
    return {
        'indices': indices,
        'values': data[indices],
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'count': len(indices),
        'multiplier': multiplier
    }


def calculate_descriptive_stats(
    data: np.ndarray,
    name: str = 'Data'
) -> Dict[str, float]:
    """
    Calcula estadísticas descriptivas completas.
    
    Args:
        data: Array 1D con datos
        name: Nombre de la variable (para output)
    
    Returns:
        Dict con estadísticas: mean, median, std, var, skew, kurtosis, min, max, q25, q75
    """
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) == 0:
        return {
            'variable': name,
            'count': 0,
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'var': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'min': np.nan,
            'max': np.nan,
            'q25': np.nan,
            'q75': np.nan,
            'range': np.nan
        }
    
    return {
        'variable': name,
        'count': len(data_clean),
        'mean': float(np.mean(data_clean)),
        'median': float(np.median(data_clean)),
        'std': float(np.std(data_clean, ddof=1)),
        'var': float(np.var(data_clean, ddof=1)),
        'skewness': float(stats.skew(data_clean)),
        'kurtosis': float(stats.kurtosis(data_clean)),
        'min': float(np.min(data_clean)),
        'max': float(np.max(data_clean)),
        'q25': float(np.percentile(data_clean, 25)),
        'q75': float(np.percentile(data_clean, 75)),
        'range': float(np.max(data_clean) - np.min(data_clean))
    }


def calculate_mann_whitney_effect_size(
    group1: np.ndarray,
    group2: np.ndarray
) -> Dict[str, float]:
    """
    Calcula test Mann-Whitney U y tamaño del efecto (r).
    
    Args:
        group1: Array 1D grupo 1
        group2: Array 1D grupo 2
    
    Returns:
        Dict con {
            'statistic': valor U,
            'p_value': p-value,
            'effect_size_r': tamaño del efecto (r)
        }
    """
    g1_clean = group1[np.isfinite(group1)]
    g2_clean = group2[np.isfinite(group2)]
    
    if len(g1_clean) == 0 or len(g2_clean) == 0:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'effect_size_r': np.nan
        }
    
    statistic, p_value = stats.mannwhitneyu(g1_clean, g2_clean, alternative='two-sided')
    
    # Tamaño del efecto r = Z / sqrt(N)
    n_total = len(g1_clean) + len(g2_clean)
    z_score = stats.norm.ppf(1 - p_value / 2)  # Aproximación
    effect_size_r = abs(z_score) / np.sqrt(n_total)
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'effect_size_r': float(effect_size_r)
    }


def calculate_spearman_correlation(
    x: np.ndarray,
    y: np.ndarray
) -> Dict[str, float]:
    """
    Calcula correlación de Spearman (no paramétrica).
    
    Args:
        x: Array 1D
        y: Array 1D
    
    Returns:
        Dict con {
            'correlation': coeficiente,
            'p_value': p-value
        }
    """
    x_clean = x[np.isfinite(x)]
    y_clean = y[np.isfinite(y)]
    
    if len(x_clean) < 2 or len(y_clean) < 2 or len(x_clean) != len(y_clean):
        return {
            'correlation': np.nan,
            'p_value': np.nan
        }
    
    corr, p_value = stats.spearmanr(x_clean, y_clean)
    
    return {
        'correlation': float(corr),
        'p_value': float(p_value)
    }


def create_summary_dataframe(
    stats_list: List[Dict[str, float]]
) -> pd.DataFrame:
    """
    Crea DataFrame a partir de lista de diccionarios de estadísticas.
    
    Args:
        stats_list: Lista de dicts retornados por calculate_descriptive_stats
    
    Returns:
        DataFrame con estadísticas
    """
    df = pd.DataFrame(stats_list)
    
    # Ordenar columnas de forma lógica
    col_order = [
        'variable', 'count', 'mean', 'median', 'std', 'min', 'q25', 'q75', 'max',
        'var', 'range', 'skewness', 'kurtosis'
    ]
    
    available_cols = [c for c in col_order if c in df.columns]
    df = df[available_cols]
    
    return df
