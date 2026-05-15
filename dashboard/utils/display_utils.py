"""
Utilidades para mostrar imágenes, CSVs y otros contenidos en Streamlit
con manejo robusto de errores y paths.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Optional, Union


def show_image(
    path: Union[str, Path],
    caption: Optional[str] = None,
    width: str = "stretch",
    show_warning: bool = True
) -> bool:
    """
    Muestra una imagen si existe, caso contrario muestra un warning.
    
    Args:
        path: Ruta a la imagen (str o Path)
        caption: Caption opcional para la imagen
        width: Ancho ('stretch' o 'content')
        show_warning: Si True, muestra warning si no existe
    
    Returns:
        True si la imagen se mostró, False si no existe
    """
    path = Path(path)
    
    if path.exists() and path.is_file():
        try:
            img = Image.open(path)
            st.image(img, caption=caption, width=width)
            return True
        except Exception as e:
            if show_warning:
                st.error(f"Error cargando imagen: {path.name}\n{str(e)}")
            return False
    else:
        if show_warning:
            st.warning(
                f"**No se encontró la imagen esperada:**\n\n"
                f"Ruta: `{path}`\n\n"
                f"Asegúrate de ejecutar primero el script de análisis correspondiente."
            )
        return False


def show_csv(
    path: Union[str, Path],
    title: Optional[str] = None,
    show_download: bool = True,
    download_name: Optional[str] = None,
    show_warning: bool = True
) -> Optional[pd.DataFrame]:
    """
    Muestra un CSV como tabla si existe, caso contrario muestra warning.
    
    Args:
        path: Ruta al CSV (str o Path)
        title: Título opcional para mostrar
        show_download: Si True, muestra botón de descarga
        download_name: Nombre del archivo para descargar (default: nombre original)
        show_warning: Si True, muestra warning si no existe
    
    Returns:
        DataFrame si fue exitoso, None si no existe
    """
    path = Path(path)
    
    if path.exists() and path.is_file():
        try:
            df = pd.read_csv(path)
            
            if title:
                st.subheader(title)
            
            st.dataframe(df, use_container_width=True)
            
            if show_download:
                download_name = download_name or path.name
                st.download_button(
                    label=f"Descargar {download_name}",
                    data=df.to_csv(index=False),
                    file_name=download_name,
                    mime="text/csv"
                )
            
            return df
        except Exception as e:
            if show_warning:
                st.error(f"Error cargando CSV: {path.name}\n{str(e)}")
            return None
    else:
        if show_warning:
            st.warning(
                f"**No se encontró el archivo CSV esperado:**\n\n"
                f"Ruta: `{path}`\n\n"
                f"Asegúrate de ejecutar primero el script de análisis correspondiente."
            )
        return None


def show_text_file(
    path: Union[str, Path],
    title: Optional[str] = None,
    show_warning: bool = True
) -> Optional[str]:
    """
    Lee y muestra el contenido de un archivo de texto.
    
    Args:
        path: Ruta al archivo de texto
        title: Título opcional para mostrar
        show_warning: Si True, muestra warning si no existe
    
    Returns:
        Contenido del archivo si fue exitoso, None si no existe
    """
    path = Path(path)
    
    if path.exists() and path.is_file():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if title:
                st.markdown(f"### {title}")
            
            st.text(content)
            return content
        except Exception as e:
            if show_warning:
                st.error(f"Error leyendo archivo: {path.name}\n{str(e)}")
            return None
    else:
        if show_warning:
            st.warning(f"Archivo no encontrado: {path}")
        return None


def check_path_exists(path: Union[str, Path], raise_error: bool = False) -> bool:
    """
    Verifica si una ruta existe.
    
    Args:
        path: Ruta a verificar
        raise_error: Si True, lanza excepción si no existe
    
    Returns:
        True si existe, False si no
    """
    path = Path(path)
    exists = path.exists()
    
    if not exists and raise_error:
        raise FileNotFoundError(f"Ruta no encontrada: {path}")
    
    return exists


def list_directory_files(
    directory: Union[str, Path],
    pattern: str = "*",
    extensions: Optional[list] = None
) -> list:
    """
    Lista archivos en un directorio con filtros opcionales.
    
    Args:
        directory: Ruta del directorio
        pattern: Patrón glob para filtrar archivos
        extensions: Lista de extensiones a incluir (ej: ['.png', '.jpg'])
    
    Returns:
        Lista de Paths que coinciden con los criterios
    """
    directory = Path(directory)
    
    if not directory.exists() or not directory.is_dir():
        return []
    
    files = list(directory.glob(pattern))
    
    if extensions:
        files = [f for f in files if f.suffix.lower() in extensions]
    
    return sorted(files)
