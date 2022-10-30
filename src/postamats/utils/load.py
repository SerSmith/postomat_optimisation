"""Скрипты для загрузки файлов по Realtive path
"""
import os
from pathlib import Path

PATH_TO_ROOT = Path(__file__).parent.parent.parent.parent

def get_full_path_from_relative(relative_path: str) -> str:
    """Превращает относительный путь в абсолютный
    Нужна для передачи путей, которые хранятся в константах,
     как относительные в пандас и проч.

    Args:
        relative_path (str): относительный путь к файлу

    Returns:
        str: абсолютный путь
    """
    return os.path.join(PATH_TO_ROOT, relative_path)
