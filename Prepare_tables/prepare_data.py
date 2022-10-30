import os
import re
from warnings import warn
import hashlib
from typing import Optional, List
from shapely.geometry import Polygon
import numpy as np
import pandas as pd

OBJECT_ID_COL = 'OBJECT_ID'


def codes_to_str(codes_series: pd.Series) -> pd.Series:
    return codes_series.fillna('').astype(str).apply(lambda x: x.split('.')[0])


def get_filenames_from_dir(path: str,
                          mandatory_substr: str='',
                          include_subdirs: bool=True,
                          file_extensions: Optional[List[str]]=None):
    """Получает пути к файлам с данными из директории и поддиректорий

    Args:
        path (str): путь к корневому каталогу с файлами
        mandatory_substr (str): обязательная подстрока, которая должна содержаться в имени файла
        include_subdirs (bool, optional):смотреть ли в поддиректориях. Defaults to True.
        file_extensions (str or [str], optional): какие расширения нас интересуют. Defaults to None.

    Returns:
        [str]: список путей к файлам
    """
    files_list = []

    if include_subdirs:
        for root, _, files in os.walk(path):
            files_list += [os.path.join(root, f) for f in files
                          if os.path.isfile(os.path.join(root, f))]
    else:
        files_list = [os.path.join(path, f) for f in os.listdir(path)
                    if os.path.isfile(os.path.join(path, f))]

    if file_extensions is not None:
        files_list = [f for f in files_list if f.split('.')[-1] in file_extensions]

    files_list = [f for f in files_list if mandatory_substr in f]

    return files_list


def get_text_hash(text: str) -> str:
    """Получает хэш текста, нужна для однозначного индексирования данных
    результата get_data_for_match в т.ч. при распределенных вычислениях

    Args:
        text (str): текст

    Returns:
        str: хэш
    """
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()


def get_row_hashes(data: pd.DataFrame) -> pd.Series:
    """Получает хэш строки датафрейма
    нужен для создания айдишника объекта

    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    row = data.astype(str).sum(axis=1)
    return row.apply(get_text_hash)


def create_geolist(geo_str):
    geo_list = re.sub(r'[^0-9,.]', '', str(geo_str)).split(',')
    if geo_list != []:
        geo_list = [[float(geo_list[i]), float(geo_list[i+1])] for i in range(0, len(geo_list)-1, 2)]
    return geo_list


def prepare_geo(geodata: str) -> str:
    geodata = re.sub(r'[^\[\]0-9,.]', '', str(geodata))
    return geodata.strip(',')


def prepare_kad_num(kad_num: str) -> str:
    kad_num = re.sub(r'[^0-9\:]', '', str(kad_num))
    if (kad_num!='' and kad_num[0]==':'):
        return kad_num[1:]
    return kad_num


def calc_polygon_centroid(coords: List[List[float]]) -> List[float]:
    # Пример coords = ((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.))
    if not isinstance(coords, list):
        warn('coords is not list')
        return np.nan
    if coords==[]:
        warn('coords is empty')
        return np.nan
    if len(coords)==1:
        warn('coords is dot')
        return coords[0]
    if len(coords)==2:
        warn('coords is line')
        x1 = coords[0][0]
        y1 = coords[0][1]
        x2 = coords[1][0]
        y2 = coords[1][1]
        return [(x1+x2)/2, (y1+y2)/2]

    plgn = Polygon(coords)
    return list(plgn.centroid.coords)[0]
