"""Функции для подготовки сырых данных
для загрузки в базу данных
"""

import os
import re
from warnings import warn
import hashlib
from typing import Optional, List
from shapely.geometry import Polygon
import numpy as np
import pandas as pd

OBJECT_ID_COL = 'OBJECT_ID'
# куда складывать промежуточные данные при подготовке табличек
# содержимое папки добавлено в гитигнор, данные будут там появлять локально пи выполнении скриптов
PREPARED_DATA_PATH = 'Notebooks/prepare_data/data'

# названия промежуточных табличек с обработанными данными из
# https://data.mos.ru/opendata/60562/data/table?versionNumber=3&releaseNumber=823
# https://dom.gosuslugi.ru/#!/houses
# которые используются для подготовки финальных табличек для заливки в базу
PREPARED_GIS_FILE = 'prepare_gis_houses_data.pickle'
PREPARED_DMR_FILE = 'prepare_dmr_houses_data.pickle'

# как мы назовем колонку с координатами центроида полигона
GEODATA_CENTER_COL = 'GEODATA_CENTER'


def codes_to_str(codes_series: pd.Series) -> pd.Series:
    """Коды при загрузке данных при наличии пропусков загружаются как float
    и в конце кода ставится ".0"
     Исправляем эту проблему: коды переводим в строковый формат, удаляем ".0"

    Args:
        codes_series (pd.Series): серия с кодами (ИНН, КПП и т.д.)

    Returns:
        pd.Series: серия с исправленными кодами (ИНН, КПП и т.д.)
    """
    return codes_series.fillna('').astype(str).apply(lambda x: x.split('.')[0])


def get_filenames_from_dir(path: str,
                          mandatory_substr: str='',
                          include_subdirs: bool=True,
                          file_extensions: Optional[List[str]]=None) -> List[str]:
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
    """Получает хэш текста, нужна для однозначного индексирования данных,
     используется в get_row_hashes

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


def create_geolist(geo_str: str) -> list:
    """Превращает кривое представление геоданных полигонов
    в лист с координатами [[x1, y1], [x2, y2], ...]
    Используется при обработке данных из json "Адресный реестр объектов недвижимости города Москвы"
    https://data.mos.ru/opendata/60562/data/table?versionNumber=3&releaseNumber=823

    Args:
        geo_str (str): _description_

    Returns:
        list: _description_
    """
    geo_list = re.sub(r'[^0-9,.]', '', str(geo_str)).split(',')
    if geo_list != []:
        geo_list = [[float(geo_list[i]), float(geo_list[i+1])]
                    for i in range(0, len(geo_list)-1, 2)]
    return geo_list


def prepare_kad_num(kad_num: str) -> str:
    """Превращает кривое представление кадастрового номера
    в стандартный вид
    Используется при обработке данных из json "Адресный реестр объектов недвижимости города Москвы"
    https://data.mos.ru/opendata/60562/data/table?versionNumber=3&releaseNumber=823

    Args:
        kad_num (str): _description_

    Returns:
        str: _description_
    """
    kad_num = re.sub(r'[^0-9\:]', '', str(kad_num))
    if (kad_num!='' and kad_num[0]==':'):
        return kad_num[1:]
    return kad_num


def calc_polygon_centroid(coords: List[List[float]]) -> List[float]:
    """Вычисляет координаты цетроида полигона
     В данных вместо полигонов встречаются точки и линии, функция их тоже обрабатывает

    Args:
        coords (List[List[float]]): координаты в формате
         [[x1, y1], [x2, y2], [x3, y3], ...]

    Returns:
        List[float]: координаты центроида в формате [x_c, y_c]
    """
    # для тестирования:
    # print(calc_polygon_centroid([[0,0], [1,0], [1,1], [0,1]]),
    #       'expect: [.5, .5]')
    # print(calc_polygon_centroid([[0,1], [0,-1], [2,0]]),
    #       'expect: [2/3, 0]')
    # print(calc_polygon_centroid([[0,-1], [0,1], [2,0]]),
    #       'expect: [2/3, 0]')
    # print(calc_polygon_centroid([[0,0], [1,0], [1.5,0.5], [1,1], [0,1], [-.5,.5]]),
    #       'expect: [.5, .5]')

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