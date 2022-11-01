"""Функции для подготовки сырых данных
для загрузки в базу данных
"""

import os
import re
from warnings import warn
from typing import Optional, List
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import sleep

import geopandas as gpd
from shapely.geometry import Polygon

from postamats.utils import helpers
from postamats.global_constants import NAN_VALUES, OBJECT_ID_COL, DMR_COLS_MAP,\
    DMR_GEODATA_COL, DMR_KAD_NUM_COLS, LATITUDE_COL, LONGITUDE_COL, ADM_AREA_TO_EXCLUDE,\
        GIS_COLS_MAP, OBJECT_ID_GIS_COL, OBJECT_TYPE_COL

tqdm.pandas()

# куда складывать промежуточные данные при подготовке табличек
# содержимое папки добавлено в гитигнор, данные будут там появлять локально пи выполнении скриптов
PREPARED_DATA_PATH = 'Notebooks/prepare_data/data'

# названия промежуточных табличек с обработанными данными из
# https://data.mos.ru/opendata/60562/data/table?versionNumber=3&releaseNumber=823
# https://dom.gosuslugi.ru/#!/houses
# которые используются для подготовки финальных табличек для заливки в базу
RAW_GIS_NAME = 'raw_gis_houses_data'
RAW_DMR_NAME = 'raw_dmr_houses_data'
# название таблички с данными жилых домов
# построенной на базе RAW_GIS_NAME и RAW_DMR_NAME
APARTMENT_HOUSES_NAME = 'apartment_houses_all_data'


def codes_to_str(codes_series: pd.Series) -> pd.Series:
    """Коды при загрузке данных при наличии пропусков загружаются как float
    и в конце кода ставится ".0"
     Исправляем эту проблему: коды переводим в строковый формат, удаляем ".0"

    Args:
        codes_series (pd.Series): серия с кодами (ИНН, КПП и т.д.)

    Returns:
        pd.Series: серия с исправленными кодами (ИНН, КПП и т.д.)
    """
    return codes_series.fillna('').astype(str).progress_apply(lambda x: x.split('.')[0])


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


def create_object_id(data: pd.DataFrame) -> pd.Series:
    """Получает хэш строки датафрейма
    нужен для создания айдишника объекта

    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    row = data.astype(str).sum(axis=1)
    return row.progress_apply(helpers.get_text_hash)


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


def prepare_geodata_add_lat_lon(data: pd.DataFrame,
                                geodata_col: str,
                                geo: bool=False,
                                latitude_col: str=LATITUDE_COL,
                                longitude_col: str=LONGITUDE_COL,
                                longitude_first: bool=True) -> pd.DataFrame:
    """Преобразует строку с данными о вершинах полигона в список
     добавляет колонки с координатами центроида полигона
    - lat - широта
    - lon - долгота

    Args:
        data (pd.DataFrame): датафрейм с колонкой geodata_col
        geodata_col (str): колонка с данными полигона
        geo (bool): True если хотим на выходе получить geopandas
        latitude_col (str, optional): как назовем колонку широты. Defaults to LATITUDE_COL.
        longitude_col (str, optional): как назовем колонку долготы. Defaults to LONGITUDE_COL.
        longitude_first (bool, optional): если в геоданных долгота стоит на первом месте.
         Defaults to True.

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    data = data.copy()
    data[geodata_col] = data[geodata_col].progress_apply(create_geolist)
    centroid = data[geodata_col].progress_apply(helpers.calc_polygon_centroid)
    data[geodata_col] = data[geodata_col].apply(lambda x: x if x != [] else np.nan)

    lon_i, lat_i = 0, 1

    if not longitude_first:
        lon_i, lat_i = 1, 0

    data[latitude_col] = centroid.apply(lambda x: x[lat_i] if isinstance(x, list) else np.nan)
    data[longitude_col] = centroid.apply(lambda x: x[lon_i] if isinstance(x, list) else np.nan)

    # проверяем корректность полученных данных
    # мы работаем в москве, значит
    # долгота должна начинаться с 5 а широта с 3
    lat_lon = data[[latitude_col, longitude_col]].dropna()
    invalid_lat = (lat_lon[latitude_col] // 10).astype(int) != 5
    invalid_lon = (lat_lon[longitude_col] // 10).astype(int) != 3
    invalid_data = lat_lon[invalid_lat | invalid_lon]
    if invalid_data.shape[0] != 0:
        raise ValueError(f'Ошибки в координатах: {invalid_data}')
    if geo:
        print('geo is True, creating polygons started')
        sleep(.5)
        data[geodata_col] = data[geodata_col].progress_apply(
            lambda x: Polygon(x) if isinstance(x,list) else np.nan
            )
        data = gpd.GeoDataFrame(data, geometry=geodata_col)
    return data


def prepare_dmr_houses_data(data_path: str, geo: bool=False) -> pd.DataFrame:
    """Подготовка данных prepareв_dmr_houses_data
     Адресного реестра объектов недвижимости города Москвы

     Используется в prepare_apartment_houses_data для создания и заливки
     в БД таблички prepared_dmr_houses_data, apartment_houses_all_data

    Где взять сырые данные:
    - нужно зайти по ссылке или найти поиском на https://data.mos.ru
    - найти датасет "Адресный реестр объектов недвижимости города Москвы"
    - скачать его в формате json

    Args:
        data_path (str): путь к файлу json с данными
        geo (bool): если хотим получить geopandas с полигонами
        в geodata

    Returns:
        pd.DataFrame: prepared_dmr_houses_data
    """
    print('prepare_dmr_houses_data started ... ')
    print(f'loading {data_path} ... ', end='')
    data = pd.read_json(data_path, encoding='cp1251', typ='frame')
    print('success')
    data.columns = [col.upper() for col in data.columns]

    # все пропуски приводим к nan
    data = data.replace({val: np.nan for val in NAN_VALUES})\
        .dropna(axis=1, how='all').dropna(axis=0, how='all')# pylint: disable=[no-member]

    # исправляем кадастровые номера
    for col in DMR_KAD_NUM_COLS:
        data[col] = data[col].progress_apply(prepare_kad_num)

    # исправляем данные о точках полигона
    # и добавляем координаты центроида полигона

    # n_fias я вляется ключом для связи с gis_houses_data
    data['N_FIAS'] = data['N_FIAS'].str.upper()
    data = data[~data['ADM_AREA'].isin(ADM_AREA_TO_EXCLUDE)]
    data['NREG'] = codes_to_str(data['NREG'])
    data['KLADR'] = codes_to_str(data['KLADR'])

    data = data[DMR_COLS_MAP.keys()]
    data.columns = [DMR_COLS_MAP[col] for col in data.columns]

    data = prepare_geodata_add_lat_lon(data,
                                       geodata_col=DMR_COLS_MAP[DMR_GEODATA_COL],
                                       geo=geo)
    print('create_object_id started')
    data[OBJECT_ID_COL] = create_object_id(data)
    print('prepare_dmr_houses_data finished')

    return data.drop_duplicates(subset=OBJECT_ID_COL).reset_index(drop=True)


def prepare_gis_houses_data(data_path: str) -> pd.DataFrame:
    """Подготовкой данных ГИС ОЖФ prepared_gis_houses_data

    используется в prepare_apartment_houses_data для создания и заливки
     в БД таблички prepared_gis_houses_data, apartment_houses_all_data

    Где взять сырые данные:
    - нужно зайти по ссылке https://dom.gosuslugi.ru/#!/houses
     или найти поиском "Реестр объектов жилищного фонда"
    - выбрать в Субъект РФ "Москва"
    - нажать "Найти" внизу справа
    - нажать "Скачать" вверху справа
    - скачается папка/архив с несколькими csv, её и будем обрабатывать

    Args:
        data_path (str): путь к папке с csv

    Returns:
        pd.DataFrame: prepared_gis_houses_data
    """
    living_area_col = 'Жилая площадь в доме'
    guid_fias_col = 'Глобальный уникальный идентификатор дома по ФИАС'
    oktmo_col = 'Код ОКТМО'
    ogrn_col = 'ОГРН организации, осуществляющей управление домом'
    kpp_col = 'КПП организации, осуществляющей управление домом'

    print('prepare_dmr_houses_data started ... ')
    print(f'loading {data_path} ... ')
    houses_data_files = get_filenames_from_dir(data_path,
                                               mandatory_substr='csv')
    houses_data_list = []
    print('files from path:', houses_data_files)
    for hdf in houses_data_files:
        houses_data_list.append(pd.read_csv(hdf, delimiter=';', low_memory=False))

    houses_data = pd.concat(houses_data_list, ignore_index=True)\
        .drop_duplicates().reset_index(drop=True)
    print('success')

    houses_data[guid_fias_col] = houses_data[guid_fias_col].str.upper()

    # убираем данные о комнатах и помещениях
    not_room = houses_data['Номер помещения (блока)'].isna() & houses_data['Номер комнаты'].isna()
    houses_data_not_room = houses_data[not_room].copy()

    # все пропуски приводим к nan
    houses_data_not_room = houses_data_not_room.replace({val: np.nan for val in NAN_VALUES})\
        .dropna(axis=1, how='all').dropna(axis=0, how='all')
    houses_data_not_room = houses_data_not_room.drop_duplicates()
    houses_data_not_room[living_area_col] = houses_data_not_room[living_area_col]\
        .str.replace(',', '.').astype(float)

    for col in [oktmo_col, ogrn_col, kpp_col]:
        houses_data_not_room[col] = codes_to_str(houses_data_not_room[col])

    prepared_data = houses_data_not_room[GIS_COLS_MAP.keys()].copy()
    prepared_data.columns = [
        GIS_COLS_MAP[col] for col in prepared_data.columns
        ]

    print('create_object_id started')
    prepared_data[OBJECT_ID_GIS_COL] = create_object_id(prepared_data)
    print('prepare_dmr_houses_data finished')

    return prepared_data.drop_duplicates(subset=OBJECT_ID_GIS_COL).reset_index(drop=True)


def prepare_apartment_houses_data(prepared_dmr: pd.DataFrame,
                                  prepared_gis: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        prepared_dmr (pd.DataFrame): подготовленные данные
         Адресного реестра объектов недвижимости города Москвы
        prepared_gis (pd.DataFrame): подготовленные данные ГИС ОЖФ

    Returns:
        pd.DataFrame: apartment_houses_all_data
    """
    print('prepare_apartment_houses_data started ... ')
    data_merged_fias = prepared_dmr.merge(
        prepared_gis,
        left_on='guid_fias',
        right_on='guid_fias_gis',
        how='left'
        )

    data_merged_kadn = prepared_dmr.dropna(subset=['kad_n']).merge(
        prepared_gis.dropna(subset=['kad_n_gis']),
        left_on='kad_n',
        right_on='kad_n_gis',
        how='inner'
        )

    data_merged_kadzu = prepared_dmr.dropna(subset=['kad_zu']).merge(
        prepared_gis.dropna(subset=['kad_n_gis']),
        left_on='kad_zu',
        right_on='kad_n_gis',
        how='inner'
        )

    data_merged = pd.concat([data_merged_fias, data_merged_kadn, data_merged_kadzu])

    # все пропуски приводим к nan
    data_merged = data_merged.replace({val: np.nan for val in NAN_VALUES})\
        .dropna(axis=1, how='all').dropna(axis=0, how='all')

    # удаляем строки без геопозиции и площади
    data_merged = data_merged.dropna(subset=[LATITUDE_COL, LONGITUDE_COL, 'total_area_gis'])

    # удаляем дубли: сортируем по наличию данных о доме и если дубль,
    # то удаляем тот, у которого нет данных о доме (оставляем last)
    data_merged['has_house_data'] = data_merged['guid_house_gis'].notna()
    data_merged = data_merged.sort_values(by='has_house_data')
    data_merged = data_merged.drop_duplicates(subset=OBJECT_ID_COL, keep='last')
    data_merged = data_merged.drop(columns='has_house_data')

    data_merged[OBJECT_TYPE_COL] = 'многоквартирный дом'
    print('prepare_apartment_houses_data finished')
    return data_merged.reset_index(drop=True)
