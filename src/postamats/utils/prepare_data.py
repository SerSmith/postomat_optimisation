"""Функции для подготовки сырых данных
для загрузки в базу данных
"""

import os
import re
from time import sleep
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

import geopandas as gpd
from shapely.geometry import Polygon

from postamats.utils import helpers
from postamats.global_constants import NAN_VALUES, OBJECT_ID_COL, DMR_COLS_MAP,\
    DMR_GEODATA_COL, DMR_KAD_NUM_COLS, LATITUDE_COL, LONGITUDE_COL, ADM_AREA_TO_EXCLUDE,\
        GIS_COLS_MAP, OBJECT_ID_GIS_COL, OBJECT_TYPE_COL, INFRA_COLS_MAP, COMBINED_ADDRESS_KEYS,\
            INFRA_WORKING_HOURS_COL, INFRA_COMBINED_ADDRESS_COL

tqdm.pandas()

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

    print('prepare_gis_houses_data started ... ')
    print(f'loading {data_path} ... ')
    houses_data_files = get_filenames_from_dir(data_path,
                                               mandatory_substr='csv')
    if len(houses_data_files) == 0:
        raise FileNotFoundError(f'В папке {data_path} нет csv фалов')

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
    print('prepare_gis_houses_data finished')

    return prepared_data.drop_duplicates(subset=OBJECT_ID_GIS_COL).reset_index(drop=True)


def prepare_apartment_houses_data(prepared_dmr: pd.DataFrame,
                                  prepared_gis: pd.DataFrame) -> pd.DataFrame:
    """Финальная сборка и заливка таблички с данными о многоквартирных домах,
     зданиях и сооружениях

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


def __extract_data_from_address(addr_list: List[Dict[str, str]],
                           addr_key: str) -> str:
    """Извлекает данные по ключу из колонки ObjectAddress. В ней лежат листы словарей
    Их ключи - адреса, районы размещения и проч.

    Args:
        addr_list (List[Dict[str, str]]): сырые данные из ячейки колонки ObjectAddress
        addr_key (str): ключ, данные по которому надо выдернуть

    Returns:
        str
    """
    if not isinstance(addr_list, list):
        return np.nan
    if not addr_list:
        return np.nan
    if addr_key not in addr_list[0]:
        return np.nan
    return addr_list[0][addr_key]


def explode_combined_address(data: pd.DataFrame,
                             combined_address_col: str=INFRA_COMBINED_ADDRESS_COL,
                             combined_address_keys: Optional[List[str]]=None,
                             ) -> pd.DataFrame:
    """Извлекает данные из колонки ObjectAddress (в ней лежат листы словарей;
     их ключи - адреса, районы размещения и проч.) и раскладывает по колонкам

    Args:
        data (pd.DataFrame): данные об объектах инфраструктуры с data.mos.ru,
         загруженные из json
        object_address_col (str): название колонки ObjectAddress
        object_addr_keys (List[str]): ключи, по которым надо забрать даные
         из ObjectAddress

    Returns:
        pd.DataFrame: _description_
    """
    data = data.copy()

    if combined_address_keys is None:
        combined_address_keys = COMBINED_ADDRESS_KEYS

    for key in combined_address_keys:
        data[key] = data[combined_address_col].apply(__extract_data_from_address, args=(key,))

    return data.drop(columns=combined_address_col)


def extract_working_hours(hours_list: List[Dict[str, str]]) -> str:
    """Парси запись с данными о времени работы
     запись - это лист словарей

    Args:
        hours_list (List[Dict[str, str]]): _description_

    Returns:
        str: _description_
    """
    if not isinstance(hours_list, list):
        return np.nan
    if not hours_list:
        return np.nan
    return '; '.join([' '.join(el.values()) for el in hours_list])


def prepare_infrastructure_objects(data: pd.DataFrame,
                                   geodata_col: str,
                                   object_type: str,
                                   working_hours_col: str=INFRA_WORKING_HOURS_COL,
                                   combined_address_col: str=INFRA_COMBINED_ADDRESS_COL,
                                   combined_address_keys: Optional[List[str]]=None,
                                   cols_map: Optional[Dict[str, str]]=None,
                                   needed_cols: Optional[List[str]]=None):
    """Обрабатывает датафрейм с сырыми данными об объектах инфраструктуры из списка:
    - Нестационарные торговые объекты по реализации печатной продукции:
     https://data.mos.ru/opendata/2781
    - Нестационарные торговые объекты: https://data.mos.ru/opendata/619
    - Многофункциональные центры предоставления государственных и муниципальных услуг
     https://data.mos.ru/opendata/-mnogofunktsionalnye-tsentry-predostavleniya-gosudarstvennyh-uslug
    - Библиотеки города: https://data.mos.ru/opendata/7702155262-biblioteki
    - Дома культуры и клубы: https://data.mos.ru/opendata/7702155262-doma-kultury-i-kluby
    - Спортивные объекты города Москвы:
     https://data.mos.ru/opendata/7708308010-sportivnye-obekty-goroda-moskvy

    Args:
        data (pd.DataFrame): датафрейм с сырыми данными

        geodata_col (str): Колонка с геоданными.

        object_type (str): тип объекта (киоск, МФЦ и т.д.)

        working_hours_col (str, optional): в некоторых справочниках есть колонка с часами работы
         В ней данные лежат в листах словарей. Если указать её название, данные распарсятся.
         Defaults to 'WorkingHours'.

        combined_address_col (ObjectAddress): в некоторых справочниках данные колонки с адресом
         содержат листы словарей, где собран адрес, район и округ; если в данных это так,
         нужно указать название колонки, чтобы она разобралась на 3

        cols_map (Optional[Dict[str, str]], optional): Мэппинг колонок. Defaults to None.
         Те колонки из сырых данных, которые встретятся в мэппинге, будут согласно ему переименованы

        needed_cols (Optional[List[str]], optional): какие колонки брать из сырых данных
         для загрузки в БД. Defaults to None.

    Raises:
        TypeError: _description_
        TypeError: _description_
        TypeError: _description_
        ValueError: _description_
    """
    print('prepare_infrastructure_objects started ... ')
    if cols_map is None:
        cols_map = INFRA_COLS_MAP

    if combined_address_keys is None:
        combined_address_keys = COMBINED_ADDRESS_KEYS

    if needed_cols is None:
        needed_cols = data.columns.to_list()

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f'data must be pd.DataFrame, {type(data)} found')
    if not isinstance(cols_map, dict):
        raise TypeError(f'cols_map must be dict, {type(cols_map)} found')
    if not isinstance(needed_cols, list):
        raise TypeError(f'needed_cols must be list, {type(needed_cols)} found')

    data = data[needed_cols].copy()

    if geodata_col not in data.columns:
        raise ValueError(f'Колонка {geodata_col} отсутствует в данных')

    data = prepare_geodata_add_lat_lon(data, geodata_col=geodata_col)

    if working_hours_col in data.columns:
        data[working_hours_col] = data[working_hours_col].apply(extract_working_hours)

    if combined_address_col in data.columns:
        data = explode_combined_address(data,
                                        combined_address_col=combined_address_col,
                                        combined_address_keys=combined_address_keys)

    if ('ObjectType' in data.columns) and ('Category' not in data.columns):
        data['Category'] = data['ObjectType'].copy()
        data = data.drop(columns='ObjectType')

    if 'Category' not in data.columns:
        data['Category'] = object_type

    if 'Specialization' in data.columns:
        data['Specialization'] = data['Specialization'].str.replace('[', '', regex=False)
        data['Specialization'] = data['Specialization'].str.replace(']', '', regex=False)

    old_cols = data.columns
    if any(old_cols.duplicated()):
        raise ValueError(f'В колонках {old_cols} есть дубли, проверьте входные данные')

    data.columns = [cols_map[col] if col in cols_map else col.lower()
                    for col in data.columns]

    if any(data.columns.duplicated()):
        raise ValueError(f'После мэппинга колонки {old_cols} задублились: {data.columns}'
                         f', проверьте {cols_map}, {needed_cols}')

    data[OBJECT_TYPE_COL] = object_type
    data[OBJECT_ID_COL] = create_object_id(data)
    print('prepare_infrastructure_objects finished')
    return data.drop_duplicates(subset=OBJECT_ID_COL).reset_index(drop=True)


def calc_population(moscow_population: pd.DataFrame,
                    apartment_houses: pd.DataFrame) -> pd.DataFrame:
    """Считает население дома по плотности населения муниципального округа
     и площади помещений

    Args:
        moscow_population (pd.DataFrame): _description_
        all_apartment_houses (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    moscow_population = moscow_population.copy()
    apartment_houses = apartment_houses.copy()

    # мэпим датафреймы по муниципальному округу
    # предварительно очищая названия
    moscow_population['district_cleaned'] = \
        moscow_population['district'].str.replace(
            ' (МО)', '', regex=False
            ).str.strip().str.upper()

    district_population = moscow_population.set_index('district_cleaned')['population']

    apartment_houses['district_cleaned'] = \
        apartment_houses['district'].str.replace(
            'муниципальный округ ', ''
            ).str.strip().str.upper()

    district_houses_area_sum = \
        apartment_houses.groupby('district_cleaned')['total_area_gis'].sum()
    district_population_density = (district_population / district_houses_area_sum).dropna()

    ah_population_density = apartment_houses['district_cleaned']\
            .map(district_population_density)

    if ah_population_density.isna().sum() != 0:
        raise ValueError('При мэппинге населения по названию округа возникли пропуски в данных')

    apartment_houses['population'] = \
        apartment_houses['total_area_gis'] * ah_population_density

    no_total_but_living = (
        (apartment_houses['total_area_gis']==0)
        |
        apartment_houses['total_area_gis'].isna()
        ) & (apartment_houses['living_area_gis'] > 0)

    apartment_houses.loc[no_total_but_living, 'total_area_gis'] = \
        apartment_houses.loc[no_total_but_living, 'living_area_gis']

    # проверяем, что население, рассчитанное двумя методами совпадает до десятков
    population1 = int(
        district_population[district_houses_area_sum.index].sum() // 10
        )
    population2 = int(apartment_houses['population'].sum() // 10)
    if population1 != population2:
        raise ValueError('Население рассчиталось неправильно, проверьте входные'
                        ' данные/алгоритм calc_population:'
                        f'{population1} != {population2}')

    return apartment_houses.drop(columns='district_cleaned')
