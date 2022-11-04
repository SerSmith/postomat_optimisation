"""Вспомогательные функции
"""
from typing import Dict, Tuple, List, Generator
import hashlib
from warnings import warn
from math import radians, cos, sin, asin, sqrt, pi
from shapely.geometry import Polygon
import numpy as np
import pandas as pd

# Радиус Земли на широте Москвы
EARTH_R = 6363568

# скорость ходьбы в метрах
WALK_SPEED = 5000 / 3600

def find_center_mass(x: np.array,
                     y: np.array,
                     m: np.array) -> Tuple[float, float, float]:
    """
    function, that recievs some points and find point - center mass
    x - np.array of x-coordinates
    y - np.array of y-coordinates
    m - np.array of mass (number of people)
    """
    cgx = np.sum(x*m)/np.sum(m)
    cgy = np.sum(y*m)/np.sum(m)
    sum_m = sum(m)
    return cgx, cgy, sum_m


def haversine(lat1: float, lon1: float,
              lat2: float, lon2: float) -> float:
    """
    Считает расстояниие между точками,
    координаты которых заданы в виде широты и долготы

    Args:
        lat1 (float): широта точки 1
        lon1 (float): долгота точки 1
        lat2 (float): широта точки 2
        lon2 (float): долгота точки 2

    Returns:
        float: расстояние между точками в метрах
    """
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # haversine formula
    # The Haversine (or great circle) distance is the angular distance between
    #  two points on the surface of a sphere.
    #  The first coordinate of each point is assumed to be the latitude,
    #  the second is the longitude, given in radians
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    hav_arg = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    hav_dist = 2 * asin(sqrt(hav_arg))
    distance = EARTH_R * hav_dist
    return distance


def haversine_vectorized(
        data: pd.DataFrame,
        lat1_col: str,
        lon1_col: str,
        lat2_col: str,
        lon2_col: str
        ) -> pd.Series:
    """Считает расстояниие между точками,
    координаты которых заданы в виде широты и долготы
    по всему датафрейму

    Args:
        data (pd.DataFrame): датафрейм с геокоординатами точек
        latx_col (str): колонка с широтой точки 1
        lonx_col (str): колонка с долготой точки 1
        laty_col (str): колонка с широтой точки 2
        lony_col (str): колонка с долготой точки 2

    Returns:
        pd.Series: _description_
    """

    # convert decimal degrees to radians
    rcoords = {}

    for col in [lat1_col, lon1_col, lat2_col, lon2_col]:
        rcoords[col] = pi * data[col] / 180
    # haversine formula
    # The Haversine (or great circle) distance is the angular distance between
    #  two points on the surface of a sphere.
    #  The first coordinate of each point is assumed to be the latitude,
    #  the second is the longitude, given in radians
    dlon = rcoords[lon2_col] - rcoords[lon1_col]
    dlat = rcoords[lat2_col] - rcoords[lat1_col]

    hav_arg = np.sin(dlat/2)**2 + \
        np.cos(rcoords[lat1_col]) * np.cos(rcoords[lat2_col]) * np.sin(dlon/2)**2
    hav_dist = 2 * np.arcsin(np.sqrt(hav_arg))
    distance = EARTH_R * hav_dist
    return distance


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
    # assert calc_polygon_centroid([[0,0], [1,0], [1,1], [0,1]]) == [.5, .5]
    # assert calc_polygon_centroid([[0,1], [0,-1], [2,0]]) == [2/3, 0]
    # assert calc_polygon_centroid([[0,-1], [0,1], [2,0]]) == [2/3, 0]
    # assert calc_polygon_centroid([[0,0], [1,0], [1.5,0.5], [1,1], [0,1], [-.5,.5]]) == [.5, .5]

    if not isinstance(coords, list):
        warn(f"{coords} is not list: {type(coords)}")
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
    return list( list(plgn.centroid.coords)[0] )


def find_degreee_to_distance(df: pd.DataFrame) -> Tuple[float, float]:
    "Функция, которая возвращает чему равен 1 градус по широте и долготе в градусах"
    lat_min = df.lat.min()
    lat_max = df.lat.max()
    lon_min = df.lon.min()
    lon_max = df.lon.max()
    lat_length= haversine(lat_min, lon_min, lat_max, lon_min)
    lon_length= haversine(lat_min, lon_min, lat_min, lon_max)
    lat_km = lat_length/1000/(lat_max-lat_min)
    lon_km = lon_length/1000/(lon_max-lon_min)
    print(f'latitude 1 degree = {lat_km} km', f'longitude 1 degree = {lon_km} km') 

    return lat_km, lon_km


def make_net_with_center_mass(df_homes: pd.DataFrame,
                              step: float,
                              distance_to_degree: Dict) -> pd.DataFrame:
    """
    Функция, которая накладывает объекты (дома) на сетку и в каждой ячейке считает центр масс
    В df_homes обязаны быть поля population, lat, lon

    """
    df = df_homes.copy()
    df.columns = [column.lower() for column in df.columns]

    step_lon = step * distance_to_degree['lon']
    step_lat = step * distance_to_degree['lat']

    df['lat_n'] = df.lat // step_lat
    df['lon_n'] = df.lon // step_lon
    df['lat_n'] = df['lat_n'].astype('int')
    df['lon_n'] = df['lon_n'].astype('int')
    df['lat_n_lon_n'] = df['lat_n'].astype('str') + '_' + df['lon_n'].astype('str')
    df['step'] = step

    df['id_center_mass'] = df['lat_n_lon_n'] + '_' + df['step'].astype(str)

    df['lat_population'] = df['lat']*df['population']
    df['lon_population'] = df['lon']*df['population']
    df_agg = df.groupby(['id_center_mass'])\
        .agg({'population':'sum','lat_population':'sum','lon_population':'sum'})\
            .reset_index().rename({'population':'sum_population'}, axis=1)

    df_agg['lat'] = df_agg['lat_population']/df_agg['sum_population']
    df_agg['lon'] = df_agg['lon_population']/df_agg['sum_population']
    df_agg['population'] = df_agg['sum_population']
    df_agg = df_agg[['id_center_mass','lat','lon','population']]
    df_agg['step'] = step

    return df_agg


def get_text_hash(text: str) -> str:
    """Получает хэш текста, нужна для однозначного индексирования данных,
     используется в get_row_hashes

    Args:
        text (str): текст

    Returns:
        str: хэш
    """
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()


def df_generator(df: pd.DataFrame, max_size: int) -> Generator:
    """Создает генератор датафреймов пандас из входного датафрейма
    каждый датафрейм в генераторе равен или меньше max_size

    Args:
        df (pd.DataFrame): датафрейм
        max_size (int): максимально допустимый размер датафрейма на выходе

    Returns:
        _type_: _description_

    Yields:
        Generator: _description_
    """
    step = int(np.ceil(df.shape[0] / max_size))
    df_slices = (df.iloc[i * max_size : (i + 1) * max_size, :].copy() for i in range(step))
    return df_slices