"""Вспомогательные функции
"""
import os
from typing import Dict, Tuple, List, Generator, Optional, Union
import hashlib
from warnings import warn
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt, pi
from shapely.geometry import Polygon
import numpy as np
import pandas as pd

from postamats.global_constants import CENTER_LAT, CENTER_LON
from postamats.utils.connections import PATH_TO_ROOT

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
    x, y, m = np.asarray(x), np.asarray(y), np.asarray(m)
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
    dlon = abs(lon2 - lon1)
    dlat = abs(lat2 - lat1)
    hav_arg = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * (sin(dlon/2)**2)
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
    dlon = abs(rcoords[lon2_col] - rcoords[lon1_col])
    dlat = abs(rcoords[lat2_col] - rcoords[lat1_col])

    hav_arg = np.sin(dlat/2)**2 + \
        np.cos(rcoords[lat1_col]) * np.cos(rcoords[lat2_col]) * (np.sin(dlon/2)**2)
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


def calc_cartesian_coords(lat_series: pd.Series,
                          lon_series: pd.Series,
                          center_lat: float=CENTER_LAT,
                          center_lon: float=CENTER_LON) -> pd.DataFrame:
    """Переводит геокоординаты в декартовы, считая нулем координат
     центр Москвы

    Args:
        lat_series (pd.Series): _description_
        lon_series (pd.Series): _description_
        moscow_center_lat (float): _description_
        moscow_center_lon (float): _description_

    Returns:
        pd.DataFrame: _description_
    """
    cart_coords = pd.DataFrame()
    cart_coords['lat'] = np.asarray(lat_series)
    cart_coords['lon'] = np.asarray(lon_series)
    cart_coords['c_lat'] = center_lat
    cart_coords['c_lon'] = center_lon
    cart_coords['x'] = haversine_vectorized(cart_coords, 'c_lat', 'c_lon', 'c_lat', 'lon')
    cart_coords['y'] = haversine_vectorized(cart_coords, 'c_lat', 'c_lon', 'lat', 'c_lon')
    minus_cond_x = cart_coords['c_lon'] > cart_coords['lon']
    minus_cond_y = cart_coords['c_lat'] > cart_coords['lat']
    cart_coords.loc[minus_cond_x, 'x'] = cart_coords.loc[minus_cond_x, 'x'] * (-1)
    cart_coords.loc[minus_cond_y, 'y'] = cart_coords.loc[minus_cond_y, 'y'] * (-1)

    return cart_coords[['x', 'y']]


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


def plot_map(
    cartes1: pd.DataFrame,
    cartes2: Optional[pd.DataFrame]=None,
    size1: int=10,
    size2: int=1,
    alpha1: float=.2,
    alpha2: float=1,
    c1: Union[str, List]='b',
    c2: Union[str, List]='r'):
    """Печатает точки на фоне предсохраненной карты Москвы

    Args:
        cartes1 (pd.DataFrame): _description_
        cartes2 (_type_, optional): _description_. Defaults to None.
        size1 (int, optional): _description_. Defaults to 10.
        size2 (int, optional): _description_. Defaults to 1.
        alpha1 (float, optional): _description_. Defaults to .2.
        alpha2 (int, optional): _description_. Defaults to 1.
        c1 (str, optional): _description_. Defaults to 'b'.
        c2 (str, optional): _description_. Defaults to 'r'.
    """

    mos_img = plt.imread(os.path.join(PATH_TO_ROOT, 'data', 'images', 'map.png'))

    bbox_geo = (37.3260, 37.9193, 55.5698, 55.9119)
    bbox_cartes = calc_cartesian_coords(bbox_geo[2:], bbox_geo[:2])
    bbox = bbox_cartes['x'].to_list() + bbox_cartes['y'].to_list()

    fig, ax = plt.subplots(figsize=(12,12))
    ax.scatter(cartes1['x'], cartes1['y'], zorder=1, alpha=alpha1, c=c1, s=size1)
    if cartes2 is not None:
        ax.scatter(cartes2['x'], cartes2['y'], zorder=1, alpha=alpha2, c=c2, s=size2)

    ax.set_xlim(bbox[0],bbox[1])
    ax.set_ylim(bbox[2],bbox[3])
    ax.axis('off')
    ax.imshow(mos_img, zorder=0, extent=bbox, aspect='equal')
    plt.show()
