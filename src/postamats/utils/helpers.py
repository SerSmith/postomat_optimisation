"""Вспомогательные функции
"""
from typing import Tuple, List
import hashlib
from warnings import warn
from math import radians, cos, sin, asin, sqrt
from shapely.geometry import Polygon
import numpy as np

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


def get_text_hash(text: str) -> str:
    """Получает хэш текста, нужна для однозначного индексирования данных,
     используется в get_row_hashes

    Args:
        text (str): текст

    Returns:
        str: хэш
    """
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()
