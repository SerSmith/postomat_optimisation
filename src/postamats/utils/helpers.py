"""Вспомогательные функции
"""
from typing import Tuple
from math import radians, cos, sin, asin, sqrt
import numpy as np

# Радиус Земли на широте Москвы
EARTH_R = 6363568

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
