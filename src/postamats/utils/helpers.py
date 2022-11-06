"""Вспомогательные функции
"""
import os
from typing import Dict, Tuple, List, Generator, Optional, Union
import hashlib
from warnings import warn
from math import radians, cos, sin, asin, sqrt, pi
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np
import pandas as pd


from typing import Optional

from postamats.global_constants import CENTER_LAT, CENTER_LON
from fastapi import Query


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


def step_algorithm(distance_matrix_filter, center_mass_pd, list_object_id, population):
    " Функция для шага жадного алгоритма "
    only_nearest_points_min_dist = distance_matrix_filter.loc[distance_matrix_filter.groupby('id_center_mass').walk_time.idxmin()]
    only_nearest_points_min_dist_with_pop = only_nearest_points_min_dist.merge(center_mass_pd, on='id_center_mass')
    quantity_people_to_postomat = only_nearest_points_min_dist_with_pop.groupby('object_id').agg({'population': 'sum'}).reset_index()
    object_id = quantity_people_to_postomat.loc[quantity_people_to_postomat.population.idxmax()]['object_id']
    list_object_id.append(object_id)
    population_step = center_mass_pd[center_mass_pd.id_center_mass.isin(list(distance_matrix_filter[distance_matrix_filter.object_id==object_id].id_center_mass))].population.sum()
    population =  population + population_step
    distance_matrix_filter = distance_matrix_filter[~distance_matrix_filter.id_center_mass.isin(list(distance_matrix_filter[distance_matrix_filter.object_id==object_id].id_center_mass))]
    print(quantity_people_to_postomat.loc[quantity_people_to_postomat.population.idxmax()]['population'], object_id)
    return distance_matrix_filter, list_object_id, population


def greedy_algo(db, list_possible_postomat, step=1, time=15*60, cnt_postomat = 1500):
    """
    Жадный алгоритм для нахождения оптимальных мест для постановки постаматов
    Args:
        list_possible_postomat (list): список возможных мест для постановки постаматов
        step (float)
        time (float) - максимальное время для шаговой доступности до постамата
        cnt_postomat (int) - максимальное количество постоматов, которое можно поставить
        db - коннекшен к БД

    Return:
        list_object_id (list) - набор id, куда мы ставим постоматы
        population (int) - количество населения, которое мы покрываем расставленными постоматами

    """
    center_mass_pd = db.get_by_filter('centers_mass', filter_dict={"step":[step]})
    object_id_str = ["'" + str(s) + "'" for s in list_possible_postomat]
    distance_matrix_filter= db.get_by_filter("distances_matrix_filter", filter_dict = {"step": [step], "object_id": object_id_str}, additional_filter=f' walk_time<{time}')
    list_object_id = []
    population = 0
    #start = datetime.datetime.now()
    while (distance_matrix_filter.shape[0]>0) & (len(list_object_id)<cnt_postomat):
        distance_matrix_filter, list_object_id, population = step_algorithm(distance_matrix_filter, center_mass_pd, list_object_id, population)
    #end = datetime.datetime.now()
    #print(end-start)
    return list_object_id, population  


def remove_prefix(text: str, prefix: str):
    return text[text.startswith(prefix) and len(prefix):]

def remove_postfix(text: str, postfix: str):
    if text.endswith(postfix):
        text = text[:-len(postfix)]
    return text

def parse_list_object_id(list_object_id: List[str] = Query(None)) -> Optional[List]:
    """
    accepts strings formatted as lists with square brackets
    names can be in the format
    "[bob,jeff,greg]" or '["bob","jeff","greg"]'
    """

    names = list_object_id

    if names is None:
        return

    # we already have a list, we can return
    if len(names) > 1:
        return names

    # if we don't start with a "[" and end with "]" it's just a normal entry
    flat_names = names[0]
    if not flat_names.startswith("[") and not flat_names.endswith("]"):
        return names

    flat_names = remove_prefix(flat_names, "[")
    flat_names = remove_postfix(flat_names, "]")

    names_list = flat_names.split(",")
    names_list = [remove_prefix(n.strip(), "\"") for n in names_list]
    names_list = [remove_postfix(n.strip(), "\"") for n in names_list]

    return names_list

def calculate_workload(center_mass_pd, distance_matrix_pd):

    only_nearest_points_min_dist = distance_matrix_pd.loc[distance_matrix_pd.groupby('id_center_mass').walk_time.idxmin()]

    only_nearest_points_min_dist_with_pop = only_nearest_points_min_dist.merge(center_mass_pd, on='id_center_mass')

    quantity_people_to_postomat = only_nearest_points_min_dist_with_pop.groupby('object_id').agg({'population': 'sum'}).reset_index()

    distance_till_nearest_postomat = only_nearest_points_min_dist_with_pop[['id_center_mass', 'walk_time','lat','lon']]


    return quantity_people_to_postomat, distance_till_nearest_postomat



def parse_inside(names):

    if names is None:
        return

    # we already have a list, we can return
    if len(names) > 1:
        return names

    # if we don't start with a "[" and end with "]" it's just a normal entry
    flat_names = names[0]
    if not flat_names.startswith("[") and not flat_names.endswith("]"):
        return names

    flat_names = remove_prefix(flat_names, "[")
    flat_names = remove_postfix(flat_names, "]")

    names_list = flat_names.split(",")
    names_list = [remove_prefix(n.strip(), "\"") for n in names_list]
    names_list = [remove_postfix(n.strip(), "\"") for n in names_list]
    return names_list


def parse_object_type_filter_list(possible_points: List[str] = Query(None)) -> Optional[List]:
    """
    accepts strings formatted as lists with square brackets
    names can be in the format
    "[bob,jeff,greg]" or '["bob","jeff","greg"]'
    """

    names_list = parse_inside(possible_points)

    return names_list


def parse_list_fixed_points(fixed_points: List[str] = Query(None)) -> Optional[List]:
    """
    accepts strings formatted as lists with square brackets
    names can be in the format
    "[bob,jeff,greg]" or '["bob","jeff","greg"]'
    """

    names_list = parse_inside(fixed_points)

    return names_list


def parse_district_type_filter_list(district_type_filter_list: List[str] = Query(None)) -> Optional[List]:
    """
    accepts strings formatted as lists with square brackets
    names can be in the format
    "[bob,jeff,greg]" or '["bob","jeff","greg"]'
    """

    names_list = parse_inside(district_type_filter_list)

    return names_list


def parse_adm_areat_type_filter_list(adm_areat_type_filter_list: List[str] = Query(None)) -> Optional[List]:
    """
    accepts strings formatted as lists with square brackets
    names can be in the format
    "[bob,jeff,greg]" or '["bob","jeff","greg"]'
    """


    names_list = parse_inside(adm_areat_type_filter_list)

    return names_list

def parse_banned_points_list(banned_points: List[str] = Query(None)) -> Optional[List]:
    """
    accepts strings formatted as lists with square brackets
    names can be in the format
    "[bob,jeff,greg]" or '["bob","jeff","greg"]'
    """

    names_list = parse_inside(banned_points)

    return names_list
def add_quates(obj_list):
    if obj_list is not None:
        return ["'" + str(s) + "'" for s in obj_list]


def parse_list_possidble_points(list_possidble_points: List[str] = Query(None)) -> Optional[List]:
    """
    accepts strings formatted as lists with square brackets
    names can be in the format
    "[bob,jeff,greg]" or '["bob","jeff","greg"]'
    """


    names_list = parse_inside(list_possidble_points)

    return names_list

def make_points_lists(db,
                      object_type_filter_list: Optional[List[str]],
                      district_type_filter_list: Optional[List[str]],
                      adm_areat_type_filter_list: Optional[List[str]],
                      banned_points: Optional[List[str]]):
    """На основе переданных фильтров формируем список точек, где могут стоят постаматы

    Args:
        db (_type_): конектшен к БД
        object_type_filter_list (Optional(List[str])): типы объектов
        district_type_filter_list (Optional(List[str])): округа
        adm_areat_type_filter_list (Optional(List[str])): районы
        banned_points (Optional(List[str])): список точек,  где запрещено ставить постаматы

    Returns:
        _type_: _description_
    """
    
    if object_type_filter_list is None:
        object_type_filter_list = []
    if district_type_filter_list is None:
        district_type_filter_list = []
    if adm_areat_type_filter_list is None:
        adm_areat_type_filter_list = []
    if banned_points is None:
        banned_points = []

    objects = db.get_by_filter("all_objects_data", {"object_type": add_quates(object_type_filter_list),
                                                    "district": add_quates(district_type_filter_list),
                                                    "adm_area": add_quates(adm_areat_type_filter_list)
                                                 }, additional_filter="object_type != 'многоквартирный дом'")

    possible_postomats = list(set(objects['object_id'].to_list()).difference(set(banned_points)))
    return possible_postomats
    
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

