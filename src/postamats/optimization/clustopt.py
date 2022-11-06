"""модуль с функциями для оптимизации на основе кластеризации
"""

import os
import json
from typing import Union, Tuple, List, Dict, Optional
from warnings import warn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances

from postamats.utils import connections, helpers
from postamats.utils.connections import PATH_TO_ROOT
from postamats.global_constants import MAX_ACTIVE_RADIUS, LATITUDE_COL, LONGITUDE_COL,\
    METERS_TO_SEC_COEF, MAX_POSTAMAT_AREA, MAX_ACTIVE_RADIUS


RANDOM_STATE = 27


def calculate_weights(data: pd.DataFrame, **kwargs) -> pd.Series:
    """Считает веса строк для кластеризатора на основе населения и т.д.

    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    # TODO: добавить расчет весов на основе населения и т.д.
    sample_weight = pd.Series(index=data.index, data=1)
    return sample_weight


def my_quality_score(mean_walk_time: float,
                mean_population: float,
                mean_wt_min: float,
                mean_population_max: float) -> float:
    """Скор для каждой выбранной тчки установки постамата

    Args:
        mean_walk_time (float): среднее время в пути для данной точки
        sum_population (float): суммарное кол-во людей, которое обслуживает точка
        mean_wt_min (float): минимальное среднее время в пути по всем выставленным точкам
        sum_population_max (float): максимальное суммарное кол-во людей, которое обслуживает точка
        по всем точкам

    Returns:
        float: геометрическое среднее, аналог f1-меры
    """
    wt_score = mean_wt_min/mean_walk_time
    pop_score = mean_population / mean_population_max
    return 2 * (wt_score) * (pop_score) / (wt_score + pop_score)


def sort_clusters_by_density(data: pd.DataFrame,
                              label_col: str='label',
                              population_col: str='population',
                              x_step: int=100,
                              y_step: int=100) -> tuple:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        label_col (str, optional): _description_. Defaults to 'label'.
        population_col (str, optional): _description_. Defaults to 'population'.
        x_step (int, optional): _description_. Defaults to 100.
        y_step (int, optional): _description_. Defaults to 100.

    Returns:
        tuple: _description_
    """
    set_labels = set(data[label_col])
    len_labels = len(set_labels)

    clusters_population_density = np.zeros((len_labels,))
    clusters_area = np.zeros((len_labels,))

    for lbl in set(data[label_col]).difference({-1}):

        slice_df = data[data[label_col]==lbl].copy()

        cell_area = x_step * y_step

        slice_df['x_step'] = (slice_df['x'] // x_step).astype(int)
        slice_df['y_step'] = (slice_df['y'] // y_step).astype(int)
        population_by_cell = slice_df.groupby(['x_step', 'y_step'])[population_col].sum()
        clusters_area[lbl] = population_by_cell.shape[0] * cell_area
        slice_population_density = population_by_cell.sum() / clusters_area[lbl]
        clusters_population_density[lbl] = slice_population_density

    clusters_by_density = np.argsort(clusters_population_density)[::-1]
    return clusters_population_density, clusters_area, clusters_by_density


def set_cluster_postamats(clust_df: pd.DataFrame,
                          clust_area: float,
                          remain_postamats_quant: Optional[int]=None,
                          num_clusters_coef: int=2,
                          dist_thresh: float=2*MAX_ACTIVE_RADIUS):
    """_summary_

    Args:
        clust_df (pd.DataFrame): _description_
        clust_area (float): _description_
        remain_postamats_quant (Optional[int], optional): _description_. Defaults to None.
        num_clusters_coef (int, optional): _description_. Defaults to 2.
        dist_thresh (float, optional): Максимальное расстояние от потенциальной точки
         размещения постамата до точки, рекомендованной оптимизатором (центра кластера),
         при котором постамат будет поставлен в данную точку. Если расстояние от точки
         рекомендации до точки потенциального размещения больше и других точек ближе нет,
         то постамат, рекомендованный оптимизатором, не размещается. Defaults to 2*MAX_ACTIVE_RADIUS.

    Returns:
        _type_: _description_
    """
    points = clust_df[clust_df['is_point']].copy()
    result = []
    if points.shape[0]==0:
        return result

    n_clusters = num_clusters_coef * int( np.ceil(clust_area / MAX_POSTAMAT_AREA) )
    # если осталось меньше постаматов, чем хочет оптимизатор, берем, сколько осталось
    if remain_postamats_quant is not None:
        # если вдруг получили 0
        if remain_postamats_quant == 0:
            warn('Доступные для расстановки постаматы закончились')
            return result
        n_clusters = min(remain_postamats_quant, n_clusters)

    # TODO: ставить больше кластеров и выбирать топ лучших
    # проверять, что кластеры стоят слишком близко

    clusterer = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    clusterer.fit_predict(clust_df.loc[~clust_df['is_point'], ['x', 'y']])
    centers = pd.DataFrame(data=clusterer.cluster_centers_, columns=['x', 'y'])

    dist = pairwise_distances(centers[['x', 'y']], points[['x', 'y']])

    neares_to_centers = set(dist.argmin(axis=1))

    closer_then_thresh = set(np.where(dist < dist_thresh)[1])

    to_select = list( set(neares_to_centers) & set(closer_then_thresh) )

    if not to_select:
        return result

    return points.loc[points.index[to_select], 'object_id'].to_list()


def remove_or_select_nearest(remove_or_select_from: pd.DataFrame,
                            whose_neighbors_remove_or_select: pd.DataFrame,
                            distance_threshold: float,
                            action: str='remove'):
    """Убирает или оставляет в remove_or_select_from точки вокруг точек из whose_neighbors_remove
    Args:
        remove_or_select_from (pd.DataFrame): откуда удалять/выбирать объекты
        whose_neighbors_remove_or_select (pd.DataFrame): объекты, окружающие какие точки удалять
        distance_threshold (float, optional): в каком радиусе вокруг whose_neighbors_remove
         удалять точки из remove_from. Defaults to POSTAMAT_TERRITORY_RADIUS.
        action (str): 'remove' - удалять точки, 'select' - добавлять точки
    """
    if action not in ['remove', 'select']:
        raise ValueError(f"action must be 'remove' or 'select', {action} received")

    if remove_or_select_from.shape[0]==0:
        warn('remove_or_select_from пуст')
        return remove_or_select_from

    if whose_neighbors_remove_or_select.shape[0]==0:
        warn_base_msg = 'whose_neighbors_remove_or_select пуст'
        if action=='remove':
            warn(f'{warn_base_msg}, {action}=remove,'
            ' удалять нечего, возвращаю remove_or_select_from as is')
            return remove_or_select_from
        if action=='select':
            warn(f'{warn_base_msg}, {action}=select,'
            'возвращать нечего, возвращаю пустой датасет')
            return pd.DataFrame(columns=remove_or_select_from.column)

        return remove_or_select_from

    dist = pairwise_distances(remove_or_select_from[['x', 'y']],
                              whose_neighbors_remove_or_select[['x', 'y']])
    to_remove_or_select = set( np.where(dist < distance_threshold)[0] )
    cond = remove_or_select_from.index.isin(to_remove_or_select)
    if action=='remove':
        cond = ~cond
    return remove_or_select_from[cond]


def get_data_from_db(
    db_config: Optional[Dict[str, str]]=None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Получаем данные из базы

    Returns:
        _type_: _description_
    """
    database = connections.DB(db_config)

    apart_query = """select object_id, lat, lon, object_type, population
    from public.apartment_houses_all_data"""
    metro_query = """select object_id_metro, lat, lon, object_type
    from public.all_metro_objects_data where object_type='кластер входов в метро'"""
    points_query = 'select object_id, lat, lon, object_type from public.all_objects_data'

    all_apart = database.get_by_sql(apart_query)
    all_metro = database.get_by_sql(metro_query)
    all_points = database.get_by_sql(points_query)

    return all_apart, all_metro, all_points


def kmeans_optimize_points(possible_points: List[str],
                           fixed_points: List[str],
                           quantity_postamats_to_place: int,
                           max_time: float=15,
                           metro_weight: float=0.5,
                           large_houses_priority: float=0.5,
                           is_local_run: bool=False) -> List[str]:
    """_summary_

    Args:
        possible_points (List[str]): _description_
        fixed_points (List[str]): _description_
        postamat_quant (int): _description_
        metro_importance (Optional[float], optional): _description_. Defaults to None.
        large_houses_priority (Optional[float], optional): _description_. Defaults to None.
        is_local_run (bool, optional): _description_. Defaults to False.

    Returns:
        List[str]: _description_
    """

    if not possible_points:
        warn('Не получено ни одной точки для размещения постамата, возвращаю пустой список')
        return []

    if quantity_postamats_to_place<=0:
        warn(f'Нельзя разместить {quantity_postamats_to_place} постаматов, возвращаю пустой список')
        return []

    if metro_weight < 0:
        raise ValueError('metro_weight не может быть отрицательным')

    if large_houses_priority < 0:
        raise ValueError('large_houses_priority не может быть отрицательным')

    db_config = None

    if is_local_run:
        config_path = os.path.join(PATH_TO_ROOT, 'db_config.json')
        with open(config_path, mode='r', encoding='utf-8') as db_file:
            db_config = json.load(db_file)

    all_apart, all_metro, all_points = get_data_from_db(db_config)

    # считаем картезиановы координаты
    all_apart_cartes = helpers.calc_cartesian_coords(all_apart[LATITUDE_COL],
                                                    all_apart[LONGITUDE_COL])
    all_apart_cartes.index = all_apart.index
    all_apart = all_apart.join(all_apart_cartes)

    all_points_cartes = helpers.calc_cartesian_coords(all_points['lat'], all_points['lon'])
    all_points_cartes.index = all_points.index
    all_points = all_points.join(all_points_cartes)

    fixed_points_df = all_points.loc[all_points['object_id'].isin(fixed_points), :].copy()

    # # теперь нам нужно исключить зафиксированные точки из возможных точек расстановки
    cleaned_points = all_points[~all_points['object_id'].isin(fixed_points)]

    # также надо взять только точки, доступные для установки
    cleaned_points = cleaned_points[cleaned_points['object_id'].isin(possible_points)]

    # фильтруем данные о зификсированных точках
    # удаляем из расчета те дома, которые уже обслуживаются постаматом
    cleaned_apart = remove_or_select_nearest(all_apart, fixed_points_df, MAX_ACTIVE_RADIUS)
    # Постаматы имеют эффективный радиус действия, оставляем только те дома,
    # которые находятся в границах эффективного радиуса действия постаматов
    # рассчитываем исходя из max_time
    postamat_effective_radius = 60 * max_time / METERS_TO_SEC_COEF
    cleaned_apart = remove_or_select_nearest(cleaned_apart,
                                             cleaned_points,
                                             distance_threshold=postamat_effective_radius,
                                             action='select')

    # добавляем точки в датасет, чтобы понимать, находятся они в нашей области или нет
    apart_vs_points = pd.concat([cleaned_apart, cleaned_points], ignore_index=True)
    apart_vs_points['is_point'] = apart_vs_points['object_id'].isin(cleaned_points['object_id'])

    # добавляем вес
    # TODO: добавить расчет весов
    sample_weight = calculate_weights(apart_vs_points,
                                      metro_weight=metro_weight,
                                      large_houses_priority=large_houses_priority)
    sample_weight[apart_vs_points['is_point']] = 0

    # теперь мы кластеризуем оставшиес точки притяжения
    dbscan = DBSCAN(eps=400, min_samples=5)
    labels=dbscan.fit_predict(apart_vs_points[['x', 'y']], sample_weight=sample_weight)
    apart_vs_points['label'] = labels

    apart_wo_points = apart_vs_points[~apart_vs_points['is_point']].copy()

    clusters_population_density, clusters_area, clusters_by_density = \
        sort_clusters_by_density(apart_wo_points)

    point_ids = []
    remain_postamats_quant = quantity_postamats_to_place
    for lbl in clusters_by_density:
        new_ids = []
        if remain_postamats_quant <= 0:
            break
        lbl_cond = apart_vs_points['label']==lbl
        new_ids = set_cluster_postamats(apart_vs_points[lbl_cond],
                                        clusters_area[lbl],
                                        remain_postamats_quant=remain_postamats_quant,
                                        num_clusters_coef=2,
                                        dist_thresh=2*MAX_ACTIVE_RADIUS
                                        )
        point_ids += new_ids
        remain_postamats_quant -= len(new_ids)
    return point_ids
