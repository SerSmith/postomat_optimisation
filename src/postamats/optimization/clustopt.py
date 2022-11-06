import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from postamats.global_constants import MAX_ACTIVE_RADIUS

MAX_POSTAMAT_AREA = np.pi * MAX_ACTIVE_RADIUS ** 2


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


def set_cluster_postamats(clust_df,
                          clust_area,
                          num_clusters_coef: int=2,
                          dist_thresh=2*MAX_ACTIVE_RADIUS):
    """Расставляет постаматы

    Args:
        clust_df (_type_): _description_
        clust_area (_type_): _description_
        dist_thresh (_type_, optional): _description_. Defaults to 2*MAX_ACTIVE_RADIUS.

    Returns:
        _type_: _description_
    """
    points = clust_df[clust_df['is_point']].copy()
    result = []
    if points.shape[0]==0:
        return result

    n_clusters = num_clusters_coef * int( np.ceil(clust_area / MAX_POSTAMAT_AREA) )
    # TODO: ставить больше кластеров и выбирать топ лучших
    # проверять, что кластеры стоят слишком близко

    clusterer = KMeans(n_clusters=n_clusters)
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
                            distance_threshold: float=MAX_ACTIVE_RADIUS,
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

    dist = pairwise_distances(remove_or_select_from[['x', 'y']],
                              whose_neighbors_remove_or_select[['x', 'y']])
    to_remove_or_select = set( np.where(dist < distance_threshold)[0] )
    cond = remove_or_select_from.index.isin(to_remove_or_select)
    if action=='remove':
        cond = ~cond
    return remove_or_select_from[cond]


