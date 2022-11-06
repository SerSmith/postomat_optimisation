import os
import json

from sklearn.cluster import KMeans, DBSCAN

from typing import Union, Tuple, List, Dict

import uvicorn
import pandas as pd

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from postamats.utils import connections, helpers
from postamats.optimization import clustopt

from postamats.utils.connections import PATH_TO_ROOT
from postamats.global_constants import LATITUDE_COL, LONGITUDE_COL
from postamats.global_constants import MAX_ACTIVE_RADIUS


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_data_from_db(db_config: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


@app.get("/optimize_points")
def optimize_points(**kwargs) -> List[str]:
    """Расставляет постаматы
    """
    possible_points = kwargs['possible_points']
    fixed_points = kwargs['fixed_points']
    postamat_quant = kwargs['postamat_quant']


    CONFIG_PATH = os.path.join(PATH_TO_ROOT, 'db_config.json')
    with open(CONFIG_PATH, mode='r') as db_file:
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
    cleaned_apart = clustopt.remove_or_select_nearest(all_apart, fixed_points_df)
    # Постаматы имеют эффективный радиус действия, оставляем только те дома,
    # которые находятся в границах эффективного радиуса действия постаматов
    # радиус берем с запасом 400 * 3 = 1200 м
    cleaned_apart = clustopt.remove_or_select_nearest(cleaned_apart,
                                                    cleaned_points,
                                                    distance_threshold=3*MAX_ACTIVE_RADIUS,
                                                    action='select')

    # добавляем точки в датасет, чтобы понимать, находятся они в нашей области или нет
    apart_vs_points = pd.concat([cleaned_apart, cleaned_points], ignore_index=True)
    apart_vs_points['is_point'] = apart_vs_points['object_id'].isin(cleaned_points['object_id'])

    # добавляем вес
    # TODO: добавить расчет весов
    sample_weight = clustopt.calculate_weights(apart_vs_points)
    sample_weight[apart_vs_points['is_point']] = 0

    # теперь мы кластеризуем оставшиес точки притяжения
    dbscan = DBSCAN(eps=400, min_samples=5)
    labels=dbscan.fit_predict(apart_vs_points[['x', 'y']], sample_weight=sample_weight)
    print(len(set(labels)))
    apart_vs_points['label'] = labels

    apart_wo_points = apart_vs_points[~apart_vs_points['is_point']].copy()

    clusters_population_density, clusters_area, clusters_by_density = \
        clustopt.sort_clusters_by_density(apart_wo_points)

    point_ids = []
    remain_postamats_quant = 1000
    for lbl in clusters_by_density:
        new_ids = []
        if remain_postamats_quant <= 0:
            break
        lbl_cond = apart_vs_points['label']==lbl
        new_ids = clustopt.set_cluster_postamats(apart_vs_points[lbl_cond],
                                                clusters_area[lbl],
                                                remain_postamats_quant)
        point_ids += new_ids
        remain_postamats_quant -= len(new_ids)
    return point_ids


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
