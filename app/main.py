from typing import Union
import os

from postamats.utils import connections

from fastapi import FastAPI

app = FastAPI()


@app.get("/say_hi")
def get_possible_postomat_places():

    return {"Hello": "World"}


@app.get("/get_all_postomat_places")
def get_all_postomat_places():

    db = connections.DB()

    #all_postamat_places = db.get_table_from_bd("all_objects_data")
    all_postamat_places = db.get_by_sql("select * from all_objects_data where object_type!='многоквартирный дом' ")

    return all_postamat_places.to_json(orient='records', force_ascii=False)

 def calculate_workload(center_mass_pd, distance_matrix_pd):
    
    distance_matrix_pd['walk_time'] = distance_matrix_pd['distance']

    only_nearest_points_min_dist = distance_matrix_pd.loc[distance_matrix_pd.groupby('id_center_mass').walk_time.idxmin()]

    only_nearest_points_min_dist_with_pop = only_nearest_points_min_dist.merge(center_mass_pd, on='id_center_mass')

    quantity_people_to_postomat = only_nearest_points_min_dist_with_pop.groupby('object_id').agg({'population': 'sum'}).reset_index()

    distance_till_nearest_postomat = only_nearest_points_min_dist_with_pop[['id_center_mass', 'walk_time']]


    return quantity_people_to_postomat, distance_till_nearest_postomat

@app.get("/get_point_statistics")
def find_heat_map(step, object_id_str):
    
    db = connections.DB()
    #object_id_str = ["'" + str(s) + "'" for s in all_postamat_places.object_id[:2000]]
    distance_matrix_filter= db.get_by_filter("distances_matrix_filter", {"step": [step], "object_id": object_id_str})
    center_mass_pd = db.get_table_from_bd('centers_mass')
    quantity_people_to_postomat, distance_till_nearest_postomat = calculate_workload(center_mass_pd, distance_matrix_filter)
    return quantity_people_to_postomat.to_json(orient='records'), distance_till_nearest_postomat.to_json(orient='records')   
