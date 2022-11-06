import uvicorn
from typing import Union
import os
from typing import List

from postamats.utils import connections

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/say_hi")
def get_possible_postomat_places():

    return {"Hello": "World"}


@app.get("/get_data_from_db")
def get_data_from_db():
    """Получаем данные из базы

    Returns:
        _type_: _description_
    """

    database = connections.DB()

    apart_query = """select object_id, lat, lon, object_type, population
    from public.apartment_houses_all_data"""
    metro_query = """select object_id_metro, lat, lon, object_type
    from public.all_metro_objects_data where object_type='кластер входов в метро'"""
    points_query = 'select object_id, lat, lon, object_type from public.all_objects_data'

    all_apart = database.get_by_sql(apart_query)
    all_metro = database.get_by_sql(metro_query)
    all_points = database.get_by_sql(points_query)

    return all_apart.to_json(orient='records', force_ascii=False)

def optimize_points():
    pass

# @app.get("/get_point_statistics")
# def find_heat_map(step, object_id_str: List[str] = Query(None)):

#     db = connections.DB()
#     distance_matrix_filter= db.get_by_filter("distances_matrix_filter", {"step": [step], "object_id": object_id_str})
#     center_mass_pd = db.get_table_from_bd('centers_mass')
#     quantity_people_to_postomat, distance_till_nearest_postomat = calculate_workload(center_mass_pd, distance_matrix_filter)
#     return quantity_people_to_postomat.to_json(orient='records'), distance_till_nearest_postomat.to_json(orient='records')   

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)