import uvicorn
from typing import Optional
import os
from typing import List
import json
import pandas as pd

from postamats.utils import connections
from postamats.utils.helpers import parse_list_fixed_points, parse_list_possidble_points, calculate_workload, parse_list_object_id

from fastapi import FastAPI, Query, Depends
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


@app.get("/get_all_postomat_places")
def get_all_postomat_places():

    db = connections.DB()

    #all_postamat_places = db.get_table_from_bd("all_objects_data")
    all_postamat_places = db.get_by_sql("select * from all_objects_data where object_type!='многоквартирный дом' ")

    return all_postamat_places.to_json(orient='records', force_ascii=False)



@app.get("/get_optimized_postomat_places")
def get_optimized_postomat_places(possible_points: List[str] = Depends(parse_list_possidble_points), 
                                  fixed_points: List[str] = Depends(parse_list_fixed_points)):

    db = connections.DB()
    str_point_object_id = ', '.join([str(s) for s in fixed_points])
    postamat_places = db.get_by_sql(f'''
        select * from all_objects_data 
        where object_type!='многоквартирный дом' and object_id in ({str_point_object_id}) 
        ''')
    # fixed_points_df = pd.DataFrame({'object_id': fixed_points})
    # all_postamat_places = db.get_by_sql("select * from all_objects_data where object_type!='многоквартирный дом' ")
    # postamat_places = pd.merge(all_postamat_places,fixed_points_df, on='object_id', how='inner' )
    return json.dumps({'optimized_points': list(postamat_places.object_id)})
    #return fixed_points


@app.get("/get_point_statistics")
def find_heat_map(step: float = 1, walk_time: float = 15, object_id_str: List[str] = Depends(parse_list_object_id)):

    db = connections.DB()
    walk_time = walk_time * 60 # переводим в секунды
    #object_id_str = ["'" + str(s) + "'" for s in all_postamat_places.object_id[:2000]]
    distance_matrix_filter= db.get_by_filter("distances_matrix_filter", filter_dict = {"step": [step], "object_id": object_id_str}, additional_filter=f' walk_time<={walk_time}')
    center_mass_pd = db.get_by_filter('centers_mass', filter_dict = {"step":[step]})
    population_all = center_mass_pd.population.sum()
    quantity_people_to_postomat, distance_till_nearest_postomat = calculate_workload(center_mass_pd, distance_matrix_filter)
    list_time_m = [3, 5, 10, 15]
    list_time_s = [i*60 for i in list_time_m]
    list_result_percent = []
    for i in list_time_s:
        distance_matrix_filter[f'less_{i}'] = distance_matrix_filter['distance']<i
        list_result_percent.append(center_mass_pd[center_mass_pd.id_center_mass.isin(distance_matrix_filter[distance_matrix_filter[f'less_{i}']].id_center_mass)].population.sum())
    list_result_percent = [round(population,3)*100/population_all for population in list_result_percent]
    list_for_json_percent = [{"time" : list_time_m[0], "percent_people" : list_result_percent[0]},{"time":list_time_m[1], "percent_people": list_result_percent[1]}, {"time":list_time_m[2], "percent_people": list_result_percent[2]}, {"time":list_time_m[3],"percent_people":list_result_percent[3]}]
    return quantity_people_to_postomat.to_json(orient='records'), distance_till_nearest_postomat.to_json(orient='records') , json.dumps(list_for_json_percent)  

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)