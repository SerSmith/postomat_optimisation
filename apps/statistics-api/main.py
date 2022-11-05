import uvicorn
from typing import Union, Optional
import os
from typing import List
import json
import pandas as pd

from postamats.utils import connections

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


def parse_list_possidble_points(possible_points: List[str] = Query(None)) -> Optional[List]:
    """
    accepts strings formatted as lists with square brackets
    names can be in the format
    "[bob,jeff,greg]" or '["bob","jeff","greg"]'
    """

    names = possible_points
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

def parse_list_fixed_points(fixed_points: List[str] = Query(None)) -> Optional[List]:
    """
    accepts strings formatted as lists with square brackets
    names can be in the format
    "[bob,jeff,greg]" or '["bob","jeff","greg"]'
    """
    names = fixed_points
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


def calculate_workload(center_mass_pd, distance_matrix_pd):

    only_nearest_points_min_dist = distance_matrix_pd.loc[distance_matrix_pd.groupby('id_center_mass').walk_time.idxmin()]

    only_nearest_points_min_dist_with_pop = only_nearest_points_min_dist.merge(center_mass_pd, on='id_center_mass')

    quantity_people_to_postomat = only_nearest_points_min_dist_with_pop.groupby('object_id').agg({'population': 'sum'}).reset_index()

    distance_till_nearest_postomat = only_nearest_points_min_dist_with_pop[['id_center_mass', 'walk_time','lat','lon']]


    return quantity_people_to_postomat, distance_till_nearest_postomat


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


# @app.get("/hello_list")
# def hello_list(names: List[str] = Depends(parse_list)):
#     """ list param method """

#     if names is not None:
#         return StreamingResponse((f"Hello {name}" for name in names))
#     else:
#         return {"message": "no names"}

@app.get("/get_point_statistics")
def find_heat_map(step: float = 1, walk_time: float = 15, object_id_str: List[str] = Depends(parse_list_object_id)):

    db = connections.DB()
    #object_id_str = ["'" + str(s) + "'" for s in all_postamat_places.object_id[:2000]]
    distance_matrix_filter= db.get_by_filter("distances_matrix_filter", filter_dict = {"step": [step], "object_id": object_id_str}, additional_filter=f' walk_time<={walk_time}')
    center_mass_pd = db.get_table_from_bd('centers_mass')
    quantity_people_to_postomat, distance_till_nearest_postomat = calculate_workload(center_mass_pd, distance_matrix_filter)
    return quantity_people_to_postomat.to_json(orient='records'), distance_till_nearest_postomat.to_json(orient='records')   

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)