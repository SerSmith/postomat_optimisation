import uvicorn
from typing import Union
from typing import List
import json

from postamats.utils import connections
from postamats.optimization import optimisation

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



@app.get("/get_optimized_postomat_places")
def get_optimized_postomat_places(possible_postomats: List[str] = Query(None),
                                  fixed_points: List[str] = Query(None),
                                  quantity_postamats_to_place=1500,
                                  step=0.5,
                                  metro_weight=0.5):

    kwargs = {'ratioGap': 0.001, 'sec': 250}

    possible_points = ["'" + str(s) + "'" for s in (list(possible_postomats) + list(fixed_points))]

    db = connections.DB()

    population_points_pd = db.get_by_filter("centers_mass", {"step": [step]})
    population_points = population_points_pd["id_center_mass"].unique()

    distances =  db.get_by_filter("distances_matrix_filter", {"step": [step], "object_id": possible_points})

    object_id_metro_pd = db.get_by_filter("all_metro_objects_data", {"object_type": ["'кластер входов в метро'"]})
    object_id_metro = object_id_metro_pd["object_id_metro"].to_list()

    distances_matrix_metro = db.get_by_filter("distances_matrix_metro_filter", {"object_id_metro": ["'" + str(s) + "'" for s in object_id_metro],
                                                                                "object_id": possible_points})

    pop_cent_mass_dict = population_points_pd.set_index('id_center_mass')['population'].to_dict()
    pop_metro_dict = object_id_metro_pd.set_index('object_id_metro')['population'].to_dict()

    population_dict = pop_cent_mass_dict | pop_metro_dict


    optimised_list, results = optimisation.optimize_by_solver(population_points,
                                                              possible_postomats,
                                                              fixed_points,
                                                              object_id_metro,
                                                              distances,
                                                              distances_matrix_metro,
                                                              quantity_postamats_to_place,
                                                              metro_weight,
                                                              population_dict,
                                                              precalculated_point=None,
                                                              **kwargs)

    output = json.dumps({'optimized_points': optimised_list,
                         'results': str(results)})

    return output


if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    db = connections.DB()
    all_objects_data = db.get_by_filter("all_objects_data", {"object_type": ["'киоск'"]})
    all_ponts = all_objects_data['object_id'].unique()
    fixed_points = all_ponts[:10]
    possible_postomats = all_ponts[10:]

    get_optimized_postomat_places(possible_postomats,
                                  fixed_points,
                                  quantity_postamats_to_place=1500,
                                  step=0.5,
                                  metro_weight=0.5)

