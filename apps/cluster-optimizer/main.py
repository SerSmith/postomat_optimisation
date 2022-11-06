import uvicorn

import json
from typing import List
from fastapi import FastAPI, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
import postamats.utils.helpers as h
from postamats.optimization.clustopt import kmeans_optimize_points
from postamats.utils import connections


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/get_kmeans_optimize_points")
def get_kmeans_optimize_points(object_type_filter_list: List[str] = Depends(h.parse_object_type_filter_list),
                                  district_type_filter_list: List[str] = Depends(h.parse_district_type_filter_list),
                                  adm_areat_type_filter_list: List[str] = Depends(h.parse_adm_areat_type_filter_list),
                                  fixed_points: List[str] = Depends(h.parse_list_fixed_points),
                                  banned_points: List[str] = Depends(h.parse_banned_points_list),
                                  quantity_postamats_to_place:int =Query(1500, description="Количество постаматов "),
                                  metro_weight:float =0.5,
                                  large_houses_priority:float =0.5,
                                  max_time:int =15):

    db = connections.DB()
    possible_postomats = h.make_points_lists(db,
                                            object_type_filter_list,
                                            district_type_filter_list,
                                            adm_areat_type_filter_list,
                                            banned_points)

    if fixed_points is None:
        fixed_points = []

    optimised_list = kmeans_optimize_points(possible_postomats,
                                            fixed_points,
                                            quantity_postamats_to_place,
                                            max_time=max_time,
                                            metro_weight=metro_weight,
                                            large_houses_priority=large_houses_priority,
                                            is_local_run=False)


    output = json.dumps({'optimized_points': optimised_list})

    return output

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
