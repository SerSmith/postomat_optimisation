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

    return all_postamat_places.to_dict(orient="records")
