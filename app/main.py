from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/say_hi")
def get_possible_postomat_places():

    return {"Hello": "World"}
