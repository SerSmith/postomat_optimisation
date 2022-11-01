import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import datetime

import json
from connections import DB


def optimisatin(fixed_points, possible_postomats):

    CONFIG_PATH = "/Users/sykuznetsov/Desktop/db_config.json"

    with open(CONFIG_PATH) as f:
        db_config = json.load(f)

    db = DB(db_config)
 
    population_points_pd = db.get_by_filter("tst_center_mass_29102022", {"step": 1})
    POPULATION_POINTS = population_points_pd["object_id"].unique()

    POSTOMAT_PLACES = fixed_points + possible_postomats

    DISTANSES = db.get_by_filter("fake_distances_matrix", {"object_id": possible_postomats})

    # Создание конкретной модели pyomo
    model = pyo.ConcreteModel()

    # Переменные
    model.has_postomat = pyo.Var(POSTOMAT_PLACES, within=pyo.Binary, initialize=0)

    model.nearest_point_time = pyo.Var(POPULATION_POINTS)

    #Ограничения

    # Одновременно не более MAX_SIMULT_PROMO акций
    def con_nearest_point_time(model, population_point, postomat_place):
        return model.nearest_point_time[population_point] >= DISTANSES[population_point, postomat_place] * model.has_postomat[postomat_place]

    model.con_nearest_point_time = pyo.Constraint(POPULATION_POINTS, POSTOMAT_PLACES ,rule=con_nearest_point_time)

    # # Целевая
    model.OBJ = pyo.Objective(expr=sum(model.nearest_point_time[p] for p in POPULATION_POINTS), sense=pyo.minimize)

    # opt = Solver olve(instance, logfile=SOLVE_LOG, solnfile=SOLNFILE)
