import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import datetime



def optimisatin(args):
 
    POPULATION_POINTS = []

    POSTOMAT_PLACES = []

    DISTANSES = {}

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
