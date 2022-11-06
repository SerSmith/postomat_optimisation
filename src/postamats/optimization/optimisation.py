import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import datetime

import json



def get_chosen_postomats(model):
    solution_dict = model.has_postomat.extract_values()
    solution_pd = pd.DataFrame(solution_dict.items(), columns=['object_id', 'place_postomat'])
    return solution_pd.loc[solution_pd['place_postomat'] > 0, 'object_id'].to_list()

def optimize_by_solver(population_points,
                       possible_postomats,
                       fixed_points,
                       object_id_metro_list,
                       distances,
                       distanses_metro,
                       quantity_postamats_to_place,
                       metro_weight,
                       population_dict,
                       precalculated_points=None,
                       **kwargs):

    postomat_places = list(fixed_points) + list(possible_postomats)

    distances_dict = {(id_center_mass, postomat_place_id): walk_time for _, postomat_place_id, id_center_mass , _, _, walk_time in distances.itertuples()}
    distances_metro_dict = {(object_id_metro, object_id): walk_time for _, object_id, object_id_metro ,  _, walk_time in distanses_metro.itertuples()}

    # Создание конкретной модели pyomo
    model = pyo.ConcreteModel()

    # Переменные
    model.has_postomat = pyo.Var(postomat_places, within=pyo.Binary, initialize=0)

    if precalculated_points is not None:
        for point in precalculated_points:
            model.has_postomat[point] = 1


    for fixed_point in fixed_points:
        model.has_postomat[fixed_point].fix(1)

    model.center_mass_time_to_nearest_postamat = pyo.Var(population_points, within=pyo.NonNegativeReals)

    #Ограничения

    # Одновременно не более MAX_SIMULT_PROMO акций
    def con_center_mass_time_to_nearest_postamat(model, *data):
        _, id_center_mass, postomat_place_id = data
        return model.center_mass_time_to_nearest_postamat[id_center_mass] >= distances_dict[(id_center_mass, postomat_place_id)] * model.has_postomat[postomat_place_id]

    model.con_center_mass_time_to_nearest_postamat = pyo.Constraint( list(distances[['id_center_mass',	'object_id']].itertuples()) ,rule=con_center_mass_time_to_nearest_postamat)


    model.metro_time_to_nearest_postamat = pyo.Var(object_id_metro_list, within=pyo.NonNegativeReals)
        # Одновременно не более MAX_SIMULT_PROMO акций
    def con_metro_time_to_nearest_postamat(model, *data):
        _, object_id_metro, postomat_place_id = data
        return model.metro_time_to_nearest_postamat[object_id_metro] >= distances_metro_dict[(object_id_metro, postomat_place_id)] * model.has_postomat[postomat_place_id]

    model.con_metro_time_to_nearest_postamat = pyo.Constraint( list(distanses_metro[['object_id_metro',	'object_id']].itertuples()) ,rule=con_metro_time_to_nearest_postamat)


    model.needed_postamats = pyo.Constraint(expr=sum([model.has_postomat[p] for  p in postomat_places])  == quantity_postamats_to_place)

    sum_center_mass = sum(model.center_mass_time_to_nearest_postamat[p] * population_dict[p] for p in population_points)
    sum_metro = sum(model.metro_time_to_nearest_postamat[p] * population_dict[p] for p in object_id_metro_list)
    # # Целевая
    model.OBJ = pyo.Objective(expr=((1 - metro_weight) * sum_center_mass + (metro_weight) * sum_metro), sense=pyo.minimize)
    # minimize

    # , executable="/usr/local/Cellar/cbc/2.10.8/bin/cbc"
    opt = SolverFactory('cbc')

    for key in kwargs:
        opt.options[key] = kwargs[key]

    results = opt.solve(model)


    optimised_list = get_chosen_postomats(model)

    optimised_list_no_fixed = list(set(optimised_list).difference(set(fixed_points)))

    return optimised_list_no_fixed, results


