import uvicorn
from typing import List
import json

from postamats.utils import connections
from postamats.optimization import optimisation
import postamats.utils.helpers as h

from fastapi import FastAPI, Depends, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="solverop",
              description="Сервис, оптимизирующий расстановку постоматов на основе cbc солвера ",
              version="2.0.0",
              сontact={
                    "name": "Кузнецов Сергей",
                    "email": "sergeysmith1995@ya.ru",
                })

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/get_optimized_postomat_places")
def get_optimized_postomat_places(object_type_filter_list: List[str] = Depends(h.parse_object_type_filter_list),
                                  district_type_filter_list: List[str] = Depends(h.parse_district_type_filter_list),
                                  adm_areat_type_filter_list: List[str] = Depends(h.parse_adm_areat_type_filter_list),
                                  fixed_points: List[str] = Depends(h.parse_list_fixed_points),
                                  banned_points: List[str] = Depends(h.parse_banned_points_list),
                                  quantity_postamats_to_place:int =Query(1500, description="Максимальное количество постаматов, которые можно поставить. (шт)"),
                                  step:float =Query(0.1, description="Размер клетки при агрегации (км)"),
                                  metro_weight:float =Query(0.5, description="Важность близости к мето по сравнению с близостью к домам. От 0(учитывать только дома) до 1(учитывать только станции метро). (безоазмерный)"),
                                  opt_time:int =Query(200, description="Максимальное время работы солвера, если за это время не удалось найти решение отличающееся от потенциально возможного оптимума менее чем на 0.01% то выдасться лучшее имеющееся решение. (сек)"),
                                  max_time:int =Query(15, description="Максимальное время, за которое человек готов дойти до постомата. Если человеку идти дольше, чем этот параметр, то мы считаем, что он не будет ходить в этот постамат. (мин)")):

    #  Зададим параметры для непосредственно солвера, полный список можно найти тут
    #  ratioGap - допустимое отклонение от потенциального оптимального решения
    #  sec - максимальное время работы солвераб после которого он выдадет ссамое  лучшее найденное решенее
    #  Если будет обнаружено, что имеющееся решениее отличается от потенциального оптимального меньше чем на ratioGap, 
    #  то солвер выдаст результат раньше
    step = 0.5
    kwargs = {'ratioGap': 0.0001, 'sec': opt_time}

    # Конектктимся к базе, данные берутся из переменных окружения(не было времени разбираться  ссекретницей клауда)
    db = connections.DB()

    possible_postomats = h.make_points_lists(db,
                                             object_type_filter_list,
                                             district_type_filter_list,
                                             adm_areat_type_filter_list,
                                             banned_points)
    if fixed_points is None:
        fixed_points = []
    
    possible_points = ["'" + str(s) + "'" for s in (list(possible_postomats) + list(fixed_points))]

    # precalculated_point = h.greedy_algo(db, possible_postomats, step, max_time * 60, quantity_postamats_to_place)
    precalculated_point = None

    population_points_pd = db.get_by_filter("centers_mass", {"step": [step]})
    population_points = population_points_pd["id_center_mass"].unique()

    distances =  db.get_by_filter("distances_matrix_filter",
                                 {"step": [step], "object_id": possible_points},
                                 additional_filter=f' walk_time<{max_time * 60}')

    object_id_metro_pd = db.get_by_filter("all_metro_objects_data", {"object_type": ["'кластер входов в метро'"]})
    object_id_metro = object_id_metro_pd["object_id_metro"].to_list()

    distances_matrix_metro = db.get_by_filter("distances_matrix_metro_filter", {"object_id_metro": ["'" + str(s) + "'" for s in object_id_metro],
                                                                                "object_id": possible_points},
                                                                                additional_filter=f' walk_time<{max_time * 60}')

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
                                                              precalculated_point,
                                                              **kwargs)
    print(len(optimised_list))
    print(str(results))

    output = json.dumps({'optimized_points': optimised_list,
                         'results': str(results)})

    return output



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, h11_max_incomplete_event_size=1640000)
    # res = get_optimized_postomat_places(object_type_filter_list=[],
    #                                     district_type_filter_list=[],
    #                                     adm_areat_type_filter_list=[],
    #                                     fixed_points=[],
    #                                     banned_points=[],
    #                                     quantity_postamats_to_place=1500,
    #                                     step=0.1,
    #                                     metro_weight=0.5,
    #                                     opt_time=250,
    #                                     max_time=15
    #                                     )
