

import json



def calculate_workload(center_mass_pd, distance_matrix_pd):

    only_nearest_points_min_dist = distance_matrix_pd.loc[distance_matrix_pd.groupby('id_center_mass').walk_time.idxmin()]

    only_nearest_points_min_dist_with_pop = only_nearest_points_min_dist.merge(center_mass_pd, on='id_center_mass')

    quantity_people_to_postomat = only_nearest_points_min_dist_with_pop.groupby('object_id').agg({'population': 'sum'}).reset_index()

    distance_till_nearest_postomat = only_nearest_points_min_dist_with_pop[['id_center_mass', 'walk_time']]


    return quantity_people_to_postomat, distance_till_nearest_postomat




def get_excel(postomat_points, meethod_name, center_mass_step):

    CONFIG_PATH = "/Users/sykuznetsov/Desktop/db_config.json"

    with open(CONFIG_PATH) as f:
        db_config = json.load(f)

    db = DB(db_config)

    postomat_points_str = ["'" + s + "'" for s in postomat_points ]
    data = db.get_by_filter("all_objects_data", {"object_id": postomat_points_str} )
    data['Модель расчета'] = meethod_name

    data = data.reset_index()

    quantity_people_to_postomat, _ = calculate_workload(postomat_points, center_mass_step)

    print(quantity_people_to_postomat)

    data = data.merge(quantity_people_to_postomat, on="object_id")

    print(data.columns)

    COLUMNS_RENAME = {
        'index': 'No п/п (номер по порядку)',
        'adm_area': 'Административный округ',
        'district': 'Район',
        'object_type': 'Тип объекта размещения',
        'lat': 'Координаты широта',
        'lon': 'Координаты долгота',
        'address': 'Адрес точки размещения'
    }
    data = data.rename(COLUMNS_RENAME)

    return_data = data[['No п/п (номер по порядку)',\
                        'Административный округ',\
                        'Район',\
                        'Тип объекта размещения',\
                        'Координаты широта',\
                        'Координаты долгота',\
                        'Адрес точки размещения',\
                        'Модель расчета',
                        'Количество людей, для которых этот постомат ближайший']]
    
    return return_data

