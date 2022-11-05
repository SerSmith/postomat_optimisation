"""Скрипты для заполнения базы данными
"""
import os
from typing import Dict, List, Optional
import json
import time
import psycopg2
import pandas as pd
import numpy as np
from tqdm import tqdm
from postamats.utils import prepare_data, connections, helpers
from postamats.utils.connections import DB, PATH_TO_ROOT
from postamats.global_constants import MANDATORY_COLS, OBJECT_ID_COL, OBJECT_TYPE_COL,\
    RAW_DMR_NAME, RAW_GIS_NAME, APARTMENT_HOUSES_NAME, ALL_OBJECTS_NAME, CENTER_MASS_NAME,\
        KIOSKS_OT, MFC_OT, LIBS_OT, CLUBS_OT, SPORTS_OT,\
            INFRA_TABLES_NAMES_BY_OBJECTS, INFRA_GEODATA_COL, INFRA_NEEDED_COLS_BY_OBJECTS,\
                LIST_STEP, MOSCOW_POPULATION_NAME, LATITUDE_COL, LONGITUDE_COL, CENTER_MASS_ID_COL
tqdm.pandas()


def get_full_path_from_relative(relative_path: str) -> str:
    """Превращает относительный путь в абсолютный
    Нужна для передачи путей, которые хранятся в константах,
     как относительные в пандас и проч.

    Args:
        relative_path (str): относительный путь к файлу

    Returns:
        str: абсолютный путь
    """
    return os.path.join(PATH_TO_ROOT, relative_path)


def get_query_from_file(sql_name: str) -> str:
    """Берет из файла sql запрос

    Args:
        sql_name (str): имя файла sql из папки sqls

    Returns:
        str: sql запрос
    """
    sqls_folder = os.path.join(PATH_TO_ROOT, 'sqls')
    path = os.path.join(sqls_folder, sql_name)
    with open(path, mode="r", encoding='utf-8') as sql_file:
        content = sql_file.read()
    return content


def calc_distances_matrix_database(config_path: str) -> None:
    """!Внимание, функция инициирует перасчет таблицы distances_matrix в БД. Продолжаем?!
    Создает и заполняет матрицу расстояний
     в БД, расчеты также происходят на стороне БД

    Args:
        config_path (str): путь к json с реквизитами базы данных
    """
    var = input('Внимание, функция инициирует перасчет таблицы distances_matrix в БД.'
                ' Продолжаем? (Y/n) ')
    if var != 'Y':
        print('aborted')
        return
    var = input('Данные в distances_matrix будут перерассчитаны,'
                ' расчет ведется на стороне БД. Вы уверены? (Y/n) ')
    if var != 'Y':
        print('aborted')
        return

    with open(config_path, mode='r', encoding='utf-8') as db_file:
        db_config = json.load(db_file)

    database = DB(db_config)
    # скрипт для создания функции расчета расстояний
    func_ddl = get_query_from_file('calculate_distance_ddl.sql')
    # скрипт для создания таблицы расстояний
    dist_ddl = get_query_from_file('distances_matrix_ddl.sql')
    # скрипт для заполнения таблицы расстояний
    dist_etl = get_query_from_file('distances_matrix.sql')
    database.execute_query(func_ddl)
    # мы не пересоздаем таблицу, если она уже готова
    try:
        database.execute_query(dist_ddl)
    except psycopg2.errors.DuplicateTable as error:# pylint: disable=no-member
        print ("Oops! An exception has occured:", error)
        print ("Exception TYPE:", type(error))
    database.execute_query(dist_etl)
    print('Команды для перерасчета отправлены в БД')


def calc_distances_matrix_locally(config_path: str,
                                 save_pickle: bool=False,
                                 include_houses: bool=False,
                                 concat_slices: bool=True,
                                 table1: str=ALL_OBJECTS_NAME,
                                 table2: str=CENTER_MASS_NAME,
                                 pickle_name: str='distances_matrix.pickle',
                                 id1_col: str=OBJECT_ID_COL,
                                 id2_col: str=CENTER_MASS_ID_COL,
                                 meters_to_sec_coef: float=1.152,
                                 max_slice_size: int=10**7) -> pd.DataFrame:
    """Создает и заполняет матрицу расстояний локально
     расчеты происходят локально на таблицах, выгружаемых из БД

    Args:
        config_path (str): путь к json с реквизитами базы данных
        save_pickle (bool, optional): Сохранять данные в data/temporary/distances_matrix.pickle
         сразу после расчета. Defaults to False.
        include_houses (bool, optional): включаем ли мы многоквартирные дома в расчеты.
         Defaults to False.
        concat_slices: (bool, optional): объединяем ли результат в единый датафрейм (True)
         или возвращаем списком срезов (False). Defaults to True.
        max_slice_size (int, optional): максимальный размер среза картезианова датафрейма (строк),
         оптимально пролезающий в память. Defaults to 10**7.
    """
    with open(config_path, mode='r', encoding='utf-8') as db_file:
        db_config = json.load(db_file)

    database = DB(db_config)
    print(f'Загружаем из БД {table1}')
    data1 = database.get_table_from_bd(table1)
    print(f'Загружаем из БД {table2}')
    data2 = database.get_table_from_bd(table2)

    if not include_houses:
        print('include_houses = False,'
        ' многоквартирные дома будут исключены из рассчета расстояний.')
        data1 = data1[data1['object_type'] != 'многоквартирный дом']

    coords1 = data1[[id1_col, LATITUDE_COL, LONGITUDE_COL]]
    coords2 = data2[[id2_col, LATITUDE_COL, LONGITUDE_COL]]

    size1, size2 = coords1.shape[0], coords2.shape[0]
    cross_size = size1 * size2
    n_splits = int( np.ceil(cross_size / max_slice_size) )
    max_slice_size = int( np.ceil(size1 / n_splits) )

    slices_gen = helpers.df_generator(coords1, max_slice_size)
    slices_list = []
    print(f'Размер картезианова датафрейма: {size1} x {size2} = {cross_size}.\n'
          f'Датафрейм будет разбит на {n_splits} частей')
    print('Получаем картезиановы датафреймы для каждой части:')
    time.sleep(.5)
    for df_slice in tqdm(slices_gen, total=n_splits):
        slices_list.append(df_slice.merge(coords2, how='cross'))
    print('Считаем расстояния:')
    time.sleep(.5)
    for i, df_slice in tqdm(enumerate(slices_list), total=n_splits):
        slices_list[i]['distance'] = helpers.haversine_vectorized(
            df_slice,
            'lat_x',
            'lon_x',
            'lat_y',
            'lon_y'
            ).round(0).astype(int)
        slices_list[i]['walk_time'] = (
            slices_list[i]['distance'] * meters_to_sec_coef
            ).round(0).astype(int)
        slices_list[i] = slices_list[i].drop(
            columns=[
                'lat_x',
                'lon_x',
                'lat_y',
                'lon_y'
        ])

    all_dists = slices_list
    if concat_slices:
        print('Объединяем срезы ... ', end='')
        all_dists = pd.concat(slices_list, ignore_index=True)
        print('успешно')
    if save_pickle:
        filepath = os.path.join(PATH_TO_ROOT, 'data', 'temporary', pickle_name)
        print(f'Сохраняем pickle в {filepath} ... ', end='')
        all_dists.to_pickle(
            filepath, protocol=4
            )
        print('успешно')
    return all_dists


class MakeCenterMass():
    """Класс для расчета и заливки в БД табличку с координатами
     центра масс сектора территории по населению этой территории
    """
    def __init__(self,
                config_path: str) -> None:
        """_summary_

        Args:
            config_path (str): путь к json с реквизитами базы данных
        """

        with open(config_path, mode='r', encoding='utf-8') as db_file:
            db_config = json.load(db_file)
        self.__database = connections.DB(db_config)
        self.center_mass = pd.DataFrame()
        self.apartment_houses = pd.DataFrame()


    def load_apartment_houses(self) -> None:
        """Загружает табличку с данными о населении и метоположении
        жилых домов
        """
        self.apartment_houses = self.__database.get_table_from_bd(APARTMENT_HOUSES_NAME)


    def make_center_mass(self,
                         list_step: Optional[List[float]]=None) -> None:
        """Рассчитывает табличку с координатами
        центра масс сектора территории по населению этой территории

        Args:
            config_path (str): путь к json с реквизитами базы данных
            list_step (List[float], optional): список размеров величины шага в км в сетке,
             которую мы накладываем на дома. Defaults to None.
        """
        if list_step is None:
            list_step = LIST_STEP

        apartment_houses = self.get_apartment_houses()

        # расчет по данным о плотности населения Москвы в целом
        # apartment_houses['population'] = apartment_houses['total_area_gis']/22

        lat_km, lon_km = helpers.find_degreee_to_distance(apartment_houses)
        distance_to_degree = {'lat': 1/lat_km, 'lon': 1/lon_km}

        results_list = []
        for step in list_step:
            df_result_step = helpers.make_net_with_center_mass(apartment_houses,
                                                               step,
                                                               distance_to_degree)
            results_list.append(df_result_step.copy())

        df_result = pd.concat(results_list)
        self.center_mass = df_result.dropna()


    def make_center_mass_load_to_db(self,
                                    list_step: Optional[List[float]]=None) -> None:
        """Подгружает из базы данные о домах и их населении,
        рассчитывает табличку с координатами центра масс сектора территории
         по населению этой территории и грузит её в базу данных
        """
        self.load_apartment_houses()
        self.make_center_mass(list_step=list_step)
        self.load_to_db()


    def load_to_db(self):
        """Грузит self.center_mass в БД
        """
        center_mass = self.get_center_mass()
        self.__database.load_to_bd(center_mass, CENTER_MASS_NAME)


    def get_center_mass(self) -> pd.DataFrame:
        """getter
        """
        if self.center_mass.shape[0] == 0:
            raise ValueError('center_mass is absend, run make_center_mass')
        return self.center_mass


    def get_apartment_houses(self):
        """getter
        """
        if self.apartment_houses.shape[0] == 0:
            raise ValueError('apartment_houses is absend, run load_apartment_houses')
        return self.apartment_houses


class FillDatabase():
    """Класс служит для заполнения базы данными
    """

    def __init__(self,
                config_path: str,
                data_root_path=os.path.join(PATH_TO_ROOT, 'data')) -> None:
        """_summary_

        Args:
            config_path (str): путь к json с реквизитами базы данных
            data_root_path (_type_, optional): путь к папке, где по подпапкам разложены
             сырые данные. Defaults to os.path.join(PATH_TO_ROOT, 'data').
        """
        self.data_root_path = data_root_path
        self.prepared_dmr = pd.DataFrame()
        self.prepared_gis = pd.DataFrame()
        self.apartment_houses = pd.DataFrame()
        self.infrastructure_objects = {}
        self.moscow_population = pd.DataFrame()
        self.all_objects = pd.DataFrame()
        with open(config_path, mode='r', encoding='utf-8') as db_file:
            db_config = json.load(db_file)
        self.__database = connections.DB(db_config)


    def read_prepare_dmr(self,
                         data_folder: str='dmr') -> None:
        """подготовка и загрузка raw_dmr_houses_data
        Где взять сырые данные:

        - нужно зайти по ссылке или найти поиском на https://data.mos.ru
        - найти датасет "Адресный реестр объектов недвижимости города Москвы"
        - скачать его в формате json
        - положить в папку postamat_optimisation/data/dmr

        Args:
            data_folder (str): имя папки с файлом данных json (файл в папке должен быть один)
        """
        path = os.path.join(self.data_root_path, data_folder)
        dmr_file = prepare_data.get_filenames_from_dir(path, mandatory_substr='json')

        if len(dmr_file) == 0:
            raise FileNotFoundError(f'В папке {data_folder} нет json фалов')
        if len(dmr_file) > 1:
            raise ValueError(f'В папке {data_folder} должен быть только один json файл')

        dmr_file = dmr_file[0]

        prepared_dmr = prepare_data.prepare_dmr_houses_data(
            dmr_file, geo=False
            )
        self.prepared_dmr = prepared_dmr


    def read_prepare_gis(self,
                         data_folder: str='gis') -> None:
        """подготовка и загрузка raw_gis_houses_data
        Где взять сырые данные:

        - нужно зайти по ссылке https://dom.gosuslugi.ru/#!/houses
         или найти поиском "Реестр объектов жилищного фонда"
        - выбрать в Субъект РФ "Москва"
        - нажать "Найти" внизу справа
        - нажать "Скачать" вверху справа
        - скачается папка/архив с несколькими csv
        - положить папку postamat_optimisation/data, назвать gis

        Args:
            data_folder (str): имя папки с csv
        """
        path = os.path.join(self.data_root_path, data_folder)
        prepared_gis = prepare_data.prepare_gis_houses_data(path)
        self.prepared_gis = prepared_gis


    def read_moscow_population(self,
                               file_name: str='moscow_population.csv'):
        """Читает данные о населении Москвы по муниципальным округам
         Взяты отсюда: https://gogov.ru/population-ru/msk
        """
        self.moscow_population = \
            pd.read_csv(os.path.join(self.data_root_path, file_name))


    def prepare_apartment_houses(self) -> None:
        """подготовка и загрузка apartment_houses_all_data
        """

        apartment_houses = prepare_data.prepare_apartment_houses_data(
            self.get_prepared_dmr(), self.get_prepared_gis()
            )
        moscow_population = self.get_moscow_population()
        apartment_houses = prepare_data.calc_population(
            moscow_population, apartment_houses
        )

        self.apartment_houses = apartment_houses


    def read_prepare_infrastructure(
        self,
        kiosks_folder: str='kiosks',
        mfc_folder: str='mfc',
        libs_folder: str='libs',
        clubs_folder: str='clubs',
        sports_folder: str='sports'
        ) -> None:
        """Исходные данные скачиваем в формате json из:
        - Нестационарные торговые объекты по реализации печатной продукции:
         https://data.mos.ru/opendata/2781
        - Нестационарные торговые объекты:
         https://data.mos.ru/opendata/619
        - Многофункциональные центры предоставления государственных и муниципальных услуг
         https://data.mos.ru/opendata/-mnogofunktsionalnye-tsentry-predostavleniya-gosudarstvennyh-uslug
        - Библиотеки города: https://data.mos.ru/opendata/7702155262-biblioteki
        - Дома культуры и клубы: https://data.mos.ru/opendata/7702155262-doma-kultury-i-kluby
        - Спортивные объекты города Москвы:
         https://data.mos.ru/opendata/7708308010-sportivnye-obekty-goroda-moskvy

        Args:
            kiosks_folder (str, optional): имя папки. Defaults to 'kiosks'.
            mfc_folder (str, optional): имя папки. Defaults to 'mfc'.
            libs_folder (str, optional): имя папки. Defaults to 'libs'.
            clubs_folder (str, optional): имя папки. Defaults to 'clubs'.
            sports_folder (str, optional): имя папки. Defaults to 'sports'.

        Raises:
            FileNotFoundError: _description_
        """

        objects_folders = {
            KIOSKS_OT: os.path.join(self.data_root_path, kiosks_folder),
            MFC_OT: os.path.join(self.data_root_path, mfc_folder),
            LIBS_OT: os.path.join(self.data_root_path, libs_folder),
            CLUBS_OT: os.path.join(self.data_root_path, clubs_folder),
            SPORTS_OT: os.path.join(self.data_root_path, sports_folder)
        }

        objects_dfs = {}

        for obj, folder in objects_folders.items():
            files_list = prepare_data.get_filenames_from_dir(
                folder, mandatory_substr='json'
                )
            if len(files_list) == 0:
                raise FileNotFoundError(f'В папке {folder} нет json фалов')

            data_list = []

            for f_path in files_list:
                data = pd.read_json(f_path, encoding='cp1251')\
                        .dropna(axis=1, how='all')# pylint: disable=[no-member]

                data_list.append( data.copy() )

            objects_dfs[obj] = pd.concat(data_list, ignore_index=True)

        for obj_type, obj_data in objects_dfs.items():
            print(f'{obj_type}')
            prepared_data = prepare_data.prepare_infrastructure_objects(
                obj_data,
                INFRA_GEODATA_COL,
                obj_type,
                needed_cols=INFRA_NEEDED_COLS_BY_OBJECTS[obj_type]
                )
            self.infrastructure_objects[obj_type] = prepared_data.copy()


    def prepare_all_objects(self):
        """Финальная сборка таблички с данными о потенциальных местах размещения постаматов

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        if not self.infrastructure_objects:
            raise ValueError('infrastructure_objects is absend, run read_prepare_infrastructure')
        if self.apartment_houses.shape[0] == 0:
            raise ValueError('apartment_houses is absend, run prepare_apartment_houses')
        all_infra_list = list(self.infrastructure_objects.values())
        all_objects = pd.concat([self.apartment_houses] + all_infra_list, ignore_index=True)
        all_objects = all_objects[MANDATORY_COLS].copy()
        self.all_objects = all_objects
        if all_objects[OBJECT_ID_COL].duplicated().sum() != 0:
            raise ValueError(f'В all_objects найдены дубли по {OBJECT_ID_COL}')
        types_count = all_objects[OBJECT_TYPE_COL].unique().shape[0]
        if types_count != 6:
            raise ValueError(f'В all_objects должно быть 6 уникальных типов, найдено {types_count}')


    def read_prepare_all(self):
        """Готовит все таблички сразу
        нужно, чтобы соблюдалась дефолтная структура папок
        """
        self.read_prepare_dmr()
        self.read_prepare_gis()
        self.read_moscow_population()
        self.prepare_apartment_houses()
        self.read_prepare_infrastructure()
        self.prepare_all_objects()


    def load_all_to_db(self):
        """Грузит всё в базу данных
        """
        prepared_dmr = self.get_prepared_dmr()
        prepared_gis = self.get_prepared_gis()
        apartment_houses = self.get_apartment_houses()
        infrastructure_objects = self.get_infrastructure_objects()
        all_objects = self.get_all_objects()
        moscow_population = self.get_moscow_population()

        self.__database.load_to_bd(prepared_dmr, RAW_DMR_NAME)
        self.__database.load_to_bd(prepared_gis, RAW_GIS_NAME)
        self.__database.load_to_bd(apartment_houses, APARTMENT_HOUSES_NAME)
        self.__database.load_to_bd(all_objects, ALL_OBJECTS_NAME)
        self.__database.load_to_bd(moscow_population, MOSCOW_POPULATION_NAME)
        for obj_type, obj_data in infrastructure_objects.items():
            self.__database.load_to_bd(obj_data, INFRA_TABLES_NAMES_BY_OBJECTS[obj_type])


    def read_prepare_load_all(self):
        """Выполняет все операции чтения и загрузки в БД
        нужно, чтобы соблюдалась дефолтная структура папок
        """
        print('read_prepare_all started')
        self.read_prepare_all()
        print('read_prepare_all finished\n\nload_all_to_db started')
        self.load_all_to_db()
        print('load_all_to_db finished')


    def get_prepared_dmr(self) -> pd.DataFrame:
        """getter
        """
        if self.prepared_dmr.shape[0] == 0:
            raise ValueError('prepared_dmr is absend, run read_prepare_dmr')
        return self.prepared_dmr


    def get_prepared_gis(self) -> pd.DataFrame:
        """getter
        """
        if self.prepared_gis.shape[0] == 0:
            raise ValueError('prepared_gis is absend, run read_prepare_gis')
        return self.prepared_gis


    def get_apartment_houses(self) -> pd.DataFrame:
        """getter
        """
        if self.apartment_houses.shape[0] == 0:
            raise ValueError('apartment_houses is absend, run prepare_apartment_houses')
        return self.apartment_houses


    def get_infrastructure_objects(self) -> Dict[str, pd.DataFrame]:
        """getter
        """
        if not self.infrastructure_objects:
            raise ValueError('infrastructure_objects is absend, run read_prepare_infrastructure')
        return self.infrastructure_objects


    def get_all_objects(self) -> pd.DataFrame:
        """getter
        """
        if self.all_objects.shape[0] == 0:
            raise ValueError('all_objects is absend, run prepare_all_objects')
        return self.all_objects


    def get_moscow_population(self) -> pd.DataFrame:
        """getter
        """
        if self.moscow_population.shape[0] == 0:
            raise ValueError('moscow_population is absend, run read_moscow_population')
        return self.moscow_population


if __name__ == '__main__':
    CONFIG_PATH = os.path.join(PATH_TO_ROOT, 'db_config.json')
    fill_db = FillDatabase(CONFIG_PATH)
    fill_db.read_prepare_load_all()

    mcm = MakeCenterMass(CONFIG_PATH)
    mcm.make_center_mass_load_to_db()

    calc_distances_matrix_database(CONFIG_PATH)
