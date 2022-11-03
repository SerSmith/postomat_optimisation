"""Скрипты для заполнения базы данными
"""
import os
from typing import Dict
import json
from pathlib import Path
import pandas as pd
from postamats.utils import prepare_data, connections
from postamats.utils.connections import PATH_TO_ROOT
from postamats.global_constants import MANDATORY_COLS, OBJECT_ID_COL, OBJECT_TYPE_COL,\
    RAW_DMR_NAME, RAW_GIS_NAME, APARTMENT_HOUSES_NAME, ALL_OBJECTS_NAME,\
        KIOSKS_OT, MFC_OT, LIBS_OT, CLUBS_OT, SPORTS_OT,\
            INFRA_TABLES_NAMES_BY_OBJECTS, INFRA_GEODATA_COL, INFRA_NEEDED_COLS_BY_OBJECTS

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


    def prepare_apartment_houses(self) -> None:
        """подготовка и загрузка apartment_houses_all_data
        """

        apartment_houses = prepare_data.prepare_apartment_houses_data(
            self.get_prepared_dmr(), self.get_prepared_gis()
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

        self.__database.load_to_bd(prepared_dmr, RAW_DMR_NAME)
        self.__database.load_to_bd(prepared_gis, RAW_GIS_NAME)
        self.__database.load_to_bd(apartment_houses, APARTMENT_HOUSES_NAME)
        self.__database.load_to_bd(all_objects, ALL_OBJECTS_NAME)
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

if __name__ == '__main__':
    CONFIG_PATH = os.path.join(PATH_TO_ROOT, 'db_config.json')
    fill_db = FillDatabase(CONFIG_PATH)
    fill_db.read_prepare_load_all()
