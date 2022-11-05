
"""Функции для работы с базой данных
"""
import os
from pathlib import Path
import json
from typing import Union
from warnings import warn
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import geopandas as gpd

PATH_TO_ROOT = Path(__file__).parent.parent.parent.parent


class DB:
    """Класс для подключения к БД и работы с ней
    """
    def __init__(self, db_config=None):
        
        if db_config is None:
            # По хорошему это надо хранить в секретах, но разбираться в секретнице клауда времени нет(
            self.host = os.environ['db_host']
            self.login = os.environ['db_login']
            self.passw = os.environ['db_passw']
            self.port = os.environ['db_port']
            self.db_name = os.environ['db_name']
        else:
            self.host = db_config['host']
            self.login = db_config['login']
            self.passw = db_config['passw']
            self.port = db_config['port']
            self.db_name = db_config['db_name']

        self.connection_for_engine = \
            f"postgresql+psycopg2://{self.login}:{self.passw}@{self.host}:{self.port}/{self.db_name}"
        self.postgis_created = False


    def __create_connection(self):
        connection = None
        try:
            connection = psycopg2.connect(
                database=self.db_name,
                user=self.login,
                password=self.passw,
                host=self.host,
                port=self.port,
            )
            print("Connection to PostgreSQL DB successful")
        except psycopg2.OperationalError as err:
            print(f"The error '{err}' occurred")
        return connection


    def load_to_bd(self,
                   df: Union[pd.DataFrame, gpd.GeoDataFrame],
                   table_name: str) -> None:
        """Загружаем датафрейм в базу

        Args:
            df (pd.DataFrame): датафрейм для загрузки
            table_name (str): куда сохраняем
        """

        engine = create_engine(self.connection_for_engine)
        incorrect_case_cols = [col for col in df.columns if col.isupper()]

        if incorrect_case_cols:
            warn(
                f'В названиях колонок есть uppercase: {incorrect_case_cols}'
                ' postgress имеет проблемы с uppercase, будет выполнен lower()'
                )
            df = df.copy()
            df.columns = [col.lower() for col in df.columns]
            warn(f'Новые названия колонок: {df.columns}')

        df.to_sql(table_name, con=engine, index=False, if_exists='replace')

        engine.dispose()


    def get_table_from_bd(self,
                          table_name: str) -> pd.DataFrame:
        """Выкачай всю таблицу

        Args:
            table_name (str): название табдицы

        Returns:
            pd.DataFrame: результат
        """
        query = f'select * from {table_name}'
        connection = self.__create_connection()
        df = pd.read_sql_query(query, connection)
        connection.close()
        return df


    def get_by_sql(self,
                   query: str) -> pd.DataFrame:
        """Кастомный запрос

        Args:
            query (str): запрос
            geo (bool): если хотим получить геопандас
            geom_col (str): название колонки с геоданными

        Returns:
            pd.DataFrame: результат
        """        
        connection = self.__create_connection()
        df = pd.read_sql_query(query, connection)
        connection.close()
        return df


    def execute_query(self, query: str) -> None:
        """Выполнить запрос на стороне базы данных

        Args:
            query (_type_): _description_

        Returns:
            _type_: _description_
        """
        conn1 = self.__create_connection()
        conn1.autocommit = True
        cursor = conn1.cursor()
        cursor.execute(query)
        conn1.commit()
        cursor.close()
        conn1.close()


    @staticmethod
    def __make_filter(column, value_list):
        # value_list_quates = ["'" + str(s) + "'" for s in value_list ]
        value_list_quates = [str(s) for s in value_list ]
        value_list_str = ", ".join(value_list_quates)
        return f"{column} IN ({str(value_list_str)})"


    def get_by_filter(self,
                      table_name: str,
                      filter_dict,
                      additional_filter: str =None) -> pd.DataFrame:

        connection = self.__create_connection()
        where_str = "AND ".join([self.__make_filter(k, filter_dict[k]) for k in filter_dict])
        query = f"select * from {table_name} WHERE {where_str}"
        if additional_filter is not None:
            query += f" AND {additional_filter}"
        df = pd.read_sql_query(query, connection)
        connection.close()
        return df



if __name__ == "__main__":
    CONFIG_PATH = os.path.join(PATH_TO_ROOT, 'db_config.json')

    with open(CONFIG_PATH, mode='r', encoding='utf-8') as f:
        db_config = json.load(f)

    db = DB(db_config)

    tmp = pd.DataFrame([[1, 2]], columns=['col1', 'col2'])
    db.load_to_bd(tmp, 'tmp')

    loaded1 = db.get_table_from_bd('tmp')
    print(loaded1)

    loaded2 = db.get_by_sql('SELECT * FROM tmp')
    print(loaded2)

    loaded3 = db.get_by_filter("tmp", {"col1": [1, 3]})
    print(loaded3)
