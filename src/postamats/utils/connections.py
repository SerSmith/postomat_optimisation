
"""Функции для работы с базой данных
"""
import json
from typing import Union
from warnings import warn
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import geopandas as gpd
from postamats.global_constants import DB_GEODATA_COL


class DB:
    """Класс для подключения к БД и работы с ней
    """
    def __init__(self, db_config):

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



    def __create_extension_postgis(self):
        """Создаем EXTENSION postgis
         чтобы заливать геоданные
        """
        conn1 = self.__create_connection()
        conn1.autocommit = True
        cursor = conn1.cursor()
        cursor.execute("CREATE EXTENSION postgis;")
        conn1.commit()
        cursor.close()
        conn1.close()
        self.postgis_created = True


    def load_to_bd(self,
                   df: Union[pd.DataFrame, gpd.GeoDataFrame],
                   table_name: str,
                   geo: bool=False) -> None:
        """Загружаем датафрейм в базу

        Args:
            df (pd.DataFrame): датафрейм для загрузки
            table_name (str): куда сохраняем
            geo (bool): если хотим залить геоданные из геопандас
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

        if geo:
            if not self.postgis_created:
                self.__create_extension_postgis()
                print('EXTENSION postgis created')
            df.to_postgis(table_name, con=engine, index=False, if_exists ='replace')
        else:
            df.to_sql(table_name, con=engine, index=False, if_exists ='replace')

        engine.dispose()


    def get_table_from_bd(self,
                          table_name: str,
                          geo: bool=False,
                          geom_col: str=DB_GEODATA_COL) -> pd.DataFrame:
        """Выкачай всю таблицу

        Args:
            table_name (str): название табдицы
            geo (bool): если хотим получить геопандас
            geom_col (str): название колонки с геоданными

        Returns:
            pd.DataFrame: результат
        """
        query = f'select * from {table_name}'
        connection = self.__create_connection()
        if geo:
            df = gpd.read_postgis(query, connection, geom_col=geom_col)
        else:
            df = pd.read_sql_query(query, connection)
        connection.close()
        return df


    def get_by_sql(self,
                   query: str,
                   geo: bool=False,
                   geom_col: str=DB_GEODATA_COL) -> pd.DataFrame:
        """Кастомный запрос

        Args:
            query (str): запрос
            geo (bool): если хотим получить геопандас
            geom_col (str): название колонки с геоданными

        Returns:
            pd.DataFrame: результат
        """        
        connection = self.__create_connection()
        if geo:
            df = gpd.read_postgis(query, connection, geom_col=geom_col)
        else:
            df = pd.read_sql_query(query, connection)
        connection.close()
        return df


    @staticmethod
    def __make_filter(column, value_list):
        # value_list_quates = ["'" + str(s) + "'" for s in value_list ]
        value_list_quates = [str(s) for s in value_list ]
        value_list_str = ", ".join(value_list_quates)
        return f"{column} IN ({str(value_list_str)})"


    def get_by_filter(self,
                      table_name: str,
                      filter_dict,
                      geo: bool=False,
                      geom_col: str=DB_GEODATA_COL) -> pd.DataFrame:

        connection = self.__create_connection()
        where_str = "AND ".join([self.__make_filter(k, filter_dict[k]) for k in filter_dict])
        query = f"select * from {table_name} WHERE {where_str}"
        if geo:
            df = gpd.read_postgis(query, connection, geom_col=geom_col)
        else:
            df = pd.read_sql_query(query, connection)
        connection.close()
        return df



if __name__ == "__main__":
    CONFIG_PATH = "/Users/sykuznetsov/Desktop/db_config.json"

    with open(CONFIG_PATH) as f:
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

    
