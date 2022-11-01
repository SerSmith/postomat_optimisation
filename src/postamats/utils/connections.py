
"""Функции для работы с базой данных
"""
import json
import os
from sqlalchemy import create_engine
import psycopg2
import pandas as pd


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


    def load_to_bd(self, df: pd.DataFrame, table_name: str) -> None:
        """Загружаем датафрейм в базу

        Args:
            df (pd.DataFrame): датафрейм для загрузки
            table_name (str): куда сохраняем
        """
        engine = create_engine(self.connection_for_engine)
        df.to_sql(table_name, con=engine, index=False, if_exists ='replace')
        engine.dispose()


    def get_table_from_bd(self, table_name:str) -> pd.DataFrame:
        """Выкачай всю таблицу

        Args:
            table_name (str): название табдицы

        Returns:
            pd.DataFrame: результат
        """        
        connection = self.__create_connection()
        df = pd.read_sql_query(f'select * from {table_name}', connection)
        connection.close()
        return df


    def get_by_sql(self, query: str) -> pd.DataFrame:
        """Кастомный запрос

        Args:
            query (str): запрос

        Returns:
            pd.DataFrame: результат
        """        
        connection = self.__create_connection()
        df = pd.read_sql_query(query, connection)
        connection.close()
        return df


    @staticmethod
    def __make_filter(column, value_list):
        # value_list_quates = ["'" + str(s) + "'" for s in value_list ]
        value_list_quates = [str(s) for s in value_list ]
        value_list_str = ", ".join(value_list_quates)
        return f"{column} IN ({str(value_list_str)})"


    def get_by_filter(self, table_name: str, filter_dict) -> pd.DataFrame:

        connection = self.__create_connection()
        where_str = "AND ".join([self.__make_filter(k, filter_dict[k]) for k in filter_dict])
        query = f"select * from {table_name} WHERE {where_str}"
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

    