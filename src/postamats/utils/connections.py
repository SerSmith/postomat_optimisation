
"""Функции для работы с базой данных
"""
import json
from warnings import warn
from sqlalchemy import create_engine
import psycopg2
import pandas as pd


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
        incorrect_case_cols = [col for col in df.columns if col.isupper()]

        if incorrect_case_cols:
            warn(
                f'В названиях колонок есть uppercase: {incorrect_case_cols}'
                ' postgress имеет проблемы с uppercase, будет выполнен lower()'
                )
            df = df.copy()
            df.columns = [col.lower() for col in df.columns]
            warn(f'Новые названия колонок: {df.columns}')

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


if __name__ == "__main__":
    CONFIG_PATH = "/Users/sykuznetsov/Desktop/db_config.json"

    with open(CONFIG_PATH) as f:
        db_config = json.load(f)

    db = DB(db_config)

    tmp = pd.DataFrame([[1, 2]], columns=['A', 'B'])
    db.load_to_bd(tmp, 'tmp')

    loaded1 = db.get_table_from_bd('tmp')

    print(loaded1)

    loaded2 = db.get_by_sql('SELECT * FROM tmp')

    print(loaded2)
