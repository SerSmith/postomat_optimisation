"""Константы для функций проекта
"""

# названия промежуточных табличек с обработанными данными из
# https://data.mos.ru/opendata/60562/data/table?versionNumber=3&releaseNumber=823
# https://dom.gosuslugi.ru/#!/houses
# которые используются для подготовки финальных табличек для заливки в базу
RAW_GIS_NAME = 'raw_gis_houses_data'
RAW_DMR_NAME = 'raw_dmr_houses_data'
# название таблички с данными жилых домов
# построенной на базе RAW_GIS_NAME и RAW_DMR_NAME
APARTMENT_HOUSES_NAME = 'apartment_houses_all_data'
# таблички инфраструктурных объектов
KIOSKS_NAME = 'kiosks_all_data'
MFC_NAME = 'mfc_all_data'
LIBS_NAME = 'libs_all_data'
CLUBS_NAME = 'clubs_all_data'
SPORTS_NAME = 'sports_all_data'
# табличка всех объектов, подходящих для размещения постаматов
ALL_OBJECTS_NAME = 'all_objects_data'

NAN_VALUES = ['', '-', '*']

# как во всех наших будут называться колонки
OBJECT_ID_COL = 'object_id'
OBJECT_TYPE_COL = 'object_type'
ADDRESS_COL = 'address'
DISTRICT_COL = 'district'
ADM_AREA_COL = 'adm_area'


# для сырой таблички с данными ГИС ОЖФ свой айдишник
OBJECT_ID_GIS_COL = 'object_id_gis'

# как мы назовем колонку с данными о широте
LATITUDE_COL = 'lat'
# как мы назовем колонку с данными о долготе
LONGITUDE_COL = 'lon'

# название колонки с геоданными в таблицах data.mos.ru
DMR_GEODATA_COL = 'GEODATA'
# как будет называться колонка с геоданными в базе данных
DB_GEODATA_COL = 'geodata'
# название колонки с геоданными в сырых данных инфраструктурных объектов
INFRA_GEODATA_COL = 'geoData'

# название колонок с кадастровыми номерами в таблице
# Адресного реестра объектов недвижимости города Москвы data.mos.ru
DMR_KAD_NUM_COLS = ['KAD_N', 'KAD_ZU']

# мэппинг колонок Адресного реестра объектов недвижимости города Москвы
# которые мы заливаем в БД
DMR_COLS_MAP = {
    # Тип объекта адресации
    # «здание», «сооружение», «владение», «домовладение»,
    # «объект незавершенного строительства», «земельный участок»,
    # «помещение», «комната», «объект адресации – помещение»,
    # «объект права», «объект адресации – комната».
    'OBJ_TYPE': 'category',
    'ONTERRITORYOFMOSCOW': 'on_moscow_territory',
    'ADDRESS': ADDRESS_COL,
    'SIMPLE_ADDRESS': 'simple_address',
    # Наименование элемента планировочной структуры или улично-дорожной сети
    'P7': 'street',
    # Тип номера дома, владения, участка: 'дом', 'владение', 'домовладение', nan, 'участок',
    # 'земельный участок', 'сооружение'
    'L1_TYPE': 'local_object_type',
    # номер ома, владения, участка
    'L1_VALUE': 'local_object_num',
    # Номер корпуса
    'L2_VALUE': 'korpus_num',
    # Номер строения или сооружения
    'L3_VALUE': 'stroenie_num',
    # Административный округ
    'ADM_AREA': ADM_AREA_COL,
    # Муниципальный округ, поселение
    'DISTRICT': DISTRICT_COL,
    # Уникальный номер адреса в Адресном реестре
    'NREG': 'num_addr_register',
    # Дата регистрации адреса в Адресном реестре
    'DREG': 'date_addr_register',
    # Уникальный номер адреса в государственном адресном реестре (код (GUID) ФИАС для адреса)
    'N_FIAS': 'guid_fias',
    # Дата начала действия адреса в государственном адресном реестре
    'D_FIAS': 'date_fias',
    # Кадастровый номер объекта недвижимости (кроме земельного участка)
    'KAD_N': 'kad_n',
    # Кадастровый номер земельного участка (для объектов капитального строительства).
    'KAD_ZU': 'kad_zu',
    # код записи из Классификатора адресов Российской Федерации (КЛАДР),
    # соответствующий объекту классификации для адресообразующего элемента нижнего уровня
    'KLADR': 'kladr_code',
    # Статус адреса
    # «ожидает выдачи разрешения на строительство»
    # (действует до получения сведений о выдаче разрешения на строительство),
    # «ожидает внесения в ГКН»
    # (действует до получения сведений о постановке объекта на кадастровый учет),
    # «внесён в ГКН» (действует с даты получения сведений о постановке объекта на кадастровый учет),
    # «ожидает аннулирования в ГКН» (не используется),
    # «аннулирован в ГКН»
    # (устанавливается на основании полученного уведомления о снятии объекта с кадастрового учета),
    # «аннулирован» (адрес погашен в Адресном реестре),
    # «-» (прочерк).
    'STATUS': 'addr_status',
    # абрис дома
    DMR_GEODATA_COL: DB_GEODATA_COL
}

# На интерактивной карте не должны учитываться следующие административные округа: ТиНАО, ЗелАО.
ADM_AREA_TO_EXCLUDE = [
    'Зеленоградский административный округ',
    'Новомосковский административный округ',
    'Троицкий административный округ'
    ]

GIS_COLS_MAP = {
    'Адрес ОЖФ': 'address_gis',
    'Идентификационный код адреса дома в ГИС ЖКХ': 'address_code_gis',
    'Глобальный уникальный идентификатор дома по ФИАС': 'guid_fias_gis',
    'Код ОКТМО': 'oktmo_code_gis',
    'Способ управления': 'management_method_gis',
    'ОГРН организации, осуществляющей управление домом': 'management_ogrn_gis',
    'КПП организации, осуществляющей управление домом': 'management_kpp_gis',
    'Наименование организации, осуществляющей управление домом': 'management_name_gis',
    'Тип дома': 'house_type_gis',
    'Состояние': 'condition_gis',
    'Общая площадь дома': 'total_area_gis',
    'Жилая площадь в доме': 'living_area_gis',
    'Дата сноса объекта': 'demolition_date_gis',
    'Кадастровый номер': 'kad_n_gis',
    'Глобальный уникальный идентификатор дома': 'guid_house_gis'
    }

# мэппинг названий колонок в сырых данных инфраструктурных объектов
INFRA_COLS_MAP = {
    'AdmArea': ADM_AREA_COL,
    'District': DISTRICT_COL,
    'ObjectType': OBJECT_TYPE_COL,
    'Address': ADDRESS_COL,
    'Name': 'commonname',
    INFRA_GEODATA_COL: DB_GEODATA_COL,
    OBJECT_ID_COL: OBJECT_ID_COL
}

# колонки с данными, которые содержат листы словарей
# в сырых данных инфраструктурных объектов
INFRA_WORKING_HOURS_COL = 'WorkingHours'
INFRA_COMBINED_ADDRESS_COL = 'ObjectAddress'

# Ключи, по которым из INFRA_COMBINED_ADDRESS_COL надо извлекать данные
# при обработке сырых данных с data.mos.ru о кисосках, МФЦ и других
# инфраструктурных объектах
COMBINED_ADDRESS_KEYS = ['AdmArea', 'District', 'Address']

# то что имеет смысл забирать из сырых данных с data.mos.ru по объектам инфраструктуры
KIOSKS_COLS = [
    'ObjectType', 'Name', 'AdmArea', 'District',
    'Address', 'FacilityArea', 'Specialization',
    'PeriodOfPlacement', 'StartDateTrading', 'EndDateTrading',
    INFRA_GEODATA_COL
    ]
KIOSKS_OT = 'киоск'

MFC_COLS = [
    'CommonName', 'AdmArea', 'District', 'Address',
    'WorkingHours','ExtraServices', 'CenterArea', 'WindowCount',
    INFRA_GEODATA_COL
    ]
MFC_OT = 'МФЦ'

LIBS_COLS = [
    'Category', 'CommonName', 'ObjectAddress', 'WorkingHours',
    'NumOfSeats', 'NumOfReaders', 'NumOfVisitors',
    INFRA_GEODATA_COL
    ]
LIBS_OT = 'библиотека'

CLUBS_COLS = ['Category', 'CommonName', 'ObjectAddress', 'WorkingHours', INFRA_GEODATA_COL]
CLUBS_OT = 'дом культуры или клуб'

SPORTS_COLS = ['Category', 'CommonName', 'ObjectAddress', 'WorkingHours', INFRA_GEODATA_COL]
SPORTS_OT = 'cпортивный объект'

INFRA_NEEDED_COLS_BY_OBJECTS = {
    KIOSKS_OT: KIOSKS_COLS,
    MFC_OT: MFC_COLS,
    LIBS_OT: LIBS_COLS,
    CLUBS_OT: CLUBS_COLS,
    SPORTS_OT: SPORTS_COLS
}

# названия таблиц для инфраструктурных объектов
INFRA_TABLES_NAMES_BY_OBJECTS = {
    KIOSKS_OT: KIOSKS_NAME,
    MFC_OT: MFC_NAME,
    LIBS_OT: LIBS_NAME,
    CLUBS_OT: CLUBS_NAME,
    SPORTS_OT: SPORTS_NAME
}

# обязательные колонки, которые должны быть в табличке по всем объектам
# потенциального размещения постаматов
MANDATORY_COLS = [OBJECT_ID_COL, ADM_AREA_COL, DISTRICT_COL, OBJECT_TYPE_COL,
                  LATITUDE_COL, LONGITUDE_COL, ADDRESS_COL]
