CREATE TABLE public.all_objects_data
(
	object_id text PRIMARY KEY,
	adm_area text,
	district text,
	object_type text,
	lat float,
	lon float,
	address text
);

COMMENT ON TABLE public.all_objects_data IS 'Таблица с данными о потенциальных местах размещения постаматов';
COMMENT ON COLUMN public.all_objects_data.object_id IS 'Идентификатор объекта';
COMMENT ON COLUMN public.all_objects_data.adm_area IS 'Административный округ';
COMMENT ON COLUMN public.all_objects_data.district IS 'Район';
COMMENT ON COLUMN public.all_objects_data.object_type IS 'Тип объекта размещения';
COMMENT ON COLUMN public.all_objects_data.lat IS 'Широта объекта размещения';
COMMENT ON COLUMN public.all_objects_data.lon IS 'Долгота объекта размещения';
COMMENT ON COLUMN public.all_objects_data.address IS 'Адрес объекта размещения';
