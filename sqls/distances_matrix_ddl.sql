CREATE TABLE public.distances_matrix
(
	object_id text,
	id_center_mass text,
	distance float,
	CONSTRAINT object_mass_id PRIMARY KEY(object_id, id_center_mass)
);

COMMENT ON TABLE public.distances_matrix IS 'Таблица с расстояниями от центров масс населения районов до объектов размещения постаматов';
COMMENT ON COLUMN public.distances_matrix.object_id IS 'Идентификатор объекта';
COMMENT ON COLUMN public.distances_matrix.id_center_mass IS 'Идентификатор центра масс';
COMMENT ON COLUMN public.distances_matrix.distance IS 'Расстояние, м';