insert into public.distances_matrix (object_id, id_center_mass, distance, walk_time) 

with
center_mass as (
	select id_center_mass, lat, lon
	from public.centers_mass cm 
	),
all_obj as (
	select object_id, lat, lon
	from public.all_objects_data aod
	-- на этапе mvp не используем дома как места расстановки
	-- так как их приоритет по ТЗ самый низкий и их много
	where aod.object_type <> 'многоквартирный дом'
),
cross_table as (
	select
		all_obj.object_id,
		center_mass.id_center_mass,
		calculate_distance(center_mass.lat, center_mass.lon, all_obj.lat, all_obj.lon) as distance
	from center_mass cross join all_obj
)
select
	cross_table.object_id,
	cross_table.id_center_mass,
	cross_table.distance,
	(cross_table.distance * 1.152)::integer as walk_time
from cross_table

on conflict (object_id, id_center_mass) do update set
    distance = EXCLUDED.distance,
	walk_time = EXCLUDED.walk_time;