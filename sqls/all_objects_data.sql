insert into public.all_objects_data (object_id, adm_area, district, object_type, lat, lon, address) 
select object_id, adm_area, district, object_type, lat, lon, address
from public.kiosks_all_data
union all
select object_id, adm_area, district, object_type, lat, lon, address
from public.mfc_all_data
union all
select object_id, adm_area, district, object_type, lat, lon, address
from public.libs_all_data
union all
select object_id, adm_area, district, object_type, lat, lon, address
from public.clubs_all_data
union all
select object_id, adm_area, district, object_type, lat, lon, address
from public.sports_all_data
union all
select object_id, adm_area, district, object_type, lat, lon, address
from public.apartment_houses_all_data

on conflict (object_id) do update set
    adm_area = EXCLUDED.adm_area,
    district = EXCLUDED.district,
    object_type = EXCLUDED.object_type,
    lat = EXCLUDED.lat,
    lon = EXCLUDED.lon,
    address = EXCLUDED.address;
