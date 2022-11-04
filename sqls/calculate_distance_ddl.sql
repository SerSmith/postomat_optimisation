CREATE OR REPLACE FUNCTION calculate_distance(lat1 float, lon1 float, lat2 float, lon2 float)
RETURNS float AS $distance$
	DECLARE
	distance float = 0;
	rlat1 float;
	rlat2 float;
	rlon1 float;
    rlon2 float;
    hav_arg float;
    BEGIN
		rlat1 = pi() * lat1 / 180;
		rlat2 = pi() * lat2 / 180;
		rlon1 = pi() * lon1 / 180;
		rlon2 = pi() * lon2 / 180;
		hav_arg = sin((rlat2 - rlat1)/2)^2 + cos(lat1) * cos(lat2) * sin((rlon2 - rlon1)/2)^2;
		distance = 6363568 * 2 * asin(sqrt(hav_arg));
		RETURN distance;
    END;
$distance$ LANGUAGE plpgsql;