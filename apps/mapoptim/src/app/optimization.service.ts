import {Injectable} from '@angular/core';
import {map, Observable} from "rxjs";
import {HttpClient} from "@angular/common/http";

@Injectable({
  providedIn: 'root'
})
export class OptimizationService {

  constructor(private http: HttpClient) {
  }

  public testOpt(): Observable<any> {
    const url = 'http://178.170.195.175/get_optimized_postomat_places';
    return this.http.get(url, {
      headers: {
        'Access-Control-Allow-Origin': '*'
      }
    }).pipe(map(r => r));
  }

  public heatMap(recomPoints: any): Observable<any> {
    let param = "?step=1&list_object_id=" + recomPoints.toString();

    const url = 'http://178.170.195.175/get_point_statistics';
    return this.http.get((url + param), {
      headers: {
        'Access-Control-Allow-Origin': '*'
      }
    }).pipe(map(r => r));
  }
  public optimized(): Observable<any> {
    let param = "?possible_points=['e3954a1c7efddf7f1bf68aaa6afe085044ac28bee7f103f3ec1c78cbdb1485b0','e004c2cb0b81e7b2f66bdeee7a429455952cb85c08a15ea4663cfdfa533a4dd0']&fixed_points=['e3954a1c7efddf7f1bf68aaa6afe085044ac28bee7f103f3ec1c78cbdb1485b0','e004c2cb0b81e7b2f66bdeee7a429455952cb85c08a15ea4663cfdfa533a4dd0']";
    const url = 'http://178.170.195.175/get_optimized_postomat_places';
    return this.http.get((url + param), {
      headers: {
        'Access-Control-Allow-Origin': '*'
      }
    }).pipe(map(r => r));
  }
  public optim(filter: any): Observable<any> {
    console.log(filter.banned_points);
    let param = "?quantity_postamats_to_place="+filter.postamatQuant+"&step=0.1&metro_weight="+
      filter.metroImportance+"&opt_tome="+filter.opt_time+"&max_time="+
      filter.max_time+filter.selectedTypes+filter.selectedDistricts+
      filter.selectedArea+filter.banned_points+filter.fixed_points;
    const url = 'http://178.170.195.175:81/get_optimized_postomat_places';
    return this.http.get((url + param), {
      headers: {
        'Access-Control-Allow-Origin': '*'
      }
    }).pipe(map(r => r));
  }

}
