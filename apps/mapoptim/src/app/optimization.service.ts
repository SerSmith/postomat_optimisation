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
    let param = "?step=0.1&list_object_id=" + recomPoints.toString();

    const url = 'http://178.170.195.175/get_point_statistics';
    return this.http.get((url + param), {
      headers: {
        'Access-Control-Allow-Origin': '*'
      }
    }).pipe(map(r => r));
  }

  public optim(filter: any): Observable<any> {
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
  public optim2(filter: any): Observable<any> {
    console.log(filter.selectedTypes);
    console.log(filter.selectedArea);
    let param = "?quantity_postamats_to_place="+filter.postamatQuant+"&step=0.1&metro_weight="+
      filter.metroImportance+"&large_houses_priority="+filter.large_houses_priority+"&max_time="+
      filter.max_time+filter.selectedTypes+filter.selectedDistricts+
      filter.selectedArea+filter.banned_points+filter.fixed_points;
    const url = 'http://178.170.195.175:82/get_kmeans_optimize_points';
    return this.http.get((url + param), {
      headers: {
        'Access-Control-Allow-Origin': '*'
      }
    }).pipe(map(r => r));
  }

}
