import {EventEmitter, Injectable, Output} from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {map, Observable} from "rxjs";
import 'leaflet.markercluster';
@Injectable({
  providedIn: 'root'
})

export class MarkerService {
  public markers: any;
  @Output() selected = new EventEmitter();

  constructor(private http: HttpClient) {
  }


  public testMap(): Observable<any> {
    const url = 'http://178.170.195.175/get_all_postomat_places';
    return this.http.get(url, {
      headers: {
        'Access-Control-Allow-Origin': '*'
      }
    }).pipe(map(r => r));
  }
  // public saveExel(): any {
  //   const url = 'http://178.170.195.175/get_excel?method_name=test&walk_time=15&step=0.5';
  //   return this.http.get(url, {
  //     headers: {
  //       'Access-Control-Allow-Origin': '*'
  //     }
  //   }).pipe(map(r => r));
  // }
}
