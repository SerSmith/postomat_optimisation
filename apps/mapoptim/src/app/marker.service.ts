import {EventEmitter, Injectable, Output} from '@angular/core';
import * as L from 'leaflet';
import {HttpClient} from "@angular/common/http";
import {map, Observable} from "rxjs";
import 'leaflet.markercluster';
const iconUrlGrey = 'assets/grey.png'
const iconUrlGreen = 'assets/img.png'
@Injectable({
  providedIn: 'root'
})

export class MarkerService {
  public markers: any;
  @Output() selected = new EventEmitter();

  constructor(private http: HttpClient) {
  }

  // public mapGroup(map: any, points: any) {
  //   if (this.markers) {
  //     map.removeLayer(this.markers);
  //   }
  //   this.markers = L.markerClusterGroup();
  //   for (let i = 0; i < points.length; i++) {
  //     let title = points[i].address + points[i].object_type;
  //     let iconUrl = points[i].recommend? iconUrlGreen: iconUrlGrey;
  //     let marker = L.marker(new L.LatLng(points[i].lat, points[i].lon), {
  //       icon: L.icon({
  //         iconUrl,
  //
  //         iconSize: [40, 41],
  //
  //         iconAnchor: [12, 41],
  //
  //         popupAnchor: [1, -34],
  //
  //         tooltipAnchor: [16, -28],
  //
  //         shadowSize: [41, 41]
  //
  //       }),
  //       title: title
  //     }).on('click', function(e) {
  //       this.selected.emit('ss');
  //     });
  //     marker.bindPopup(title);
  //     this.markers.addLayer(marker);}
  //
  //   map.addLayer(this.markers);
  // }

  public testMap(): Observable<any> {
    const url = 'http://178.170.195.175/get_all_postomat_places';
    return this.http.get(url, {
      headers: {
        'Access-Control-Allow-Origin': '*'
      }
    }).pipe(map(r => r));
  }
}
