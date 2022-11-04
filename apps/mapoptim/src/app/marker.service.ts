import {Injectable} from '@angular/core';
import * as L from 'leaflet';
import {HttpClient} from "@angular/common/http";
import {map, Observable} from "rxjs";
import 'leaflet.markercluster';

@Injectable({
  providedIn: 'root'
})
export class MarkerService {
  capitals: string = '/assets/data/usa-capitals.geojson';
  test: string = '/assets/data/points.geojson'
  public addressPoints = [
    [-37.8210922667, 175.2209316333, "2"],
    [-37.8210819833, 175.2213903167, "3"],
    [-37.8210881833, 175.2215004833, "3A"],
    [-37.8211946833, 175.2213655333, "1"],
    [-37.8209458667, 175.2214051333, "5"],
    [-37.8208292333, 175.2214374833, "7"],
    [-37.8325816, 175.2238798667, "537"],
    [-37.8315855167, 175.2279767, "454"],
    [-37.8096336833, 175.2223743833, "176"],
    [-37.80970685, 175.2221815833, "178"],
    [-37.8102146667, 175.2211562833, "190"],
    [-37.8088037167, 175.2242227, "156"],
    [-37.8112330167, 175.2193425667, "210"],
    [-37.8116368667, 175.2193005167, "212"],
    [-37.80812645, 175.2255449333, "146"],
    [-37.8080231333, 175.2286383167, "125"],
    [-37.8089538667, 175.2222222333, "174"],
    [-37.8080905833, 175.2275400667, "129"]
  ]

  constructor(private http: HttpClient) {
    // this.http.get(this.capitals).subscribe((res: any) => {
    //   const test = JSON.parse(res);
    //   console.log(test);
    // });

  }

  makeCapitalMarkers(map: any): void {
    this.http.get(this.capitals).subscribe((res: any) => {
      for (const c of res.features) {

        const lon = c.geometry.coordinates[0];

        const lat = c.geometry.coordinates[1];

        const marker = L.marker([lat, lon]);

        marker.addTo(map);

      }
      // let markers4 = L.markerClusterGroup();
      // markers4.addLayer(L.marker(getRandomLatLng(map)));
      //  map.addLayer(markers4);
    });
  }

  makeCapitalCircleMarkers(map: any): void {
    this.http.get(this.test).subscribe((res: any) => {
      console.log(res);
      const test = JSON.parse(res);
      console.log(test);
    });
    this.http.get(this.capitals).subscribe((res: any) => {
      const maxPop = Math.max(...res.features.map((x: any) => x.properties.population), 0);

      for (const c of res.features) {

        const lon = c.geometry.coordinates[0];

        const lat = c.geometry.coordinates[1];

        const circle = L.circleMarker([lat, lon], {radius: MarkerService.scaledRadius(c.properties.population, maxPop)}).addTo(map);

        // circle.addTo(map);

      }

    });

  }
  public mapGroup(map: any){
    let markers = L.markerClusterGroup();

    for (let i = 0; i < this.addressPoints.length; i++) {
      let a = this.addressPoints[i];
      let title = a[2].toString();
      let marker = L.marker(new L.LatLng(<number>a[0], <number>a[1]), {
        title: title
      });
      marker.bindPopup(title);
      markers.addLayer(marker);
    }

    map.addLayer(markers);
  }

  static scaledRadius(val: number, maxVal: number): number {

    return 20 * (val / maxVal);

  }

  public testMap(): Observable<any> {
    const url = 'http://178.170.195.175/get_all_postomat_places';
    return this.http.get(url, {
      headers: {
        'Access-Control-Allow-Origin': '*'
      }
    }).pipe(map(r => r));
  }
  public heatMap(map: any){

  }
}
