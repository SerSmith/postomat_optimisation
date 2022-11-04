import {AfterViewInit, Component} from '@angular/core';
import * as L from 'leaflet';
import {MarkerService} from '../marker.service';
import 'leaflet.heat/dist/leaflet-heat.js'
import {addressPoints} from 'src/assets/realworld.100000'
import 'leaflet.markercluster';


const iconRetinaUrl = 'assets/img.png';

const iconUrl = 'assets/img.png'

const shadowUrl = 'assets/marker-shadow.png';

const iconDefault = L.icon({

  iconRetinaUrl,

  iconUrl,

  // shadowUrl,

  iconSize: [40, 41],

  iconAnchor: [12, 41],

  popupAnchor: [1, -34],

  tooltipAnchor: [16, -28],

  shadowSize: [41, 41]

});

L.Marker.prototype.options.icon = iconDefault;

@Component({
  selector: 'app-map',
  templateUrl: './map.component.html',
  styleUrls: ['./map.component.css']
})
export class MapComponent implements AfterViewInit {
  private map: any;
  public filtersShowed: boolean = false;
  public test: boolean = true;

  private initMap(): void {

    this.map = L.map('map', {

      center: [-37.8869090667, 175.3657417333],

      zoom: 3

    });
    const tiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {

      maxZoom: 18,

      minZoom: 3,

      attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'

    });

    tiles.addTo(this.map);
  }

  constructor(private markerService: MarkerService) {
    // this.markerService
    //   .testMap()
    //   .pipe(take(1))
    //   .subscribe(fields => {
    //     console.log(fields);
    //   });
  }

  ngAfterViewInit(): void {
    this.initMap();
    // this.markerService.makeCapitalMarkers(this.map);
    this.markerService.mapGroup(this.map);
    this.markerService.heatMap(this.map);
    this.onMapReady();
    // this.markerService.makeCapitalCircleMarkers(this.map);
  }

  public showFilters(): void {
    this.filtersShowed = !this.filtersShowed;
  }

  onMapReady() {
    let newAddressPoints = addressPoints.map(function (p) {
      return [p[0], p[1]];
    });
      const heat = (L as any).heatLayer(newAddressPoints).addTo(this.map);
      // const heat = (L as any).heatLayer(newAddressPoints).delete
  }

  public teest() {
    this.test = !this.test;
    this.onMapReady();
  }

}
