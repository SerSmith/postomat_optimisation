import {AfterViewInit, Component} from '@angular/core';
import * as L from 'leaflet';
import {MarkerService} from '../marker.service';
import 'leaflet.heat/dist/leaflet-heat.js'
import 'leaflet.markercluster';
import {OptimizationService} from "../optimization.service";
import {MapPointsService} from "../map-points.service";

const iconUrlGrey = 'assets/grey.png';
const iconUrlGreen = 'assets/green.png';
const iconUrlRed = 'assets/red.png';
const iconUrlBlue = 'assets/blue.png';
const iconRetinaUrl = 'assets/img.png';
const iconUrl = 'assets/img.png'
const iconDefault = L.icon({
  iconRetinaUrl,
  iconUrl,
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
  public t: any;
  public recomm: any;
  public markers: any;
  public showedInfo: boolean = false;
  public selectedPoint: any;
  public parametersList: any;

  public recommPoints: any = [];
  public recommIds: any = [];

  public banPoints: any = [];
  public banIds: any = [];

  public fixedPoints: any = [];
  public fixedIds: any = [];

  private initMap(): void {
    this.map = L.map('map', {
      center: [55.755864, 37.617698],
      zoom: 11
    });
    const tiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18,
      minZoom: 7,
      attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    });
    tiles.addTo(this.map);
  }

  constructor(private markerService: MarkerService,
              public optim: OptimizationService,
              public pointsMap: MapPointsService) {
    if (!this.pointsMap.get()) {
      this.markerService.testMap()
        .subscribe(fields => {
          this.t = JSON.parse(fields);
          this.pointsMap.set(this.t);
          this.mapGroup(this.map, this.t);
        })
    }
  }

  ngAfterViewInit(): void {
    this.initMap();
    if (this.pointsMap.get()) {
      this.t = this.pointsMap.get();
      this.mapGroup(this.map, this.t);
    }
  }

  public showFilters(): void {
    this.filtersShowed = !this.filtersShowed;
  }

  onMapReady(point: any) {
    let heatPoint = new Array;
    point.forEach((el: any) => {
      let option = this.setOptHeatMap(el.walk_time);
      heatPoint.push([el.lat, el.lon, option])
    })
    const heat = (L as any).heatLayer(heatPoint, {radius: 30, opacity:0.1}).addTo(this.map);
  }

  public applyFilters(filters: any): void {
    let filteredPoints = this.t;
    if (filters.onlyRecommed) {
      filteredPoints = [];
      filteredPoints = (this.t.filter((el: any) => el.type === 'recom'));
    }
    // if (filters.onlyRecommed) {
    //   filteredPoints.push(this.t.filter((el: any) => el.type === 'recom'));
    // }
    // if (filters.onlyRecommed) {
    //   filteredPoints.push(this.t.filter((el: any) => el.type === 'recom'));
    // }
    // if (filters.onlyRecommed) {
    //   filteredPoints.push(this.t.filter((el: any) => el.type === 'recom'));
    // }
    let selectedPoints = new Array;
    filteredPoints.forEach((el: any) => {
      if (filters.selectedTypes) {
        if (filters.selectedArea) {
          if (filters.selectedDistricts) {
            if ((filters.selectedTypes.includes(el.object_type))
              && (filters.selectedArea.includes(el.adm_area))
              && (filters.selectedDistricts.includes(el.district))) {
              selectedPoints.push(el);
            }
          } else {
            if ((filters.selectedTypes.includes(el.object_type))
              && (filters.selectedArea.includes(el.adm_area))) {
              selectedPoints.push(el);
            }
          }
        } else {
          if (filters.selectedDistricts) {
            if ((filters.selectedTypes.includes(el.object_type))
              && (filters.selectedDistricts.includes(el.district))) {
              selectedPoints.push(el);
            }
          } else {
            if (filters.selectedTypes.includes(el.object_type)) {
              selectedPoints.push(el);
            }
          }
        }
      } else {
        if (filters.selectedArea) {
          if (filters.selectedDistricts) {
            if ((filters.selectedArea.includes(el.adm_area))
              && (filters.selectedDistricts.includes(el.district))) {
              selectedPoints.push(el);
            }
          } else {
            if (filters.selectedArea.includes(el.adm_area)) {
              selectedPoints.push(el);
            }
          }
        } else {
          if (filters.selectedDistricts) {
            if ((el.district === filters.selectedDistricts)) {
              selectedPoints.push(el);
            }
          } else {
            selectedPoints = filteredPoints;
          }
        }
      }
    })
    this.mapGroup(this.map, selectedPoints);
  }

  public optFilters(filters: any): void {
    if (filters.selectedTypes) {
      filters.selectedTypes = "&object_type_filter_list=" + this.arrayToString2(filters.selectedTypes);
    } else {
      filters.selectedTypes = "";
    }
    if (filters.selectedDistricts) {
      filters.selectedDistricts = "&district_type_filter_list=" + this.arrayToString2(filters.selectedDistricts);
    } else {
      filters.selectedDistricts = "";
    }
    if (filters.selectedArea) {
      filters.selectedArea = "&adm_areat_type_filter_list=" + this.arrayToString2(filters.selectedArea);
    } else {
      filters.selectedArea = "";
    }
    if (this.fixedIds.length) {
      filters.fixed_points = "&fixed_points=" + this.arrayToString2(this.fixedIds);
    } else {
      filters.fixed_points = "";
    }
    if (this.banIds.length) {
      filters.banned_points = "&banned_points=" + this.arrayToString2(this.banIds);
    } else {
      filters.banned_points = "";
    }

    this.optim.optim(filters).subscribe(rec => {
      this.recommPoints = [];
      this.recomm = JSON.parse(rec);
      this.t.forEach((el: any) => {
        if (this.recomm.optimized_points.includes(el.object_id)) {
          el.type = 'recom';
          this.recommPoints.push(el);
          this.recommIds.push(el.object_id);
        }
      })
      this.mapGroup(this.map, this.t);
      this.setHeatMap();
    });
  }

  public resetFilters() {
    this.mapGroup(this.map, this.t);
  }

  public setHeatMap() {
    let stringIds = this.arrayToString(this.recommIds);
    this.optim.heatMap(stringIds)
      .subscribe(fields => {
        let r = JSON.parse(fields[1]);
        this.parametersList = JSON.parse(fields[2]);
        this.parametersList.forEach((param: any) => param.percent_people = Math.trunc(param.percent_people))
        this.onMapReady(r);
      })
  }

  public mapGroup(map: any, points: any) {
    if (this.markers) {
      map.removeLayer(this.markers);
    }
    this.markers = L.markerClusterGroup();
    for (let i = 0; i < points.length; i++) {
      let iconUrl = this.applyIcon(points[i].type);
      let marker = L.marker(new L.LatLng(points[i].lat, points[i].lon), {
        icon: L.icon({
          iconUrl,
          iconSize: [30, 41],
          iconAnchor: [12, 41],
          popupAnchor: [1, -34],
          tooltipAnchor: [16, -28],
          shadowSize: [41, 41]
        }),
      }).on('click', (e) => {
        this.showInfo(points[i]);
      });
      this.markers.addLayer(marker);
    }

    map.addLayer(this.markers);
  }

  public showInfo(point: any): void {
    if (this.selectedPoint === point) {
      this.closeInformation();
    } else {
      this.selectedPoint = point;
      this.showedInfo = true;
    }
  }

  public closeInformation(): void {
    this.showedInfo = false;
    this.selectedPoint = null;
  }

  public changePointType(event: any): void {
    switch (event.type) {
      case 'recom':
        this.recommPoints.push(this.t.find((el: any) => el.object_id == event.id));
        this.recommIds.push(event.id);
        break;
      case 'fixed':
        this.fixedPoints.push(this.t.find((el: any) => el.object_id == event.id));
        this.fixedIds.push(event.id);
        break;
      case 'ban':
        this.banPoints.push(this.t.find((el: any) => el.object_id == event.id));
        this.banIds.push(event.id);
        break;
      default:
        break ;
    }

    console.log(this.banPoints);
    this.t.find((el: any) => el.object_id == event.id).type = event.type;
    this.mapGroup(this.map, this.t);
    console.log(this.t.find((el: any) => el.object_id == event.id));
  }

  private applyIcon(type: string): string {
    switch (type) {
      case 'recom':
        return iconUrlGreen;
      case 'fixed':
        return iconUrlBlue;
      case 'ban':
        return iconUrlRed;
      case 'permis':
        return iconUrlGrey;
      default:
        return iconUrlGrey;
    }
  }

  private arrayToString(array: any): string {
    console.log(array);
    let stringIds = "["
    for (let param of array) {
      stringIds = stringIds + "'" + param + "',";
    }
    stringIds = stringIds.slice(0, -1);
    stringIds = stringIds + "]";
    return stringIds;
  }
  private arrayToString2(array: any): string {
    let stringIds = "["
    for (let param of array) {
      stringIds = stringIds + param + ",";
    }
    stringIds = stringIds.slice(0, -1);
    stringIds = stringIds + "]";
    return stringIds;
  }

  public setOptHeatMap(walk_time: number): number {
    if (walk_time < 420) {
      return 100;
    }
    if (walk_time > 420 && walk_time < 600) {
      return 60;
    }
    if (walk_time > 600 && walk_time < 900) {
      return 40;
    }
    return 20;
  }
}
