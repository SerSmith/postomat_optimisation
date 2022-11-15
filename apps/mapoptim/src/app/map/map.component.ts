import * as L from 'leaflet';
import 'leaflet.markercluster';
import 'leaflet.markercluster/dist/leaflet.markercluster';

import {AfterViewInit, Component} from '@angular/core';
import {MarkerService} from '../marker.service';
// import 'leaflet.heat/dist/leaflet-heat.js'
import '../leaflet-heat.js';

import {OptimizationService} from "../optimization.service";
import {MapPointsService} from "../map-points.service";


const iconUrlGrey = '../../assets/grey.png';
const iconUrlGreen = '../../assets/green.png';
const iconUrlRed = '../../assets/red.png';
const iconUrlBlue = '../../assets/blue.png';
const iconRetinaUrl = '../../assets/img.png';
const iconUrl = '../../assets/img.png'
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
  public saveLink: string = '';
  public parametersList: any;
  public loader: boolean = false;

  public recommPoints: any = [];
  public recommIds: any = [];

  public banPoints: any = [];
  public banIds: any = [];
  public heat: any;
  public optType: string = '';

  public fixedPoints: any = [];
  public fixedIds: any = [];
  public heatmap: any;
  public currentRadius: any;
  public r: any;

  private initMap(): void {
    this.map = L.map('map', {
      center: [55.755864, 37.617698],
      zoom: 11
    });
    this.currentRadius = this.getRadius(11);
    const tiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18,
      minZoom: 10,
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
        });
    }
    this.setSaveLink();
  }

  ngAfterViewInit(): void {
    this.initMap();
    if (this.pointsMap.get()) {
      this.t = this.pointsMap.get();
      this.mapGroup(this.map, this.t);
    }
    this.map.on('zoomstart', (ev: any) => {
      this.currentRadius = this.getRadius(ev.target._zoom);
    });
  }

  public showFilters(): void {
    this.filtersShowed = !this.filtersShowed;
  }

  onMapReady(point: any, radius: any) {
    let heatPoint = new Array;
    point.forEach((el: any) => {
      let option = this.setOptHeatMap(el.walk_time);
      heatPoint.push([el.lat, el.lon, option])
    })
    if(this.heat){
    this.map.removeLayer(this.heat);}
    this.heat = (L as any).heatLayer(heatPoint, {radius: 20, maxZoom: 7, opacity: 0.5,
      blur: 10}).addTo(this.map);
  }

  public getRadius(currentZoom: any): any {
    var radius;
    if (currentZoom === 18) {
      radius = 200
    } else if (currentZoom === 17) {
      radius = 170;
    } else if (currentZoom === 16) {
      radius = 140;
    } else if (currentZoom === 15) {
      radius = 110;
    } else if (currentZoom === 14) {
      radius = 100;
    } else if (currentZoom === 13) {
      radius = 90;
    } else if (currentZoom === 12) {
      radius = 30;
    } else if (currentZoom === 11) {
      radius = 16;
    } else if (currentZoom === 10) {
      radius = 12;
    } else if (currentZoom === 9) {
      radius = 10;
    } else if (currentZoom === 8) {
      radius = 8;
    } else if (currentZoom === 7) {
      radius = 4;
    }
    if (this.r) {
      this.onMapReady(this.r, radius);
    }
    return radius;
  }

  public applyFilters(filters: any): void {
    let filteredPoints = this.t;
    if (filters.onlyRecommed || filters.onlyBan || filters.onlyDop || filters.onlyFix) {
      filteredPoints = [];
      if (filters.onlyRecommed) {
        this.recommPoints.forEach((el: any) => {
          filteredPoints.push(el);
        })
      }
      if (filters.onlyBan) {
        this.banPoints.forEach((el: any) => {
          filteredPoints.push(el);
        })
      }
      if (filters.onlyFix) {
        this.fixedPoints.forEach((el: any) => {
          filteredPoints.push(el);
        })
      }
      if (filters.onlyDop) {
        this.t.forEach((el: any) => {
          if (el.type !== 'Рекомендованная' && el.type !== 'Фиксированная' && el.type !== 'Запрещенная')
            filteredPoints.push(el);
        })
      }
    }
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

  public optFilters(event: any): void {
    let filters = event.form;
    this.optType = event.typeOpt.optSelect;
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
    if (this.heat) {
      this.map.removeLayer(this.heat);
    }
    if (this.optType === 'Оптимизация на основе солвера') {
      this.optim.optim(filters).subscribe(rec => {
        this.loader = true;
        this.recommPoints = [];
        this.recommIds = [];
        this.recomm = JSON.parse(rec);
        this.t.forEach((el: any) => {
          if (this.recomm.optimized_points.includes(el.object_id)) {
            el.type = 'Рекомендованная';
            this.recommPoints.push(el);
            this.recommIds.push(el.object_id);
            this.loader = false;
          }
        })
        this.mapGroup(this.map, this.recommPoints);
        this.setHeatMap();
        this.setSaveLink();
      });
    } else {
      this.optim.optim2(filters).subscribe(rec => {
        this.loader = true;
        this.recommPoints = [];
        this.recommIds = [];
        this.recomm = JSON.parse(rec);
        this.t.forEach((el: any) => {
          if (this.recomm.optimized_points.includes(el.object_id)) {
            el.type = 'Рекомендованная';
            this.recommPoints.push(el);
            this.recommIds.push(el.object_id);
            this.loader = false;
          }
        })
        this.mapGroup(this.map, this.recommPoints);
        this.setSaveLink();
        this.setHeatMap();
      });
    }
  }

  public resetFilters() {
    this.mapGroup(this.map, this.t);
  }

  public setHeatMap() {
    let arr = new Array();
    if (this.fixedIds.length) {
      this.fixedIds.forEach((el: any) =>
          arr.push(el)
      )
    }
    if (this.recommIds.length) {
      this.recommIds.forEach((el: any) =>
          arr.push(el)
      )
    }
    let stringIds = this.arrayToString(arr);
    console.log(arr);
    this.optim.heatMap(stringIds)
      .subscribe(fields => {
        this.r = JSON.parse(fields[1]);
        this.parametersList = JSON.parse(fields[2]);
        this.parametersList.forEach((param: any) => param.percent_people = Math.trunc(param.percent_people))
        this.onMapReady(this.r, this.currentRadius);
      })
  }

  public mapGroup(map: any, points: any) {
    if (this.markers) {
      map.removeLayer(this.markers);
      this.markers = L.markerClusterGroup();
    } else
    {
      this.markers = L.markerClusterGroup();
    }

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
    if (event.type !== event.prevType) {
      switch (event.type) {
        case 'Рекомендованная':
          this.recommPoints.push(this.t.find((el: any) => el.object_id == event.id));
          this.recommIds.push(event.id);
          this.setSaveLink();
          this.setHeatMap();
          break;
        case 'Фиксированная':
          this.fixedPoints.push(this.t.find((el: any) => el.object_id == event.id));
          this.fixedIds.push(event.id);
          this.setSaveLink();
          this.setHeatMap();
          break;
        case 'Запрещенная':
          this.banPoints.push(this.t.find((el: any) => el.object_id == event.id));
          this.banIds.push(event.id);
          break;
        default:
          break;
      };
      switch (event.prevType) {
        case 'Рекомендованная':
          let rec = this.t.find((el: any) => el.object_id == event.id);
          this.recommPoints.slice(this.recommPoints.indexOf(rec),1);
          this.recommIds.slice(this.recommIds.indexOf(event.id),1);
          console.log(this.recommIds);
          this.setSaveLink();
          break;
        case 'Фиксированная':
          let fix = this.t.find((el: any) => el.object_id == event.id);
          this.fixedPoints.slice(this.fixedPoints.indexOf(fix),1);
          this.fixedIds.slice(this.fixedIds.indexOf(event.id),1);
          this.setSaveLink();
          break;
        case 'Запрещенная':
           let ban = this.t.find((el: any) => el.object_id == event.id);
          this.banPoints.slice(this.banPoints.indexOf(ban),1);
          this.banIds.slice(this.fixedIds.indexOf(event.id),1);
          break;
        default:
          break;
      }
    }
    localStorage.setItem('recommended', this.recommPoints);
    localStorage.setItem('fixed', this.fixedPoints);
    localStorage.setItem('banned', this.banPoints);
    this.t.find((el: any) => el.object_id == event.id).type = event.type;
    this.mapGroup(this.map, this.t);
  }

  private applyIcon(type: string): string {
    switch (type) {
      case 'Рекомендованная':
        return iconUrlGreen;
      case 'Фиксированная':
        return iconUrlBlue;
      case 'Запрещенная':
        return iconUrlRed;
      case 'Допустимая':
        return iconUrlGrey;
      default:
        return iconUrlGrey;
    }
  }

  private arrayToString(array: any): string {
    let stringIds = '';
    if (array?.length) {
      stringIds = "["
      for (let param of array) {
        stringIds = stringIds + "'" + param + "',";
      }
      stringIds = stringIds.slice(0, -1);
      stringIds = stringIds + "]";
    }
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
    if (walk_time <= 180) {
      return 1;
    }
    if (walk_time > 180 && walk_time <= 300) {
      return 0.75;
    }
    if (walk_time > 300 && walk_time <= 600) {
      return 0.5;
    }
    if (walk_time > 600 && walk_time < 900) {
      return 0.25;
    }
    return 0;
  }

  public setSaveLink() {
    let arr = new Array;
    arr = [];
    if (this.fixedIds.length) {
      this.fixedIds.forEach((el: any) =>
        arr.push(el)
      )
    }
    if (this.recommIds.length) {
      this.recommIds.forEach((el: any) =>
        arr.push(el)
      )
    }
    let stringArr = this.arrayToString(arr);
    let list_object_id: string;
    if (stringArr) {
      list_object_id = '&list_object_id=' + stringArr;
    } else {
      list_object_id = ""
    }
    this.saveLink = 'http://178.170.195.175/get_excel?method_name=' + this.optType + '&walk_time=15&step=0.5' + list_object_id;
  }
  public close(){
    this.loader = false;
  }
}
