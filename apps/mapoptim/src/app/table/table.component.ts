import { Component, OnInit } from '@angular/core';
import {FormControl} from "@angular/forms";
import {MarkerService} from "../marker.service";
import {MapPointsService} from "../map-points.service";

@Component({
  selector: 'app-table',
  templateUrl: './table.component.html',
  styleUrls: ['./table.component.css']
})
export class TableComponent implements OnInit {
  public t: any;
   constructor( public pointsMap: MapPointsService) {
     this.t=this.pointsMap.get()
   }
  ngOnInit(): void {
  }

}
