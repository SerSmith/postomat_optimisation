import {Component, OnInit} from '@angular/core';
import {MarkerService} from "../marker.service";
import {MapPointsService} from "../map-points.service";

@Component({
  selector: 'app-table',
  templateUrl: './table.component.html',
  styleUrls: ['./table.component.css']
})
export class TableComponent implements OnInit {
  public t: any;

  constructor(public pointsMap: MapPointsService, private markerService: MarkerService) {
    this.t = this.pointsMap.get();
    if (!this.t) {
      this.markerService.testMap()
        .subscribe(fields => {
          this.t = JSON.parse(fields);
          this.pointsMap.set(this.t);
        });
    }
  }

  ngOnInit(): void {
  }

}
