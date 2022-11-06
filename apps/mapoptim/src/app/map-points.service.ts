import {Injectable} from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class MapPointsService {
  public points: any;

  constructor() {
  }

  public set(points: any) {
    this.points = points;
  }

  public get(): any {
    return this.points;
  }
}
