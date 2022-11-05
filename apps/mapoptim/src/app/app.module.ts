import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppComponent } from './app.component';
import { MapComponent } from './map/map.component';
import {HttpClientModule} from "@angular/common/http";
import {MarkerService} from "./marker.service";
import {TableComponent} from "./table/table.component";
import {AppRoutingModule} from "./AppRoutingModule.module";

@NgModule({
  declarations: [
    AppComponent,
    MapComponent,
    TableComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    AppRoutingModule
  ],
  providers: [MarkerService],
  bootstrap: [AppComponent]
})
export class AppModule { }
