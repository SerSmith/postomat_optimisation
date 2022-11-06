import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppComponent } from './app.component';
import { MapComponent } from './map/map.component';
import {HttpClientModule} from "@angular/common/http";
import {MarkerService} from "./marker.service";
import {TableComponent} from "./table/table.component";
import {AppRoutingModule} from "./AppRoutingModule.module";
import {FiltersComponent} from "./filters/filters.component";
import {MatFormFieldModule} from "@angular/material/form-field";
import {MatSelectModule} from "@angular/material/select";
import {BrowserAnimationsModule} from "@angular/platform-browser/animations";
import {ReactiveFormsModule} from "@angular/forms";
import {LeftPanelComponent} from "./left-panel/left-panel.component";

@NgModule({
  declarations: [
    AppComponent,
    MapComponent,
    TableComponent,
    FiltersComponent,
    LeftPanelComponent
  ],
  imports: [
    BrowserAnimationsModule,
    BrowserModule,
    HttpClientModule,
    AppRoutingModule,
    MatFormFieldModule,
    MatFormFieldModule,
    MatSelectModule,
    ReactiveFormsModule
  ],
  providers: [MarkerService],
  bootstrap: [AppComponent]
})
export class AppModule { }
