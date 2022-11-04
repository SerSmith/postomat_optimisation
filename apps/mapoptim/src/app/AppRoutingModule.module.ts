import {RouterModule, Routes} from "@angular/router";
import {NgModule} from "@angular/core";
import {MapComponent} from "./map/map.component";
import {TableComponent} from "./table/table.component";

const routes: Routes = [
  {
    path: 'map',
    component: MapComponent,
  },
  {
    path: 'table',
    component: TableComponent,
  },
  {path: '**', component: MapComponent},
]

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {
}
