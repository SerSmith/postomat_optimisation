import {Component, EventEmitter, Input, OnInit, Output} from '@angular/core';
import {FormControl, FormGroup} from "@angular/forms";

@Component({
  selector: 'app-left-panel',
  templateUrl: './left-panel.component.html',
  styleUrls: ['./left-panel.component.css']
})
export class LeftPanelComponent implements OnInit {
  @Output() closeInformation = new EventEmitter();
  @Output() changePointType = new EventEmitter();
  @Input() selectedPoint: any;
  public typeForm: FormGroup;
  public types = ['recom', 'fixed', 'ban', 'permis'];
  public select = [];
  public typesMap = {
    recom: "Рекомендованная",
    fixed: 'Фиксированная',
    ban: 'Запрещенная',
    permis: 'Допустимая'
  }

  constructor() {
    this.typeForm = new FormGroup({
      selectType: new FormControl(),
    })
  }

  ngOnInit(): void {
    this.select = [];
    (this.select as any)[this.selectedPoint.type] = true;
  }

  public closeInfo() {
    this.closeInformation.emit();
  }

  public changeType() {
    this.changePointType.emit(
      {
        id: this.selectedPoint.object_id,
        type: this.typeForm.value.selectType
      });
  }
}
