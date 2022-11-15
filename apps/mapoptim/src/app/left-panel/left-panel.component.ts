import {Component, EventEmitter, Input, OnChanges, Output} from '@angular/core';
import {FormControl, FormGroup} from "@angular/forms";

@Component({
  selector: 'app-left-panel',
  templateUrl: './left-panel.component.html',
  styleUrls: ['./left-panel.component.css']
})
export class LeftPanelComponent implements OnChanges {
  @Output() closeInformation = new EventEmitter();
  @Output() changePointType = new EventEmitter();
  @Input() selectedPoint: any;
  public typeForm: FormGroup;
  public types = ['Рекомендованная', 'Фиксированная', 'Запрещенная', 'Допустимая'];
  public select = [];
  public prevType: any;

  constructor() {
    this.typeForm = new FormGroup({
      selectType: new FormControl(null),
    })
  }

  ngOnChanges(): void {
    this.prevType = this.selectedPoint.type ? this.selectedPoint.type : 'Допустимая';
    this.typeForm.setValue({selectType: this.prevType});
  }

  public closeInfo() {
    this.closeInformation.emit();
  }

  public changeType() {
    this.changePointType.emit(
      {
        id: this.selectedPoint.object_id,
        type: this.typeForm.value.selectType,
        prevType: this.prevType
      });
  }
}
