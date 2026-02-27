#include "DimensionSelectionAction.h"

#include <PointData/DimensionsPickerAction.h>

DimensionSelectionAction::DimensionSelectionAction(QObject* parent) :
    mv::gui::GroupAction(parent, "DimensionSelection"),
    _pickerAction(new DimensionsPickerAction(this, "DimensionsPickerAction"))
{
    setText("Dimensions");

    addAction(_pickerAction);
}
