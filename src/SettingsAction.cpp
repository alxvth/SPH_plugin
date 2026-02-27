#include "SettingsAction.h"

#include "SphPlugin.h"

SettingsAction::SettingsAction(SPHPlugin* parent) :
    mv::gui::GroupAction(dynamic_cast<QObject*>(parent), "SettingsAction", true),
    _hierarchySettingsAction(this),
    _tsneSettingsAction(this),
    _advancedSettingsAction(this),
    _dimensionSelectionAction(this),
    _refineAction(this),
    _refineTsneSettingsAction(this, "Refine t-SNE")
{
    setText("Spatial Hierarchy");

    _refineTsneSettingsAction.getTsneComputeAction().setEnabled(false);

    addAction(&_hierarchySettingsAction);
    addAction(&_tsneSettingsAction);
    addAction(&_advancedSettingsAction);
    addAction(&_refineAction);
    addAction(&_refineTsneSettingsAction);
    addAction(&_dimensionSelectionAction);

    connect(&_advancedSettingsAction.getNormSchemeAction(), &OptionAction::currentIndexChanged, this, [this](const int32_t currentIndex) {
        
        const bool doTSNE = (currentIndex >= 1) ? false : true;

        _tsneSettingsAction.getExaggerationIterAction().setEnabled(doTSNE);
        _tsneSettingsAction.getExponentialDecayAction().setEnabled(doTSNE);
        _tsneSettingsAction.getExaggerationFactorAction().setEnabled(doTSNE);
        _tsneSettingsAction.getExaggerationToggleAction().setEnabled(doTSNE);
        _tsneSettingsAction.getGradientDescentTypeAction().setEnabled(doTSNE);

        if(doTSNE)
            _tsneSettingsAction.getNumDefaultUpdateIterationsAction().setValue(1000);
        else
            _tsneSettingsAction.getNumDefaultUpdateIterationsAction().setValue(500);

        });
}
