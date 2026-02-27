#include "SettingsTsneAction.h"

using namespace sph;
using namespace mv::gui;

/// ////////////////// ///
/// TsneSettingsAction ///
/// ////////////////// ///

TsneSettingsAction::TsneSettingsAction(QObject* parent, std::string title) :
    GroupAction(parent, "TsneSettingsAction", false),
    _tsneParameters(),
    _numNewIterationsAction(this, "Continue iter."),
    _numDefaultUpdateIterationsAction(this, "Update iter."),
    _exaggerationIterAction(this, "Exaggeration iter."),
    _exponentialDecayAction(this, "Exponential decay"),
    _exaggerationFactorAction(this, "Exaggeration factor"),
    _exaggerationToggleAction(this, "Auto exaggeration"),
    _iterationsPublishExtendAction(this, "Set Ref. extends at"),
    _publishExtendsOnceAction(this, "Set Ref. extends once", true),
    _initAction(this, "Init embedding with..."),
    _numComputedIterationsAction(this, "Computed iterations"),
    _gradientDescentTypeAction(this, "GD implementation"),
    _ignoreAdjustToLowNumberOfPointsAction(this, "Keep GD impl.", false),
    _tsneComputationAction(this)
{
    setText(QString(title.c_str()));

    /// UI set up: add actions
    addAction(&_exaggerationIterAction);
    addAction(&_exponentialDecayAction);
    addAction(&_exaggerationToggleAction);
    addAction(&_iterationsPublishExtendAction);
    addAction(&_publishExtendsOnceAction);
    addAction(&_initAction);
    addAction(&_numNewIterationsAction);
    addAction(&_numDefaultUpdateIterationsAction);
    addAction(&_numComputedIterationsAction);
    addAction(&_gradientDescentTypeAction);
    addAction(&_ignoreAdjustToLowNumberOfPointsAction);
    addAction(&_tsneComputationAction);

    _numNewIterationsAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _exaggerationIterAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _exponentialDecayAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _numComputedIterationsAction.setDefaultWidgetFlags(IntegralAction::LineEdit);
    _numNewIterationsAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _numDefaultUpdateIterationsAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _iterationsPublishExtendAction.setDefaultWidgetFlags(IntegralAction::SpinBox);

    _numDefaultUpdateIterationsAction.initialize(0, 10000, 1000u);
    _numNewIterationsAction.initialize(0, 10000, 0);
    _iterationsPublishExtendAction.initialize(1, 10000, 250);
    _exaggerationIterAction.initialize(0, 10000, 250);
    _exponentialDecayAction.initialize(0, 10000, 70);
    _exaggerationFactorAction.initialize(0, 100, 4, 2);
    _gradientDescentTypeAction.initialize({ "GPU (Compute)", "GPU (Raster)", "CPU" });
    _initAction.initialize({ "Random", "PCA", "Spectral" }, "Random");

    _numComputedIterationsAction.initialize(0, 100000, 0);
    _numComputedIterationsAction.setEnabled(false);

    _exaggerationToggleAction.setChecked(true);
    _exaggerationToggleAction.setToolTip("Auto val is: 4 + (number of embedded points) / 60000.0");
    _exaggerationFactorAction.setEnabled(false);

    _iterationsPublishExtendAction.setToolTip("Should be larger or equal to number of exaggeration iterations");
    _publishExtendsOnceAction.setToolTip("Only set the reference extends once, when computing the top level embedding first");
    _gradientDescentTypeAction.setToolTip("Gradient Descent Implementation: GPU (Compute, A-tSNE),  GPU (Raster, A-tSNE), CPU (Barnes-Hut)");
    _ignoreAdjustToLowNumberOfPointsAction.setToolTip("For low number of points CPU GD is automaticallty set.\nThis options prevents that adjustment.");

    const auto updateNumIterations = [this]() -> void {
        _tsneParameters.numIterations = _numDefaultUpdateIterationsAction.getValue();
        };

    const auto updateExaggerationIter = [this]() -> void {
        _tsneParameters.gradDescentParams._remove_exaggeration_iter = _exaggerationIterAction.getValue();
        _tsneParameters.gradDescentParams._mom_switching_iter = _exaggerationIterAction.getValue();
        };

    const auto updateExponentialDecay = [this]() -> void {
        _tsneParameters.gradDescentParams._exponential_decay_iter = _exponentialDecayAction.getValue();
        };

    const auto updateExaggerationFactor = [this]() -> void {
        double exaggeration = -1.0; // auto exaggeration is computed in TsneAnalysis.cpp based on number of landmarks in scale

        if (_exaggerationToggleAction.isChecked())
            _exaggerationFactorAction.setValue(0);
        else
            exaggeration = _exaggerationFactorAction.getValue();

        _tsneParameters.gradDescentParams._exaggeration_factor = exaggeration;
        };

    const auto updateGradientDescentTypeAction = [this]() -> void {
        switch (_gradientDescentTypeAction.getCurrentIndex())
        {
        case 0: _tsneParameters.gradientDescentType = GradientDescentType::GPUcompute; break;
        case 1: _tsneParameters.gradientDescentType = GradientDescentType::GPUraster; break;
        case 2: _tsneParameters.gradientDescentType = GradientDescentType::CPU; break;
        }

        };

    const auto updateReadOnly = [this]() -> void {
        auto enable = !isReadOnly();

        _numNewIterationsAction.setEnabled(enable);
        _numDefaultUpdateIterationsAction.setEnabled(enable);
        _iterationsPublishExtendAction.setEnabled(enable);
        _publishExtendsOnceAction.setEnabled(enable);
        _initAction.setEnabled(enable);
        _exaggerationIterAction.setEnabled(enable);
        _exaggerationFactorAction.setEnabled(enable);
        _exaggerationToggleAction.setEnabled(enable);
        _exponentialDecayAction.setEnabled(enable);
        _gradientDescentTypeAction.setEnabled(enable);
        _ignoreAdjustToLowNumberOfPointsAction.setEnabled(enable);

        if ((_numComputedIterationsAction.getValue() > 0) && _publishExtendsOnceAction.isChecked())
            _iterationsPublishExtendAction.setEnabled(false);

        if (_numNewIterationsAction.getValue() == 0)
            enable = false;
        };

    connect(&_numDefaultUpdateIterationsAction, &IntegralAction::valueChanged, this, [this, updateNumIterations](const std::int32_t& value) {
        updateNumIterations();
        });

    connect(&_numComputedIterationsAction, &IntegralAction::valueChanged, this, [this](const std::int32_t& value) {

        if (_publishExtendsOnceAction.isChecked())
            _iterationsPublishExtendAction.setEnabled(false);

        });

    connect(&_publishExtendsOnceAction, &ToggleAction::toggled, this, [this](const bool& val) {

        if (_numComputedIterationsAction.getValue() == 0)
            return;

        _iterationsPublishExtendAction.setEnabled(!_publishExtendsOnceAction.isChecked());

        });

    connect(&_exaggerationIterAction, &IntegralAction::valueChanged, this, [this, updateExaggerationIter](const std::int32_t& value) {
        updateExaggerationIter();
        });

    connect(&_exponentialDecayAction, &IntegralAction::valueChanged, this, [this, updateExponentialDecay](const std::int32_t& value) {
        updateExponentialDecay();
        });

    connect(&_exaggerationFactorAction, &DecimalAction::valueChanged, this, [this, updateExaggerationFactor](const float& value) {
        updateExaggerationFactor();
        });

    connect(&_exaggerationToggleAction, &IntegralAction::toggled, this, [this, updateExaggerationFactor](const bool toggled) {
        _exaggerationFactorAction.setEnabled(!toggled);
        updateExaggerationFactor();
        });

    connect(&_gradientDescentTypeAction, &OptionAction::currentIndexChanged, this, [this, updateGradientDescentTypeAction](const std::int32_t& currentIndex) {
        updateGradientDescentTypeAction();
        });

    connect(this, &GroupAction::readOnlyChanged, this, [this, updateReadOnly](const bool& readOnly) {
        updateReadOnly();
        });

}

void TsneSettingsAction::adjustToLowNumberOfPoints(size_t numEmbPoints) {

    if (_ignoreAdjustToLowNumberOfPointsAction.isChecked()) {
        return;
    }

    if (numEmbPoints < 500) {
        _gradientDescentTypeAction.setCurrentIndex(2);
        _numDefaultUpdateIterationsAction.setValue(500);
    }
    else {
        _gradientDescentTypeAction.setCurrentIndex(0);

        if (numEmbPoints < 100'000)
            _numDefaultUpdateIterationsAction.setValue(1000);
        else if (numEmbPoints < 200'000)
            _numDefaultUpdateIterationsAction.setValue(2000);
        else
            _numDefaultUpdateIterationsAction.setValue(4000);
    }

}
