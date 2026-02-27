#pragma once

#include <sph/EmbedTsne.hpp>

#include "TsneComputationAction.h"

#include <actions/DecimalAction.h>
#include <actions/GroupAction.h>
#include <actions/IntegralAction.h>
#include <actions/OptionAction.h>
#include <actions/ToggleAction.h>

#include <string>

/// ////////////////// ///
/// TsneSettingsAction ///
/// ////////////////// ///

class TsneSettingsAction : public mv::gui::GroupAction
{
public:

    TsneSettingsAction(QObject* parent, std::string title = "TSNE");

    sph::TsneEmbeddingParameters& getTsneParameters() { return _tsneParameters; }

    void adjustToLowNumberOfPoints(size_t numEmbPoints);

public: // Action getters

    mv::gui::IntegralAction& getExaggerationIterAction() { return _exaggerationIterAction; };
    mv::gui::IntegralAction& getExponentialDecayAction() { return _exponentialDecayAction; };
    mv::gui::DecimalAction& getExaggerationFactorAction() { return _exaggerationFactorAction; };
    mv::gui::ToggleAction& getExaggerationToggleAction() { return _exaggerationToggleAction; };
    mv::gui::IntegralAction& getIterationsPublishExtendAction() { return _iterationsPublishExtendAction; };
    mv::gui::ToggleAction& getPublishExtendsOnceAction() { return _publishExtendsOnceAction; };
    mv::gui::OptionAction& getInitAction() { return _initAction; };
    mv::gui::IntegralAction& getNumNewIterationsAction() { return _numNewIterationsAction; };
    mv::gui::IntegralAction& getNumDefaultUpdateIterationsAction() { return _numDefaultUpdateIterationsAction; };
    mv::gui::IntegralAction& getNumComputedIterationsAction() { return _numComputedIterationsAction; };
    mv::gui::OptionAction& getGradientDescentTypeAction() { return _gradientDescentTypeAction; };
    mv::gui::ToggleAction& getIgnoreAdjustToLowNumberOfPointsAction() { return _ignoreAdjustToLowNumberOfPointsAction; };
    TsneComputationAction& getTsneComputeAction() { return _tsneComputationAction; }

private:
    sph::TsneEmbeddingParameters    _tsneParameters;                        /** TSNE parameters */
    mv::gui::IntegralAction         _exaggerationIterAction;                /** Exaggeration iteration action */
    mv::gui::IntegralAction         _exponentialDecayAction;                /** Exponential decay of exaggeration action */
    mv::gui::DecimalAction          _exaggerationFactorAction;              /** Exaggeration factor action */
    mv::gui::ToggleAction           _exaggerationToggleAction;              /** Exaggeration toggle action */
    mv::gui::IntegralAction         _iterationsPublishExtendAction;         /** Number of iterations at which to publish reference extends action */
    mv::gui::ToggleAction           _publishExtendsOnceAction;              /** Whether reference extends should only be set once, when the top level is computed */
    mv::gui::OptionAction           _initAction;                            /** Whether to initialize embedding with PCA, Spectral or Random */
    mv::gui::IntegralAction         _numNewIterationsAction;                /** Number of new iterations action */
    mv::gui::IntegralAction         _numDefaultUpdateIterationsAction;      /** Number of default update iterations action */
    mv::gui::IntegralAction         _numComputedIterationsAction;           /** Number of computed iterations action */
    mv::gui::OptionAction           _gradientDescentTypeAction;             /** GPU or CPU gradient descent */
    mv::gui::ToggleAction           _ignoreAdjustToLowNumberOfPointsAction; /** Whether to allow adjustToLowNumberOfPoints making adjustments */
    TsneComputationAction           _tsneComputationAction;                 /** t-SNE embedding compute action */
};
