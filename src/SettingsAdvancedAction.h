#pragma once

#include <actions/DecimalAction.h>
#include <actions/GroupAction.h>
#include <actions/IntegralAction.h>
#include <actions/OptionAction.h>
#include <actions/ToggleAction.h>

#include <sph/utils/Settings.hpp>

#include <cstdint>

using namespace mv::gui;

/// ////////////////////// ///
/// AdvancedSettingsAction ///
/// ////////////////////// ///

class AdvancedSettingsAction : public GroupAction
{
public:

    AdvancedSettingsAction(QObject* parent);

    sph::utils::KnnIndex getDataIndexSetting() const;
    float getMaxDistanceSetting() const;

public: // Setter
    void setNumDataPoints(int64_t numberDataPoints) { _numDataPoints = numberDataPoints; }

public: // Action getters

    OptionAction& getNormDataAction() { return _normDataAction; }
    ToggleAction& getSymmetricKnnAction() { return _symmetricKnnAction; }
    ToggleAction& getConnectedKnnAction() { return _connectedKnnAction; }
    OptionAction& getKnnIndexTypeAction() { return _knnIndexTypeAction; }
    DecimalAction& getPruneTransitionsValueAction() { return _pruneTransitionValueAction; }
    IntegralAction& getPruneTransitionsStepsAction() { return _pruneTransitionStepsAction; }
    ToggleAction& getWeightRWbySize() { return _weightRWbySize; }
    IntegralAction& getNumGeodesicSamplesAction() { return _numGeodesicSamplesAction; }
    DecimalAction& getClampDataAction() { return _clampDataAction; }
    DecimalAction& getMinReductionAction() { return _minReductionAction; }
    ToggleAction& getExactKnnAction() { return _exactKnnAction; }
    ToggleAction& getAlwaysMergeToggle() { return _alwaysMergeAction; }
    ToggleAction& getPercentileOrValeAction() { return _percentileOrValeAction; }
    ToggleAction& getMergeWithAllAboveToggle() { return _mergeWithAllAboveAction; }
    DecimalAction& getMaxDistanceSlider() { return _maxDistAction; }
    OptionAction& getRandomWalkReductionAction() { return _randomWalkReductionAction; }
    OptionAction& getNormSchemeAction() { return _normSchemeAction; }

protected:
    OptionAction            _normDataAction;                /** Whether to normalize the data  */
    ToggleAction            _symmetricKnnAction;            /** Compute symmetric knn  */
    ToggleAction            _connectedKnnAction;            /** Compute connected knn  */
    OptionAction            _knnIndexTypeAction;            /** knn (faiss) index */
    DecimalAction           _pruneTransitionValueAction;    /** Threshold below which transition values (random walk values) are ignored */
    IntegralAction          _pruneTransitionStepsAction;    /** Threshold below which transition values (random walk values) are ignored */
    ToggleAction            _weightRWbySize;                /** Weight random walks merging by component size */
    IntegralAction          _numGeodesicSamplesAction;      /** Number of geodesic samples */
    DecimalAction           _clampDataAction;               /** Clamp top x% of data values */
    DecimalAction           _minReductionAction;            /** Min Reduction Percentage */
    ToggleAction            _exactKnnAction;                /** Whether to compute exact knn */
    ToggleAction            _alwaysMergeAction;             /** Similarity should be larger zero */
    ToggleAction            _percentileOrValeAction;        /** Interpret min sim as percentile or value */
    ToggleAction            _mergeWithAllAboveAction;       /** Merge with all neighbors above the threshold */
    DecimalAction           _maxDistAction;                 /** Minimal similarity */
    OptionAction            _randomWalkReductionAction;     /** RandomWalk Reduction */
    OptionAction            _normSchemeAction;              /** Whether to norm data for t-SNE or UMAP */
    
private:
    int64_t                 _numDataPoints;
};
