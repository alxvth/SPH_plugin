#include "SettingsAdvancedAction.h"

#include <sph/NearestNeighbors.hpp>

using namespace sph;

AdvancedSettingsAction::AdvancedSettingsAction(QObject* parent) :
    GroupAction(parent, "AdvancedSettingsAction", false),
    _numDataPoints(0),
    _knnIndexTypeAction(this, "Faiss Index"),
    _clampDataAction(this, "Data clamp top %"),
    _pruneTransitionValueAction(this, "Prune value"),
    _pruneTransitionStepsAction(this, "Prune steps"),
    _weightRWbySize(this, "Weight RW by size", false),
    _numGeodesicSamplesAction(this, "Geo samples"),
    _minReductionAction(this, "Min reduction"),
    _exactKnnAction(this, "Exact Knn", false),
    _normDataAction(this, "Norm Data"),
    _symmetricKnnAction(this, "Symmetric kNN", true),
    _connectedKnnAction(this, "Connected kNN", true), 
    _alwaysMergeAction(this, "Always merge", false),
    _percentileOrValeAction(this, "Min sim is percentile", false),
    _mergeWithAllAboveAction(this, "Merge with multiple", false),
    _maxDistAction(this, "Minimum Sim", 0.f, 1.f, 0.f, 3),
    _randomWalkReductionAction(this, "RW reduciton"),
    _normSchemeAction(this, "Norm scheme")
{
    setText("Advanced");
    setObjectName("Advanced");

    /// UI set up: add actions
    addAction(&_normDataAction);
    addAction(&_symmetricKnnAction);
    addAction(&_connectedKnnAction);
    addAction(&_knnIndexTypeAction);
    addAction(&_normSchemeAction);
    addAction(&_randomWalkReductionAction);
    addAction(&_clampDataAction);
    addAction(&_pruneTransitionValueAction);
    addAction(&_pruneTransitionStepsAction);
    addAction(&_weightRWbySize);
    addAction(&_numGeodesicSamplesAction);
    addAction(&_minReductionAction);
    addAction(&_alwaysMergeAction);
    addAction(&_mergeWithAllAboveAction);
    addAction(&_percentileOrValeAction);
    addAction(&_maxDistAction);

    _knnIndexTypeAction.setToolTip("knn index:\n>10'000: IVFFlat\n>100'000: HNSW\n >1'000'000 IVFFlat_HNSW\n>50'000'000: HNSW_IVFPQ\nsmall data: BruteForce");
    _randomWalkReductionAction.setToolTip("Random walk reduction setting");
    _normSchemeAction.setToolTip("Norm for t-SNE or UMAP");
    _pruneTransitionValueAction.setToolTip("Threshold below which transition values (random walk values) are ignored (set by Prune steps)");
    _pruneTransitionStepsAction.setToolTip("Steps below which transition values (random walk values) are ignored (sets Prune values)");
    _weightRWbySize.setToolTip("Weight random walks merging by component size");
    _numGeodesicSamplesAction.setToolTip("Number of samples for geodesic Hausdorff distance, 0 means all");
    _clampDataAction.setToolTip("Clamp top x% of data values");
    _minReductionAction.setToolTip("Minimum reduction percentage");
    _exactKnnAction.setToolTip("Whether to compute exact knn (or approximated)");
    _normDataAction.setToolTip("Normalize input data.\nSTANADRD: z = (x - u) / s [channel-wise]\nROBUST:clamps data to 95% and normalizes values to [0, 1] [globally]");
    _symmetricKnnAction.setToolTip("Whether to compute symmetric knn");
    _connectedKnnAction.setToolTip("Whether to compute connected components");
    _alwaysMergeAction.setToolTip("Always merge with most similar neighbor, independent of any minimum similarity value.\nIf no similarity is available, merge with a random neighbor.");
    _percentileOrValeAction.setToolTip("Interpret min sim as percentile or value");
    _mergeWithAllAboveAction.setToolTip("Merge with all spatial neighbors whose sim is above threshold.\nOtherwise merge the most similar neighbor");
    _maxDistAction.setToolTip("Maximum distance value for merging");

    _normDataAction.initialize(QStringList({ "NONE", "STANDARD", "ROBUST" }), "NONE");
    _knnIndexTypeAction.initialize(QStringList({ "BruteForce", "Flat", "IVFFlat", "HNSW", "HNSWSQ", "IVFFlat_HNSW", "HNSW_IVFPQ", "Auto" }), "Auto");
    _randomWalkReductionAction.initialize(QStringList({ "NONE", "PROPORTIONAL", "PROPORTIONAL_HALF", "PROPORTIONAL_DOUBLE", "CONSTANT", "CONSTANT_LOW", "CONSTANT_HIGH" }), "PROPORTIONAL");
    _normSchemeAction.initialize(QStringList({ "t-SNE", "UMAP" }), "t-SNE");

    _pruneTransitionValueAction.initialize(0.f, 1.0f, 0.f, 4);
    _pruneTransitionValueAction.setSingleStep(0.0001f);

    _pruneTransitionStepsAction.initialize(0, 10, 0);

    _numGeodesicSamplesAction.initialize(0, 10000, 100);

    _clampDataAction.initialize(0.f, 1.0f, 0.f, 3);
    _clampDataAction.setSingleStep(0.001f);

    _minReductionAction.initialize(0.f, 1.0f, 0.98f, 4);
    _minReductionAction.setSingleStep(0.0001f);

    _maxDistAction.setSingleStep(0.01f);
    _maxDistAction.setEnabled(true);

    const auto updateReadOnly = [this]() -> void {
        const auto enabled = !isReadOnly();

        _knnIndexTypeAction.setEnabled(enabled);
        _randomWalkReductionAction.setEnabled(enabled);
        _normSchemeAction.setEnabled(enabled);
        _pruneTransitionValueAction.setEnabled(enabled);
        _pruneTransitionStepsAction.setEnabled(enabled);
        _weightRWbySize.setEnabled(enabled);
        _numGeodesicSamplesAction.setEnabled(enabled);
        _clampDataAction.setEnabled(enabled);
        _maxDistAction.setEnabled(enabled);
        _minReductionAction.setEnabled(enabled);
        _mergeWithAllAboveAction.setEnabled(enabled);
        _percentileOrValeAction.setEnabled(enabled);
        _normDataAction.setEnabled(enabled);
        _alwaysMergeAction.setEnabled(enabled);
        _connectedKnnAction.setEnabled(enabled);
        _symmetricKnnAction.setEnabled(enabled);

        };

    connect(this, &GroupAction::readOnlyChanged, this, [this, updateReadOnly](const bool& readOnly) {
        updateReadOnly();
        });

    connect(&_alwaysMergeAction, &ToggleAction::toggled, this, [this](bool toggled) {
        _maxDistAction.setEnabled(!toggled);
        _percentileOrValeAction.setEnabled(!toggled);

        if (_alwaysMergeAction.isChecked())
            _mergeWithAllAboveAction.setChecked(false);
        });


    updateReadOnly();
}

utils::KnnIndex AdvancedSettingsAction::getDataIndexSetting() const
{
    QString currentOption = _knnIndexTypeAction.getCurrentText();
    auto allOptions = _knnIndexTypeAction.getOptions();

    auto indexType = utils::KnnIndex::BruteForce;

    if (currentOption == "Auto") {
        indexType = sph::NearestNeighbors::indexHeuristic(_numDataPoints);
    }
    else {
#ifdef DEBUG
        using underlying_type = std::underlying_type<KnnIndex>::type;
        constexpr underlying_type maxLevel = static_cast<underlying_type>(KnnIndex::HNSW_IVFPQ);

        assert(allOptions.indexOf(currentOption) <= maxLevel);
#endif // DEBUG

        indexType = static_cast<utils::KnnIndex>(allOptions.indexOf(currentOption));
    }

    return indexType;
}

float AdvancedSettingsAction::getMaxDistanceSetting() const
{
    float minSim = _maxDistAction.getValue();

    if (_alwaysMergeAction.isChecked())
        minSim = -1.f;

    return minSim;
}
