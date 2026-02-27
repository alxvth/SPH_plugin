#include "SettingsHierarchyAction.h"

#include <sph/utils/Logger.hpp>

#include <array>

#include <QGridLayout>

using namespace sph;

HierarchySettings::HierarchySettings(QObject* parent) :
    GroupAction(parent, "Spatial Hierarchy", true),
    _minComponentsAction(this, "Minimum Comp", 1, 150'000, 5000),
    _numRandomWalkAction(this, "Num walks", 0,  1000, 50),
    _lenRandomWalkAction(this, "Length walks", 0,  1000, 25),
    _weightRandomWalkAction(this, "Rnd walk weight"),
    _handleRandomWalkAction(this, "Rnd handling"),
    _randomWalkPairSimsAction(this, "Sim pairs", true),
    _numDataKnn(this, "Num data knn", 0, 500, 0),
    _neighConnectivityAction(this, "Connectivity"),
    _dataKnnMetricAction(this, "Data knn Metric"),
    _componentSimAction(this, "Comp knn Metric"),
    _startAnalysisAction(this, "Start"),
    _cachingActiveAction(this, "Caching active", true),
    _levelUpDownActions(this),
    _lockComponentsSlider(false), 
    _numDataPoints(0)
{
    /// UI set up: add actions
    addAction(&_neighConnectivityAction);
    addAction(&_dataKnnMetricAction);
    addAction(&_componentSimAction);
    addAction(&_numDataKnn);
    addAction(&_minComponentsAction);
    addAction(&_numRandomWalkAction);
    addAction(&_lenRandomWalkAction);
    addAction(&_weightRandomWalkAction);
    addAction(&_handleRandomWalkAction);
    addAction(&_randomWalkPairSimsAction);
    addAction(&_cachingActiveAction);
    addAction(&_startAnalysisAction);
    addAction(&_levelUpDownActions);

    _minComponentsAction.setToolTip("Minimum number of components on highest hierarchy level");
    _randomWalkPairSimsAction.setToolTip("Embedding sims based on random walk pair sims");
    _numRandomWalkAction.setToolTip("Number of random walks");
    _lenRandomWalkAction.setToolTip("Length of each random walk");
    _weightRandomWalkAction.setToolTip("Weighting of each step in random walks");
    _numDataKnn.setToolTip("Number of nearest neighbors on data level");
    _neighConnectivityAction.setToolTip("Number of spatial connections");
    _dataKnnMetricAction.setToolTip("Metric on data level");
    _componentSimAction.setToolTip("Similarity measure between superpixel components");
    _startAnalysisAction.setToolTip("Start the analysis");
    _cachingActiveAction.setToolTip("Whether to load and save results from and to disk");

    _neighConnectivityAction.initialize(QStringList({ "Four", "Eight" }));
    _neighConnectivityAction.setCurrentIndex(1);

    _dataKnnMetricAction.initialize(QStringList({ "L2", "Cosine", "Inner product"}));
    _dataKnnMetricAction.setCurrentIndex(0);

    _componentSimAction.initialize(QStringList({ "Neighborhood Overlap", "Geodesic Distance", "Random Walks", "Geodesic Walks", "Random Walks (Single Overlap)", "Euclidean Set"}));
    _componentSimAction.setCurrentIndex(2);

    _weightRandomWalkAction.initialize(QStringList({ "Constant", "Linear", "Normal", "Only last", "First visit"}));
    _weightRandomWalkAction.setCurrentIndex(2);

    _handleRandomWalkAction.initialize(QStringList({ "MERGE_RW_ONLY", "MERGE_RW_NEW_WALKS", "MERGE_RW_NEW_WALKS_AND_KNN", "MERGE_DATA_NEW_WALKS" }));
    _handleRandomWalkAction.setCurrentIndex(0);

    auto updateEnableUI = [this](const int32_t newOption) {
        constexpr std::array<int32_t, 3> walkRelatedOptions = { 2, 3, 4 };
        bool isWalkRelated = std::any_of(walkRelatedOptions.cbegin(), walkRelatedOptions.cend(), [newOption](int32_t x) { return x == newOption; });

        _numRandomWalkAction.setEnabled(isWalkRelated);
        _lenRandomWalkAction.setEnabled(isWalkRelated);
        _weightRandomWalkAction.setEnabled(isWalkRelated);
        _handleRandomWalkAction.setEnabled(isWalkRelated);
        _randomWalkPairSimsAction.setEnabled(isWalkRelated);
        };

    connect(&_componentSimAction, &OptionAction::currentIndexChanged, [this, updateEnableUI](const int32_t newOption) {
        updateEnableUI(newOption);
        });

    updateEnableUI(_componentSimAction.getCurrentIndex());
}

void HierarchySettings::setCurrentLevel(int64_t level, int64_t maxLevel)
{
    // Enable/Disable UI buttons for going a scale up a down
    // and update UI info text
    _levelUpDownActions.setLevel(level);
}

void HierarchySettings::setNumDataPoints(int64_t n)
{ 
    _numDataPoints = n;

    if (_minComponentsAction.getMaximum() < _numDataPoints)
        _minComponentsAction.setValue(_numDataPoints);

    _minComponentsAction.setMaximum(_numDataPoints);
}

utils::NeighConnection HierarchySettings::getNeighConnectionSetting() const
{
    QString currentOption = _neighConnectivityAction.getCurrentText();
    auto allOptions = _neighConnectivityAction.getOptions();

    auto ncon = utils::NeighConnection::FOUR;

    switch (allOptions.indexOf(currentOption)) {
    case 0:
        break;
    case 1:
        ncon = utils::NeighConnection::EIGHT;
        break;
    }

    return ncon;
}

utils::KnnMetric HierarchySettings::getDataMetricSetting() const
{
    QString currentOption = _dataKnnMetricAction.getCurrentText();
    auto allOptions = _dataKnnMetricAction.getOptions();

    auto ncon = utils::KnnMetric::L2;

    switch (allOptions.indexOf(currentOption)) {
    case 0:
        break;
    case 1:
        ncon = utils::KnnMetric::COSINE;
        break;
    case 2:
        ncon = utils::KnnMetric::INNER_PRODUCT;
        break;
    }

    return ncon;
}

utils::ComponentSim HierarchySettings::getComponentSimSetting() const
{
    QString currentOption = _componentSimAction.getCurrentText();
    auto allOptions = _componentSimAction.getOptions();

    auto csim = utils::ComponentSim::NEIGH_OVERLAP;

    switch (allOptions.indexOf(currentOption)) {
    case 0:
        // set by default
        csim = utils::ComponentSim::NEIGH_OVERLAP;
        break;
    case 1:
        csim = utils::ComponentSim::GEO_CENTROID;
        break;
    case 2:
        csim = utils::ComponentSim::NEIGH_WALKS;
        break;
    case 3:
        csim = utils::ComponentSim::GEO_WALKS;
        break;
    case 4:
        csim = utils::ComponentSim::NEIGH_WALKS_SINGLE_OVERLAP;
        break;
    case 5:
        csim = utils::ComponentSim::EUCLID_CENTROID;
        break;
    }

    return csim;
}

sph::utils::RandomWalkHandling HierarchySettings::getRandomWalkHandlingSetting() const
{
    QString currentOption = _handleRandomWalkAction.getCurrentText();
    auto allOptions = _handleRandomWalkAction.getOptions();

    auto rwsim = static_cast<utils::RandomWalkHandling>(allOptions.indexOf(currentOption));

    return rwsim;
}


/// ////////////////// ///
/// GUI: Scale Up&Down ///
/// ////////////////// ///

LevelDownUpActions::LevelDownUpActions(QObject* parent) :
    WidgetAction(parent, "LevelDownUpActions"),
    _levelAction(this, "Level", 0, 1, 0),
    _upAction(this, "Up"),
    _downAction(this, "Down"),
    _numLevels(0)
{
    setText("Level");

    _levelAction.setToolTip("Jump to level");
    _upAction.setToolTip("Go a level up");
    _downAction.setToolTip("Go a level down");

    _levelAction.setDefaultWidgetFlags(IntegralAction::SpinBox);

    _levelAction.setEnabled(false);
    _upAction.setEnabled(false);
    _downAction.setEnabled(false);

    _levelAction.setSuffix(" (not initialized)");

    connect(&_levelAction, &IntegralAction::valueChanged, this, [this](int32_t val) {
        emit levelChanged(val);
        });

    connect(&_upAction, &TriggerAction::triggered, this, [this]() {
        _levelAction.setValue(_levelAction.getValue() + 1);
        });
    connect(&_downAction, &TriggerAction::triggered, this, [this]() {
        _levelAction.setValue(_levelAction.getValue() - 1);
        });
}

void LevelDownUpActions::setLevel(size_t currentScale)
{
    _upAction.setEnabled(true);
    _downAction.setEnabled(true);

    if (currentScale >= _numLevels)
        _upAction.setEnabled(false);

    if (currentScale <= 0)
        _downAction.setEnabled(false);

    _levelAction.setValue(static_cast<int32_t>(currentScale));
}

void LevelDownUpActions::setNumLevels(size_t numLevels)
{
    if (numLevels == 0)
    {
        Log::warn("LevelDownUpActions::setNumLevels: numLevels must be larger 0");
        return; 
    }

    if (_numLevels == 0)
    {
        _levelAction.setEnabled(true);
        _upAction.setEnabled(true);
        _downAction.setEnabled(true);
    }

    _numLevels = numLevels;
    _levelAction.setMaximum(static_cast<int32_t>(_numLevels - 1));
    _levelAction.setSuffix(" of " + QString::number(_numLevels - 1));
}


LevelDownUpActions::Widget::Widget(QWidget* parent, LevelDownUpActions* scaleDownUpActions) :
    WidgetActionWidget(parent, scaleDownUpActions)
{
    auto layout = new QGridLayout();

    layout->setContentsMargins(0, 0, 0, 0);

    layout->addWidget(scaleDownUpActions->getLevelAction().createWidget(this), 0, 0, 1, 2);
    layout->addWidget(scaleDownUpActions->getDownAction().createWidget(this), 1, 0);
    layout->addWidget(scaleDownUpActions->getUpAction().createWidget(this), 1, 1);

    setLayout(layout);
}
