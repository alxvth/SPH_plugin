#pragma once

#include <actions/DecimalAction.h>
#include <actions/GroupAction.h>
#include <actions/IntegralAction.h>
#include <actions/OptionAction.h>
#include <actions/ToggleAction.h>
#include <actions/TriggerAction.h>
#include <actions/WidgetAction.h>
#include <actions/WidgetActionWidget.h>

#include <sph/utils/Settings.hpp>

using namespace mv::gui;

/// ////////////////// ///
/// GUI: Level Up&Down ///
/// ////////////////// ///

class LevelDownUpActions : public WidgetAction
{
    Q_OBJECT

protected:
    /** Widget class for TSNE computation action */
    class Widget : public WidgetActionWidget {
    public:

        /**
         * Constructor
         * @param parent Pointer to parent widget
         * @param tsneComputationAction Pointer to TSNE computation action
         */
        Widget(QWidget* parent, LevelDownUpActions* scaleDownUpActions);
    };

    /**
     * Get widget representation of the TSNE computation action
     * @param parent Pointer to parent widget
     * @param widgetFlags Widget flags for the configuration of the widget (type)
     */
    QWidget* getWidget(QWidget* parent, const std::int32_t& widgetFlags) override {
        return new Widget(parent, this);
    };

public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     */
    LevelDownUpActions(QObject* parent);

    void setNumLevels(size_t numLevels);

    void setLevel(size_t currentScale);

signals:
    void levelChanged(int32_t newLevel);

protected: // Action getters

    IntegralAction& getLevelAction() { return _levelAction; }
    TriggerAction& getUpAction() { return _upAction; }
    TriggerAction& getDownAction() { return _downAction; }

private:
    IntegralAction          _levelAction;
    TriggerAction           _upAction;         /** Go a level up   */
    TriggerAction           _downAction;       /** Go a level down */

    size_t                  _numLevels;         /** Total number scales*/
};


/// ///////////////// ///
/// HierarchySettings ///
/// ///////////////// ///

class HierarchySettings : public GroupAction
{
public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     */
    HierarchySettings(QObject* parent);

    sph::utils::NeighConnection getNeighConnectionSetting() const;
    sph::utils::KnnMetric getDataMetricSetting() const;
    sph::utils::ComponentSim getComponentSimSetting() const;
    sph::utils::RandomWalkHandling getRandomWalkHandlingSetting() const;

public: // Setter

    /** Enable/disables UI buttons for going up and down accordingly */
    void setCurrentLevel(int64_t level, int64_t maxLevel);
    void setNumDataPoints(int64_t n);

public: // Action getters

    OptionAction& getNeighConnectivityOption() { return _neighConnectivityAction; }
    OptionAction& getDataKnnMetricAction() { return _dataKnnMetricAction; }
    OptionAction& getComponentSimAction() { return _componentSimAction; }
    IntegralAction& getMinComponentsSlider() { return _minComponentsAction; }
    IntegralAction& getNumRandomWalkSlider() { return _numRandomWalkAction; }
    IntegralAction& getLenRandomWalkSlider() { return _lenRandomWalkAction; }
    OptionAction& getWeightingRandomWalkOption() { return _weightRandomWalkAction; }
    OptionAction& getHandleRandomWalkAction() { return _handleRandomWalkAction; }
    ToggleAction& getRandomWalkPairSims() { return _randomWalkPairSimsAction; }
    IntegralAction& getNumDataKnnSlider() { return _numDataKnn; }
    TriggerAction& getStartAnalysisButton() { return _startAnalysisAction; }
    LevelDownUpActions& getLevelDownUpActions() { return _levelUpDownActions; }
    ToggleAction& getCachingActiveAction() { return _cachingActiveAction; }

private:
    OptionAction            _neighConnectivityAction;       /** Neighborhood connectivity */
    OptionAction            _dataKnnMetricAction;           /** Data level distance measure */
    OptionAction            _componentSimAction;            /** Superpixel component distance measure */
    IntegralAction          _minComponentsAction;           /** Minimal number components */
    IntegralAction          _numRandomWalkAction;           /** Random walk number */
    IntegralAction          _lenRandomWalkAction;           /** Random walk length */
    OptionAction            _weightRandomWalkAction;        /** Random walk step weighting */
    OptionAction            _handleRandomWalkAction;        /** Random walk handling */
    ToggleAction            _randomWalkPairSimsAction;      /** Similarities from random walks */
    IntegralAction          _numDataKnn;                    /** Number of k nearest neighbors on data level */
    TriggerAction           _startAnalysisAction;           /** Start computation */
    LevelDownUpActions      _levelUpDownActions;            /** Level Up and Down actions */
    ToggleAction            _cachingActiveAction;           /** Whether results should be loaded and saved to disk */

    bool                    _lockComponentsSlider;          /** Internal lock for slider */
    int64_t                 _numDataPoints;                 /** Currently set goal for number of components */
};
