#pragma once

#include "DimensionSelectionAction.h"
#include "RefineAction.h"
#include "SettingsAdvancedAction.h"
#include "SettingsHierarchyAction.h"
#include "SettingsTsneAction.h"

#include <actions/GroupAction.h>

class SPHPlugin;

/**
 * SphSettingsAction class
 */
class SettingsAction : public mv::gui::GroupAction
{
public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     */
    SettingsAction(SPHPlugin* parent = nullptr);

public: // Action getters

    HierarchySettings& getHierarchySettingsAction() { return _hierarchySettingsAction; }
    TsneSettingsAction& getTsneSettingsAction() { return _tsneSettingsAction; }
    AdvancedSettingsAction& getAdvancedSettingsAction() { return _advancedSettingsAction; }
    DimensionSelectionAction& getDimensionSelectionAction() { return _dimensionSelectionAction; }
    RefineAction& getRefineAction() { return _refineAction; }
    TsneSettingsAction& getRefineTsneSettingsAction() { return _refineTsneSettingsAction; }

private:
    HierarchySettings           _hierarchySettingsAction;   /** Hierarchy settings action */
    TsneSettingsAction          _tsneSettingsAction;        /** t-SNE embedding settings action */
    AdvancedSettingsAction      _advancedSettingsAction;    /** Advanced settings action */
    DimensionSelectionAction    _dimensionSelectionAction;  /** Dimension selection action */
    RefineAction                _refineAction;              /** Refine action */
    TsneSettingsAction          _refineTsneSettingsAction;  /** Refine t-SNE embedding settings action */
};
