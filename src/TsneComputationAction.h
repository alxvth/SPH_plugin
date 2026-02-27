#pragma once

#include <actions/HorizontalGroupAction.h>
#include <actions/TriggerAction.h>

class QMenu;

class TsneComputationAction : public mv::gui::HorizontalGroupAction
{
public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     */
    TsneComputationAction(QObject* parent);

    /**
     * Get the context menu for the action
     * @param parent Parent widget
     * @return Context menu
     */
    QMenu* getContextMenu(QWidget* parent = nullptr) override;

    void changeEnabled(bool cont, bool stop)
    {
        _continueComputationAction.setEnabled(cont);
        _stopComputationAction.setEnabled(stop);
    }

    void setStarted()
    {
        changeEnabled(false, true);
    }

    void setFinished()
    {
        changeEnabled(true, false);
    }

public: // Action getters

    mv::gui::TriggerAction& getContinueComputationAction() { return _continueComputationAction; }
    mv::gui::TriggerAction& getStopComputationAction() { return _stopComputationAction; }
    mv::gui::TriggerAction& getRestartComputationAction() { return _restartComputationAction; }

private:
    mv::gui::TriggerAction   _continueComputationAction;     /** Continue computation action */
    mv::gui::TriggerAction   _stopComputationAction;         /** Stop computation action */
    mv::gui::TriggerAction   _restartComputationAction;      /** Stop computation action */
};
