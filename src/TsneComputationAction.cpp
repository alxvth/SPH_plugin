#include "TsneComputationAction.h"

#include <QMenu>

TsneComputationAction::TsneComputationAction(QObject* parent) :
    HorizontalGroupAction(parent, "TsneComputationAction"),
    _continueComputationAction(this, "Continue"),
    _stopComputationAction(this, "Stop"),
    _restartComputationAction(this, "Restart")
{
    setText("Computation");

    addAction(&_continueComputationAction);
    addAction(&_stopComputationAction);
    addAction(&_restartComputationAction);

    _continueComputationAction.setToolTip("Continue with the t-SNE computation");
    _stopComputationAction.setToolTip("Stop the current t-SNE computation");
    _restartComputationAction.setToolTip("Restart with new gradient descent settings");

    _continueComputationAction.setEnabled(false);
}

QMenu* TsneComputationAction::getContextMenu(QWidget* parent /*= nullptr*/)
{
    auto menu = new QMenu(text(), parent);

    menu->addAction(&_continueComputationAction);
    menu->addAction(&_stopComputationAction);

    return menu;
}
