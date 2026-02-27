#include "RefinedSelectionMapping.h"

#include "Utils.h"

#include <sph/utils/Logger.hpp>

#include <Dataset.h>
#include <Set.h>

using namespace sph;

RefinedSelectionMapping::RefinedSelectionMapping(QObject* parent) :
    WidgetAction(parent, "RefinedSelectionMapping")
{
    setText("RefinedSelectionMapping");
    setObjectName("RefinedSelectionMapping");
}

void RefinedSelectionMapping::setInputData(const mv::Dataset<Points>& input) 
{ 
    _inputData = input; 
    connect(&_inputData, &mv::Dataset<mv::DatasetImpl>::dataSelectionChanged, this, &RefinedSelectionMapping::onSelectionInInputData);
}

void RefinedSelectionMapping::setEmbeddingData(const mv::Dataset<Points>& emb) 
{ 
    _levelEmbedding = emb;
    connect(&_levelEmbedding, &mv::Dataset<mv::DatasetImpl>::dataSelectionChanged, this, &RefinedSelectionMapping::onSelectionInLevelEmbedding);
}

void RefinedSelectionMapping::setImgColoredByEmb(const mv::Dataset<Points>& col)
{ 
    _dataColoredByLevelEmb = col;
    connect(&_dataColoredByLevelEmb, &mv::Dataset<mv::DatasetImpl>::dataSelectionChanged, this, &RefinedSelectionMapping::onSelectionInColoredByEmb);
}

void RefinedSelectionMapping::setAvgComponentDataPixel(const mv::Dataset<Points>& avgs)
{ 
    _avgComponentDataPixel = avgs;
    connect(&_avgComponentDataPixel, &mv::Dataset<mv::DatasetImpl>::dataSelectionChanged, this, &RefinedSelectionMapping::onSelectionInPixelAverages);
}

void RefinedSelectionMapping::onSelectionInInputData()
{
    const bool allIsSync = areLocksInSync();
    const bool toBeHandled = isNotYetHandled(SelectionDatasets::INPUT);
    const bool doNothing = !allIsSync && !toBeHandled;

    if (doNothing)
        return;

    Log::trace("onSelectionInInputData");

    if (allIsSync)
        markAsHandled(SelectionDatasets::GLOBAL);

    markAsHandled(SelectionDatasets::INPUT);

    if (isNotYetHandled(SelectionDatasets::EMBEDDING))
        selectionMapping(_inputData, &_mappingDataToLevel, _levelEmbedding);

    if (isNotYetHandled(SelectionDatasets::RECOLOR_IMAGE))
        copySelection(_inputData, _dataColoredByLevelEmb);

    if (isNotYetHandled(SelectionDatasets::AVERAGES))
        copySelection(_inputData, _avgComponentDataPixel);

}

void RefinedSelectionMapping::onSelectionInLevelEmbedding()
{
    const bool allIsSync = areLocksInSync();
    const bool toBeHandled = isNotYetHandled(SelectionDatasets::EMBEDDING);
    const bool doNothing = !allIsSync && !toBeHandled;

    if (doNothing)
        return;

    Log::trace("onSelectionInInputData");

    if (allIsSync)
        markAsHandled(SelectionDatasets::GLOBAL);

    markAsHandled(SelectionDatasets::EMBEDDING);

    if (isNotYetHandled(SelectionDatasets::INPUT))
        selectionMapping(_levelEmbedding, &_mappingLevelToData, _inputData);

    if (isNotYetHandled(SelectionDatasets::RECOLOR_IMAGE))
        copySelection(_inputData, _dataColoredByLevelEmb);

    if (isNotYetHandled(SelectionDatasets::AVERAGES))
        copySelection(_inputData, _avgComponentDataPixel);
}

void RefinedSelectionMapping::onSelectionInColoredByEmb()
{
    const bool allIsSync = areLocksInSync();
    const bool toBeHandled = isNotYetHandled(SelectionDatasets::RECOLOR_IMAGE);
    const bool doNothing = !allIsSync && !toBeHandled;

    if (doNothing)
        return;

    Log::trace("onSelectionInInputData");

    if (allIsSync)
        markAsHandled(SelectionDatasets::GLOBAL);

    markAsHandled(SelectionDatasets::RECOLOR_IMAGE);

    if (isNotYetHandled(SelectionDatasets::EMBEDDING))
        selectionMapping(_dataColoredByLevelEmb, &_mappingDataToLevel, _levelEmbedding);

    if (isNotYetHandled(SelectionDatasets::INPUT))
        selectionMapping(_levelEmbedding, &_mappingLevelToData, _inputData);

    if (isNotYetHandled(SelectionDatasets::AVERAGES))
        copySelection(_inputData, _avgComponentDataPixel);

}

void RefinedSelectionMapping::onSelectionInPixelAverages()
{
    const bool allIsSync = areLocksInSync();
    const bool toBeHandled = isNotYetHandled(SelectionDatasets::AVERAGES);
    const bool doNothing = !allIsSync && !toBeHandled;

    if (doNothing)
        return;

    Log::trace("onSelectionInPixelAverages");

    if (allIsSync)
        markAsHandled(SelectionDatasets::GLOBAL);

    markAsHandled(SelectionDatasets::AVERAGES);

    if (isNotYetHandled(SelectionDatasets::EMBEDDING))
        selectionMapping(_avgComponentDataPixel, &_mappingDataToLevel, _levelEmbedding);

    if (isNotYetHandled(SelectionDatasets::INPUT))
        selectionMapping(_levelEmbedding, &_mappingLevelToData, _inputData);

    if (isNotYetHandled(SelectionDatasets::RECOLOR_IMAGE))
        copySelection(_inputData, _dataColoredByLevelEmb);

}
