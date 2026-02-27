#include "SphPlugin.h"

#include "Utils.h"

#include <ImageData/Images.h>
#include <PointData/DimensionsPickerAction.h>
#include <PointData/InfoAction.h>
#include <PointData/PointData.h>

#include <actions/PluginTriggerAction.h>

#include <sph/utils/CommonDefinitions.hpp>
#include <sph/utils/Embedding.hpp>
#include <sph/utils/EvalIO.hpp>
#include <sph/utils/Logger.hpp>
#include <sph/utils/Math.hpp>
#include <sph/utils/Settings.hpp>
#include <sph/utils/Scaler.hpp>
#include <sph/utils/Timer.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

Q_PLUGIN_METADATA(IID "manivault.studio.SPHPlugin")

std::chrono::steady_clock::time_point __tsneStartTime;

using namespace sph;

/// ////// ///
/// PLUGIN ///
/// ////// ///

SPHPlugin::SPHPlugin(const PluginFactory* factory) :
    AnalysisPlugin(factory)
{
}

void SPHPlugin::init()
{
    // Input and output data
    setOutputDataset(mv::data().createDataset<Points>("Points", "SPH embedding", getInputDataset()));
    auto inputImageDataset  = getInputDataset<Images>();
    _inputData              = inputImageDataset->getParent<Points>();
    auto outputDataset      = getOutputDataset<Points>();

    // Add the settings to the output data
    outputDataset->addAction(_settingsAction.getHierarchySettingsAction());
    outputDataset->addAction(_settingsAction.getTsneSettingsAction());
    outputDataset->addAction(_settingsAction.getAdvancedSettingsAction());
    outputDataset->addAction(_settingsAction.getRefineAction());
    outputDataset->addAction(_settingsAction.getRefineTsneSettingsAction());
    outputDataset->addAction(_settingsAction.getDimensionSelectionAction());

    _settingsAction.getDimensionSelectionAction().getPickerAction()->setPointsDataset(_inputData);

    // Open input data hierarchy entry to show the new output data and focus on output data
    _inputData->getDataHierarchyItem().setExpanded(true);
    _inputData->getDataHierarchyItem().deselect();
    outputDataset->getDataHierarchyItem().select();

    // Do not show data info by default to give more space to other settings
    outputDataset->_infoAction->collapse();

    // Store data meta data
    _data.numDimensions     = _inputData->getNumDimensions();
    _data.numPoints         = _inputData->getNumPoints();
    _imgSize                = inputImageDataset->getImageSize();

    // Set reference number for component ceiling
    _settingsAction.getHierarchySettingsAction().setNumDataPoints(_data.numPoints);
    _settingsAction.getAdvancedSettingsAction().setNumDataPoints(_data.numPoints);

    _settingsAction.getRefineAction().setSPHPlugin(this);
    _settingsAction.getRefineAction().setTsneSettingsAction(&_settingsAction.getRefineTsneSettingsAction());
    _settingsAction.getRefineAction().setParentEmbedding(outputDataset);

    std::vector<float> initialData;
    constexpr size_t numInitialDataDimensions = 2;
    initialData.resize(numInitialDataDimensions * _data.numPoints);

    // Set initial data (default 2 dimensions, all points at (0,0) )
    {
        outputDataset->setData(initialData.data(), _data.numPoints, numInitialDataDimensions);
        events().notifyDatasetDataChanged(outputDataset);
    }

    // Create image hierarchy dataset
    {
        _superpixelComponents = mv::data().createDataset<Points>("Points", "Superpixel Hierarchy", outputDataset);

        std::vector<float> tempData(static_cast<size_t>(_data.numPoints), 0);
        _superpixelComponents->setData(std::move(tempData), 1);
        events().notifyDatasetDataChanged(_superpixelComponents);

        _superpixelImage = mv::data().createDataset<Images>("Images", "Superpixel images", _superpixelComponents);

        _superpixelImage->setType(ImageData::Type::Stack);
        _superpixelImage->setNumberOfImages(1);
        _superpixelImage->setImageSize(_imgSize);
        _superpixelImage->setNumberOfComponentsPerPixel(1);

        events().notifyDatasetDataChanged(_superpixelImage);
    }

    // Get copy of input data
    {
        std::vector<uint32_t> enabledDimensionsIDs = getEnabledDimensions();
        _data.numDimensions = enabledDimensionsIDs.size();

        // store data from of landmarks
        _data.dataVec.resize(_data.numDimensions * _data.numPoints);
        _inputData->populateDataForDimensions<std::vector<float>, std::vector<uint32_t>>(_data.dataVec, enabledDimensionsIDs);
    }

    // Init scatter color data
    {
        _dataColoredByEmb = mv::data().createDataset<Points>("Points", "Scatter colors", outputDataset);
        _dataColoredByEmb->setData(initialData.data(), _data.numPoints, numInitialDataDimensions);
        events().notifyDatasetDataChanged(_dataColoredByEmb);

        _imgColoredByEmb = mv::data().createDataset<Images>("Images", "Scatter colors", _dataColoredByEmb);

        _imgColoredByEmb->setType(ImageData::Type::Stack);
        _imgColoredByEmb->setNumberOfImages(numInitialDataDimensions);
        _imgColoredByEmb->setImageSize(_imgSize);
        _imgColoredByEmb->setNumberOfComponentsPerPixel(1);

        events().notifyDatasetDataChanged(_imgColoredByEmb);
    }

    // Init avg pixel data and image
    {
        std::vector<float> initialAvgData(_data.numPoints * _data.numDimensions, 0);
        _avgComponentDataPixel = mv::data().createDataset<Points>("Points", "Average Data (Pixel)", outputDataset);
        _avgComponentDataPixel->setData(std::move(initialAvgData), _data.numDimensions);
        _avgComponentDataPixel->setDimensionNames(_inputData->getDimensionNames());
        events().notifyDatasetDataChanged(_avgComponentDataPixel);

        _avgComponentDataPixelImg = mv::data().createDataset<Images>("Images", "Average Data (Image)", _avgComponentDataPixel);

        _avgComponentDataPixelImg->setType(ImageData::Type::Stack);
        _avgComponentDataPixelImg->setNumberOfImages(_data.numDimensions);
        _avgComponentDataPixelImg->setImageSize(_imgSize);
        _avgComponentDataPixelImg->setNumberOfComponentsPerPixel(1);

        events().notifyDatasetDataChanged(_avgComponentDataPixelImg);
    }

    // Init embedding point meta data
    {
        std::vector<float> initialPointData(_data.numPoints, 0);
        _representSizeDataset = mv::data().createDataset<Points>("Points", "Represented Data Size", outputDataset);
        _representSizeDataset->setData(initialPointData.data(), _data.numPoints, 1);
        events().notifyDatasetDataChanged(_representSizeDataset);

        _notMergedNotesDataset = mv::data().createDataset<Points>("Points", "Not Merged Nodes", outputDataset);
        _notMergedNotesDataset->setData(initialPointData.data(), _data.numPoints, 1);
        events().notifyDatasetDataChanged(_notMergedNotesDataset);

        _randomWalkPointSim = mv::data().createDataset<Points>("Points", "Random Walk ProbDist", outputDataset);
        _randomWalkPointSim->setData(initialPointData.data(), _data.numPoints, 1);
        events().notifyDatasetDataChanged(_randomWalkPointSim);
        
        initialPointData.resize(_data.numPoints * _data.numDimensions, 0);
        _avgComponentDataSuper = mv::data().createDataset<Points>("Points", "Average Data (Superpixel)", outputDataset);
        _avgComponentDataSuper->setData(initialPointData.data(), _data.numPoints, _data.numDimensions);
        _avgComponentDataSuper->setDimensionNames(_inputData->getDimensionNames());
        events().notifyDatasetDataChanged(_avgComponentDataSuper);
    }

    // Connect selection mappings
    connect(&_inputData,                &Dataset<Points>::dataSelectionChanged,         this, &SPHPlugin::onSelectionInInputData);
    connect(&_output[0],                &Dataset<DatasetImpl>::dataSelectionChanged,    this, &SPHPlugin::onSelectionInEmbedding);
    connect(&_dataColoredByEmb,         &Dataset<Points>::dataSelectionChanged,         this, &SPHPlugin::onSelectionInImgColoredByEmb);
    connect(&_superpixelComponents,     &Dataset<Points>::dataSelectionChanged,         this, &SPHPlugin::onSelectionInSuperPixelComponents);
    connect(&_avgComponentDataPixel,    &Dataset<Points>::dataSelectionChanged,         this, &SPHPlugin::onSelectionInPixelAverages);

    connect(&_inputData, &Dataset<Points>::dataChanged, this, []() { 
        Log::warn("Input data changed. This well NOT be reflected in the computation or output of this plugin. If you want that to happen, implement it.");
        });

    /// Connect UI elements ///

    // Start computation
    connect(&_settingsAction.getHierarchySettingsAction().getStartAnalysisButton(), &TriggerAction::triggered, this, &SPHPlugin::computeHierarchy);

    // Go up and down the hierarchy for current entire view
    connect(&_settingsAction.getHierarchySettingsAction().getLevelDownUpActions(), &LevelDownUpActions::levelChanged, this, [this](int32_t newLevel) {
        if (!_isInit)
            return;

        // Reset interaction counter in UI
        _settingsAction.getTsneSettingsAction().getNumComputedIterationsAction().setValue(0);
        _computeEmbedding.setNumIterations(0);
        _computeEmbedding.setPublishExtendsIter(_settingsAction.getTsneSettingsAction().getIterationsPublishExtendAction().getValue());

        // if number of points is low, we want to default to the CPU gradient descent
        _settingsAction.getTsneSettingsAction().adjustToLowNumberOfPoints(_computeHierarchy.getProbDistOnLevel(newLevel).size());
        _settingsAction.getRefineTsneSettingsAction().adjustToLowNumberOfPoints(_computeHierarchy.getProbDistOnLevel(newLevel).size());

        // change level, update selection mapping and compute embedding
        updateEmbedding(newLevel);
        });

    /// Connect library computations///

    // update image hierarchy in core
    connect(&_computeHierarchy, &ComputeHierarchyWrapper::computedImageHierarchy, this, [this]() {
        Log::info("SPHPlugin:: Update hierarchy data in core");

        const auto& h = _computeHierarchy.getHierarchy();
        auto numLevels = h.getNumLevels();
        std::vector<float> componentIDs(static_cast<size_t>(numLevels * _data.numPoints), 0);

        // randomly shuffle component IDs for more distinct color mapping of spatial neighbors
        std::random_device rd;
        std::mt19937 g(rd());

        SPH_PARALLEL
        for (int64_t level = 0; level < static_cast<int64_t>(numLevels); level++) {

            std::vector<float> shuffledIDs(h.numComponentsOn(level));
            std::iota(shuffledIDs.begin(), shuffledIDs.end(), 0.f);
            std::shuffle(shuffledIDs.begin(), shuffledIDs.end(), g);

            for (int64_t point = 0; point < _data.numPoints; point++) {
                componentIDs[point * numLevels + level] = shuffledIDs[h.pixelComponentsOn(level)[point]];
            }
        }

        _superpixelComponents->setData(std::move(componentIDs), numLevels);
        events().notifyDatasetDataChanged(_superpixelComponents);

        _superpixelImage->setNumberOfImages(static_cast<uint32_t>(numLevels));
        events().notifyDatasetDataChanged(_superpixelImage);

        });

    connect(&_computeHierarchy, &ComputeHierarchyWrapper::computedKnnHierarchy, this, [this]() {
        const auto numLevels  = _computeHierarchy.getHierarchy().getNumLevels();
        const auto newLevel   = numLevels - 1;

        // Update UI
        _settingsAction.getHierarchySettingsAction().getLevelDownUpActions().setNumLevels(numLevels);
        _settingsAction.getTsneSettingsAction().getTsneComputeAction().setEnabled(true);
        _settingsAction.getHierarchySettingsAction().getStartAnalysisButton().setEnabled(true);

        _settingsAction.getRefineAction().setCurrentLevel(newLevel);

        // if number of points is low, we want to default to the CPU gradient descent
        _settingsAction.getTsneSettingsAction().adjustToLowNumberOfPoints(_computeHierarchy.getProbDistOnLevel(newLevel).size());
        _settingsAction.getRefineTsneSettingsAction().adjustToLowNumberOfPoints(_computeHierarchy.getProbDistOnLevel(newLevel).size());

        updateEmbedding(newLevel);
        _isInit = true;
        });

    // update embedding
    connect(&_computeEmbedding, &ComputeEmbeddingWrapper::embeddingUpdate, this, &SPHPlugin::setEmbeddingInManiVault);

    connect(&_computeEmbedding, &ComputeEmbeddingWrapper::finished, this, []() {
        Log::info("SPHPlugin::computeEmbedding: finished in {0} milliseconds", utils::timeSince(__tsneStartTime));
        });

    connect(&_computeEmbedding, &ComputeEmbeddingWrapper::workerStarted, this, [this]() {
        _isBusy = false;
        _settingsAction.getTsneSettingsAction().getTsneComputeAction().setStarted();
        });

    connect(&_computeEmbedding, &ComputeEmbeddingWrapper::workerEnded, this, [this]() {
        _settingsAction.getTsneSettingsAction().getTsneComputeAction().setFinished();
        });

    connect(&_settingsAction.getTsneSettingsAction().getTsneComputeAction().getStopComputationAction(), &TriggerAction::triggered, this, [this](bool checked) {
        _computeEmbedding.stopComputation();
        });

    connect(&_settingsAction.getTsneSettingsAction().getTsneComputeAction().getContinueComputationAction(), &TriggerAction::triggered, this, [this](bool checked) {
        _computeEmbedding.continueComputation(_settingsAction.getTsneSettingsAction().getNumNewIterationsAction().getValue());
        });

    connect(&_settingsAction.getTsneSettingsAction().getTsneComputeAction().getRestartComputationAction(), &TriggerAction::triggered, this, [this](bool checked) {
        _computeEmbedding.stopComputation();
        updateInitEmbedding();
        _computeEmbedding.restartComputation(_settingsAction.getTsneSettingsAction().getTsneParameters());
        });
}

void SPHPlugin::onSelectionInInputData()
{
    if (!_isInit)
        return;

    const bool allIsSync = areLocksInSync();
    const bool toBeHandled = isNotYetHandled(SelectionDatasets::INPUT);
    const bool doNothing = !allIsSync && !toBeHandled;

    if (doNothing)
        return;

    Log::trace("onSelectionInInputData");

    if (allIsSync) 
        markAsHandled(SelectionDatasets::GLOBAL);
        
    markAsHandled(SelectionDatasets::INPUT);

    // Selection in image maps to selection in hsne embedding 
    if (isNotYetHandled(SelectionDatasets::EMBEDDING))
        selectionMapping(_inputData, _mappingDataToLevel, getOutputDataset<Points>());

    if(isNotYetHandled(SelectionDatasets::RECOLOR_IMAGE))
        copySelection(_inputData, _dataColoredByEmb);

    if (isNotYetHandled(SelectionDatasets::SUPERPIXELS))
        copySelection(_inputData, _superpixelComponents);

    if (isNotYetHandled(SelectionDatasets::AVERAGES))
        copySelection(_inputData, _avgComponentDataPixel);
}

void SPHPlugin::onSelectionInEmbedding()
{
    if (!_isInit)
        return;

    const bool allIsSync = areLocksInSync();
    const bool toBeHandled = isNotYetHandled(SelectionDatasets::EMBEDDING);
    const bool doNothing = !allIsSync && !toBeHandled;

    if (doNothing)
        return;

    Log::trace("onSelectionInEmbedding");

    if (allIsSync)
        markAsHandled(SelectionDatasets::GLOBAL);

    markAsHandled(SelectionDatasets::EMBEDDING);

    // Selection in embedding maps to selection in image using _mappingLevelToData
    if (isNotYetHandled(SelectionDatasets::INPUT))
        selectionMapping(getOutputDataset<Points>(), _mappingLevelToData, _inputData);

    if (isNotYetHandled(SelectionDatasets::RECOLOR_IMAGE))
        copySelection(_inputData, _dataColoredByEmb);

    if (isNotYetHandled(SelectionDatasets::SUPERPIXELS))
        copySelection(_inputData, _superpixelComponents);

    if (isNotYetHandled(SelectionDatasets::AVERAGES))
        copySelection(_inputData, _avgComponentDataPixel);

    // update _randomWalkPointSim
    updateRandomWalkPointSimDataset();
}

void SPHPlugin::onSelectionInImgColoredByEmb()
{
    if (!_isInit)
        return;

    const bool allIsSync = areLocksInSync();
    const bool toBeHandled = isNotYetHandled(SelectionDatasets::RECOLOR_IMAGE);
    const bool doNothing = !allIsSync && !toBeHandled;

    if (doNothing)
        return;

    Log::trace("onSelectionInImgColoredByEmb");

    if (allIsSync)
        markAsHandled(SelectionDatasets::GLOBAL);

    markAsHandled(SelectionDatasets::RECOLOR_IMAGE);

    auto levelEmbedding = getOutputDataset<Points>();

    // Map from image to superpixel back to image
    // we want to select all pixels that belong to the superpixel of the selected image pixel
    if (isNotYetHandled(SelectionDatasets::EMBEDDING))
        selectionMapping(_dataColoredByEmb, _mappingDataToLevel, levelEmbedding);

    if (isNotYetHandled(SelectionDatasets::INPUT))
        selectionMapping(levelEmbedding, _mappingLevelToData, _inputData);

    if (isNotYetHandled(SelectionDatasets::SUPERPIXELS))
        copySelection(_inputData, _superpixelComponents);

    if (isNotYetHandled(SelectionDatasets::AVERAGES))
        copySelection(_inputData, _avgComponentDataPixel);
}

void SPHPlugin::onSelectionInSuperPixelComponents()
{
    if (!_isInit)
        return;

    const bool allIsSync = areLocksInSync();
    const bool toBeHandled = isNotYetHandled(SelectionDatasets::SUPERPIXELS);
    const bool doNothing = !allIsSync && !toBeHandled;

    if (doNothing)
        return;

    Log::trace("onSelectionInSuperPixelComponents");

    if (allIsSync)
        markAsHandled(SelectionDatasets::GLOBAL);

    markAsHandled(SelectionDatasets::SUPERPIXELS);

    auto levelEmbedding = getOutputDataset<Points>();

    // Map from image to superpixel back to image
    // we want to select all pixels that belong to the superpixel of the selected image pixel
    if (isNotYetHandled(SelectionDatasets::EMBEDDING))
        selectionMapping(_superpixelComponents, _mappingDataToLevel, levelEmbedding);

    if (isNotYetHandled(SelectionDatasets::INPUT))
        selectionMapping(levelEmbedding, _mappingLevelToData, _inputData);

    if (isNotYetHandled(SelectionDatasets::RECOLOR_IMAGE))
        copySelection(_inputData, _dataColoredByEmb);

    if (isNotYetHandled(SelectionDatasets::AVERAGES))
        copySelection(_inputData, _avgComponentDataPixel);
}

void SPHPlugin::onSelectionInPixelAverages()
{
    if (!_isInit)
        return;

    const bool allIsSync = areLocksInSync();
    const bool toBeHandled = isNotYetHandled(SelectionDatasets::AVERAGES);
    const bool doNothing = !allIsSync && !toBeHandled;

    if (doNothing)
        return;

    Log::trace("onSelectionInSuperPixelComponents");

    if (allIsSync)
        markAsHandled(SelectionDatasets::GLOBAL);

    markAsHandled(SelectionDatasets::AVERAGES);

    auto levelEmbedding = getOutputDataset<Points>();

    // Map from image to superpixel back to image
    // we want to select all pixels that belong to the superpixel of the selected image pixel
    if (isNotYetHandled(SelectionDatasets::EMBEDDING))
        selectionMapping(_avgComponentDataPixel, _mappingDataToLevel, levelEmbedding);

    if (isNotYetHandled(SelectionDatasets::INPUT))
        selectionMapping(levelEmbedding, _mappingLevelToData, _inputData);

    if (isNotYetHandled(SelectionDatasets::RECOLOR_IMAGE))
        copySelection(_inputData, _dataColoredByEmb);

    if (isNotYetHandled(SelectionDatasets::SUPERPIXELS))
        copySelection(_inputData, _superpixelComponents);
}

void SPHPlugin::updateRandomWalkPointSimDataset()
{
    const mv::Dataset<Points>& selectionEmbedding = _output[0]->getSelection<Points>();

    std::vector<float> randomWalkPointSims(_mappingLevelToData->size(), 0.f);

    if (selectionEmbedding->indices.size() >= 1)
    {
        const auto& randomWalkSimsLevel = _computeHierarchy.getProbDistOnLevel(_currentLevel);
        assert(randomWalkSimsLevel.size() == _mappingLevelToData->size());
        const auto& randomWalkSimsPoint = randomWalkSimsLevel[selectionEmbedding->indices[0]];

        for (auto it = randomWalkSimsPoint.cbegin(); it != randomWalkSimsPoint.cend(); it++) {
            randomWalkPointSims[it->first] += it->second;
        }
    }

    _randomWalkPointSim->setData(std::move(randomWalkPointSims), 1);
    events().notifyDatasetDataChanged(_randomWalkPointSim);
}

void SPHPlugin::updateAverageDatasets() {

    if (_mappingLevelToData == nullptr) {
        return;
    }

    std::vector<float> avgDataSuperpixels = computeAveragePerDimensionForSuperpixels(_data, *_mappingLevelToData);

    // Map (scatter) from superpixels to pixels
    std::vector<float> avgDataPixels = mapSuperpixelAverageToPixels(avgDataSuperpixels, _data.getNumPoints(), *_mappingLevelToData);

    _avgComponentDataSuper->setData(std::move(avgDataSuperpixels), _data.getNumDimensions());
    events().notifyDatasetDataChanged(_avgComponentDataSuper);

    _avgComponentDataPixel->setData(std::move(avgDataPixels), _data.getNumDimensions());
    events().notifyDatasetDataChanged(_avgComponentDataPixel);
}

void SPHPlugin::deselectAll()
{
    _inputData->getSelection<Points>()->indices.clear();
    events().notifyDatasetDataSelectionChanged(_inputData);
}

void SPHPlugin::setEmbeddingInManiVault(const std::vector<float>& emb)
{
    auto outputDataset = getOutputDataset<Points>();
    outputDataset->setData(emb, 2);
    events().notifyDatasetDataChanged(outputDataset);

    updateColorImage();

    _settingsAction.getTsneSettingsAction().getNumComputedIterationsAction().setValue(_computeEmbedding.getCurrentIterations());
}

void SPHPlugin::updateMappingsAndTransitionsReferences() 
{
    auto& hierarchy = _computeHierarchy.getHierarchy();

    // Update UI
    _settingsAction.getHierarchySettingsAction().setCurrentLevel(_currentLevel, hierarchy.getNumLevels());

    // Update selection mappings
    _mappingLevelToData = &(hierarchy.mapFromLevelToPixel[_currentLevel]);
    _mappingDataToLevel = &(hierarchy.mapFromPixelToLevel()[_currentLevel]);

    _currentTransitionMatrix = &_computeHierarchy.getProbDistOnLevel(_currentLevel);
    _numCurrentEmbPoints = _currentTransitionMatrix->size();
}

void SPHPlugin::updateEmbedding(int64_t level)
{
    if (_isBusy) {
        Log::trace("SPHPlugin::updateEmbedding: Is busy. Returning");
        return;
    }

    utils::ScopedTimer<std::chrono::milliseconds> updateScaleTimer("Level update (total)");

    _isBusy = true;
    _currentLevel = level;
    _settingsAction.getRefineAction().setCurrentLevel(_currentLevel);

    Log::info("SPHPlugin::updateEmbedding: to {0}", _currentLevel);

    // Make sure no points are selected before a level change
    Log::info("SPHPlugin::updateEmbedding: deselecting all");
    deselectAll();

    updateMappingsAndTransitionsReferences();
    Log::info("SPHPlugin::updateEmbedding: num points in embedding {0}", _numCurrentEmbPoints);

    updateAverageDatasets();

    // compute embedding (handles rescaling and reinitialization)
    computeEmbedding();
}

void SPHPlugin::computeHierarchy()
{
    Log::info("SPHPlugin::computeHierarchy");

    // Settings
    auto ihs            = getImageHierarchySettings();
    auto lss            = getLevelSimilaritiesSettings();
    auto nns            = getDataKnnSettings();
    auto rws            = getRandomWalkSettings();
    auto dataNorm       = getDataNormalizationScheme();
    auto filePath       = QFileInfo(getInputDataset<Images>()->getImageFilePaths().first()).dir().absolutePath().toStdString();
    auto fileName       = _inputData->getGuiName().toStdString();
    auto cacheActive    = _settingsAction.getHierarchySettingsAction().getCachingActiveAction().isChecked();

    std::filesystem::path cacheSettingsPath = std::filesystem::path(filePath) / "sph-cache" / "settings.cache";
    utils::saveCurrentSettings(cacheSettingsPath, nns, ihs, rws, lss);

    // auto-set nn based on data size
    if (nns.numNearestNeighbors <= 0) {
        float perplexity = _data.getNumPoints() / 100.f;
        perplexity = std::clamp(perplexity, 10.f, 100.f);
        nns.numNearestNeighbors = static_cast<int64_t>(perplexity) * 3;   // 3 is perplexity multiplier
    }

    // point itself will be one of the computed nn
    nns.numNearestNeighbors++;

    lss.ks = { static_cast<int64_t>(nns.numNearestNeighbors) };

    // apply normalization
    if (dataNorm != utils::Scaler::NONE) {
        utils::scale(_data, dataNorm);
    }

    // Start computation in another thread
    _computeHierarchy.startComputation(
        _data.getDataView(),
        _imgSize.height(), _imgSize.width(), 
        ihs, lss, rws, nns, 
        filePath, fileName, 
        cacheActive);

    // Update UI
    _settingsAction.getHierarchySettingsAction().getStartAnalysisButton().setEnabled(false);
}

void SPHPlugin::updateInitEmbedding()
{
    const QString initOption = _settingsAction.getTsneSettingsAction().getInitAction().getCurrentText();

    Log::info("SPHPlugin::updateInitEmbedding: Init embedding with {}", initOption.toStdString());

    auto initRandom = [this]() -> void {
        _computeEmbedding.initEmbedding(_currentLevel, _numCurrentEmbPoints);
        };

    auto getAverateDataOnLevel = [this]() -> std::vector<float> {
        assert(_avgComponentDataSuper->getNumPoints() == _numCurrentEmbPoints);
        std::vector<float> avgs(_avgComponentDataSuper->getNumPoints() * _avgComponentDataSuper->getNumDimensions(), 0.f);
        std::vector<uint32_t> enabledDimensionsIDs(_avgComponentDataSuper->getNumDimensions());
        std::iota(enabledDimensionsIDs.begin(), enabledDimensionsIDs.end(), 0);
        _avgComponentDataSuper->populateDataForDimensions<std::vector<float>, std::vector<uint32_t>>(avgs, enabledDimensionsIDs);
        return avgs;
        };

    if (initOption == "PCA")
    {
        size_t numPC = 2;
        std::vector<float> pca;
        bool success = false;

        if (_currentLevel == 0) {
            std::tie(pca, success) = utils::pca(_data.dataVec, _data.numDimensions, numPC);
        }
        else {
            std::vector<float> avgSuperpixelData = getAverateDataOnLevel();
            std::tie(pca, success) = utils::pca(avgSuperpixelData, _avgComponentDataSuper->getNumDimensions(), numPC);
        }

        if (success && numPC == 2) {
            utils::scaleEmbeddingToOne(pca);
            _computeEmbedding.initEmbedding(_currentLevel, _numCurrentEmbPoints, std::move(pca));
        }
        else {
            initRandom();
        }
    }
    else if (initOption == "Spectral")
    {
        std::vector<float> spectral;
        bool success = false;

        if (_currentLevel == 0) {
            std::tie(spectral, success) = utils::spectralEmbedding(_computeHierarchy.getImageHierarchyComp()->getDataKnnGraph());
        }
        else {
            Log::warn("SPHPlugin::updateInitEmbedding: Option Spectral not implemented for abstraction level. Computing PCA...");
            std::vector<float> avgSuperpixelData = getAverateDataOnLevel();
            size_t numPC = 2;
            std::tie(spectral, success) = utils::pca(avgSuperpixelData, _avgComponentDataSuper->getNumDimensions(), numPC);

            if (numPC != 2)
                success = false;
        }

        if (success)
        {
            utils::scaleEmbeddingToOne(spectral);
            _computeEmbedding.initEmbedding(_currentLevel, _numCurrentEmbPoints, std::move(spectral));
        }
        else {
            initRandom();
        }
    }
    else { // initOption == "RANDOM"
        initRandom();
    }

}

void SPHPlugin::computeEmbedding()
{
    Log::info("SPHPlugin::computeEmbedding: starting...");

    _computeEmbedding.stopComputation();

    __tsneStartTime = utils::now();

    const auto normScheme = getNormalizationScheme();
    _computeEmbedding.setNormScheme(normScheme);

    updateInitEmbedding();

    Log::info("SPHPlugin::computeEmbedding: Embedding extends (init): " + utils::computeExtends(_computeEmbedding.getInitEmbedding()).getMinMaxString());

    // update meta datasets
    {
        assert(_mappingLevelToData->size() == _numCurrentEmbPoints);

        // _representSizeDataset
        std::vector<float> representedDataPoints (_mappingLevelToData->size());

        SPH_PARALLEL
        for (int64_t i = 0; i < static_cast<int64_t>(_mappingLevelToData->size()); i++)
        {
            assert((*_mappingLevelToData)[i].size() > 0);
            float representedDataSize = static_cast<float>(std::log((*_mappingLevelToData)[i].size() + 1));
            representedDataPoints[i] = std::clamp(representedDataSize, 0.f, 10.f);
        }
        _representSizeDataset->setData(std::move(representedDataPoints), 1);
        events().notifyDatasetDataChanged(_representSizeDataset);

        // _notMergedNotesDataset
        std::vector<float> notMergedNodes(_mappingLevelToData->size(), 0.f);

        if (_currentLevel > 0)
        {
            // On data level, nodes cannot be merged
            const auto& notMergedNodesLevel = _computeHierarchy.getHierarchy().notMergedNodes[_currentLevel - 1];

            SPH_PARALLEL
            for (int64_t i = 0; i < static_cast<int64_t>(notMergedNodesLevel.size()); i++)
            {
                notMergedNodes[notMergedNodesLevel[i]] = 1.f;
            }
            _notMergedNotesDataset->setData(std::move(notMergedNodes), 1);
            events().notifyDatasetDataChanged(_notMergedNotesDataset);
        }
        
        // _randomWalkPointSim, only update on selection, init with default 0
        std::vector<float> randomWalkPointSims(_mappingLevelToData->size(), 0.f);
        _randomWalkPointSim->setData(std::move(randomWalkPointSims), 1);
        events().notifyDatasetDataChanged(_randomWalkPointSim);
    }

    _computeEmbedding.setPublishExtendsIter(_settingsAction.getTsneSettingsAction().getIterationsPublishExtendAction().getValue());

    if (normScheme == utils::NormalizationScheme::TSNE) {
        sph::TsneEmbeddingParameters& tSNEParams = _settingsAction.getTsneSettingsAction().getTsneParameters();

        if (!_settingsAction.getTsneSettingsAction().getExaggerationToggleAction().isChecked())
            tSNEParams.gradDescentParams._exaggeration_factor = _settingsAction.getTsneSettingsAction().getExaggerationFactorAction().getValue();
        else
            tSNEParams.gradDescentParams._exaggeration_factor = 4 + _numCurrentEmbPoints / 60000.0;

        tSNEParams.symmetricProbDist = true;    // LevelSimilarities computes symmetric probability distributions

        _computeEmbedding.startComputation(*_currentTransitionMatrix, tSNEParams);
    }
    else {
        sph::UmapEmbeddingParameters umapParams;
        umapParams.numEpochs = _settingsAction.getTsneSettingsAction().getNumDefaultUpdateIterationsAction().getValue();
        umapParams.singleStep = false;
        umapParams.presetEmbedding = true;
        _computeEmbedding.startComputation(*_currentTransitionMatrix, umapParams);
    }

}

void SPHPlugin::updateColorImage()
{
    extractEmbPositions(getOutputDataset<Points>(), *_mappingLevelToData, _imgSize, _dataColoredByEmb);
}

NearestNeighborsSettings SPHPlugin::getDataKnnSettings()
{
    NearestNeighborsSettings nns;

    nns.knnIndex                    = _settingsAction.getAdvancedSettingsAction().getDataIndexSetting();
    nns.knnMetric                   = _settingsAction.getHierarchySettingsAction().getDataMetricSetting();
    nns.numNearestNeighbors         = _settingsAction.getHierarchySettingsAction().getNumDataKnnSlider().getValue();
    nns.symmetricNeighbors          = _settingsAction.getAdvancedSettingsAction().getSymmetricKnnAction().isChecked();
    nns.neighborConnectComponents   = _settingsAction.getAdvancedSettingsAction().getConnectedKnnAction().isChecked();
    nns.computeConnectComponents    = true;
    nns.L2squared                   = false;

    return nns;
}

ImageHierarchySettings SPHPlugin::getImageHierarchySettings()
{
    ImageHierarchySettings ihs;

    ihs.componentSim        = _settingsAction.getHierarchySettingsAction().getComponentSimSetting();
    ihs.rwHandling          = _settingsAction.getHierarchySettingsAction().getRandomWalkHandlingSetting();
    ihs.neighborConnection  = _settingsAction.getHierarchySettingsAction().getNeighConnectionSetting();
    ihs.maxDist             = _settingsAction.getAdvancedSettingsAction().getMaxDistanceSetting();
    ihs.mergeMultiple       = _settingsAction.getAdvancedSettingsAction().getMergeWithAllAboveToggle().isChecked();
    ihs.usePercentile       = _settingsAction.getAdvancedSettingsAction().getPercentileOrValeAction().isChecked();
    ihs.minNumComp          = _settingsAction.getHierarchySettingsAction().getMinComponentsSlider().getValue();
    ihs.minReduction        = _settingsAction.getAdvancedSettingsAction().getMinReductionAction().getValue() * 100;
    ihs.normKnnDistances    = getNormalizationScheme();
    ihs.rwWeightMergeBySize = _settingsAction.getAdvancedSettingsAction().getWeightRWbySize().isChecked();
    ihs.rwReduction         = getRandomWalkReductionSetting();

    ihs.numGeodesicSamples = _settingsAction.getAdvancedSettingsAction().getNumGeodesicSamplesAction().getValue();
    if (ihs.numGeodesicSamples == 0)
        ihs.numGeodesicSamples = std::numeric_limits<size_t>::max();

    return ihs;
}

LevelSimilaritiesSettings SPHPlugin::getLevelSimilaritiesSettings()
{
    LevelSimilaritiesSettings lss;

    lss.componentSim = _settingsAction.getHierarchySettingsAction().getComponentSimSetting();
    lss.randomWalkPairSims = _settingsAction.getHierarchySettingsAction().getRandomWalkPairSims().isChecked();
    lss.ks = { };
    lss.exactKnn = _settingsAction.getAdvancedSettingsAction().getExactKnnAction().isChecked();
    lss.normalizeProbDist = getNormalizationScheme();
    lss.computeSymmetricProbDist = getNormalizationScheme();
    lss.weightTransitionBySize = false;
    lss.forceComputeDistances = false;

    return lss;
}

utils::Scaler SPHPlugin::getDataNormalizationScheme()
{
    utils::Scaler normData = utils::Scaler::NONE;
    int32_t normOption = _settingsAction.getAdvancedSettingsAction().getNormDataAction().getCurrentIndex();

    switch (normOption)
    {
    case 0: normData = utils::Scaler::NONE;     break;
    case 1: normData = utils::Scaler::STANDARD; break;
    case 2: normData = utils::Scaler::ROBUST;   break;
    }

    return normData;
}

utils::NormalizationScheme SPHPlugin::getNormalizationScheme()
{
    utils::NormalizationScheme normScheme = utils::NormalizationScheme::TSNE;
    int32_t normOption = _settingsAction.getAdvancedSettingsAction().getNormSchemeAction().getCurrentIndex();

    switch (normOption)
    {
    case 0: normScheme = utils::NormalizationScheme::TSNE;  break;
    case 1: normScheme = utils::NormalizationScheme::UMAP;  break;
    }

    return normScheme;
}

utils::RandomWalkReduction SPHPlugin::getRandomWalkReductionSetting()
{
    utils::RandomWalkReduction rwReduction = utils::RandomWalkReduction::PROPORTIONAL_COMPONENT_REDUCTION;
    int32_t rwOption = _settingsAction.getAdvancedSettingsAction().getRandomWalkReductionAction().getCurrentIndex();
    
    switch (rwOption)
    {
    case 0: rwReduction = utils::RandomWalkReduction::NONE;     break;
    case 1: rwReduction = utils::RandomWalkReduction::PROPORTIONAL_COMPONENT_REDUCTION; break;
    case 2: rwReduction = utils::RandomWalkReduction::PROPORTIONAL_HALF;   break;
    case 3: rwReduction = utils::RandomWalkReduction::PROPORTIONAL_DOUBLE;   break;
    case 4: rwReduction = utils::RandomWalkReduction::CONSTANT;   break;
    case 5: rwReduction = utils::RandomWalkReduction::CONSTANT_LOW;   break;
    case 6: rwReduction = utils::RandomWalkReduction::CONSTANT_HIGH;   break;
    }

    return rwReduction;
}

utils::RandomWalkSettings SPHPlugin::getRandomWalkSettings()
{
    auto rwSettings = utils::RandomWalkSettings();

    rwSettings.numRandomWalks   = _settingsAction.getHierarchySettingsAction().getNumRandomWalkSlider().getValue();
    rwSettings.singleWalkLength = _settingsAction.getHierarchySettingsAction().getLenRandomWalkSlider().getValue();
    rwSettings.pruneValue       = _settingsAction.getAdvancedSettingsAction().getPruneTransitionsValueAction().getValue();
    rwSettings.pruneSteps       = static_cast<uint64_t>(_settingsAction.getAdvancedSettingsAction().getPruneTransitionsStepsAction().getValue());

    int32_t weightingOption     = _settingsAction.getHierarchySettingsAction().getWeightingRandomWalkOption().getCurrentIndex();

    switch (weightingOption)
    {
    case 0: rwSettings.importanceWeighting = utils::ImportanceWeighting::CONSTANT;  break;
    case 1: rwSettings.importanceWeighting = utils::ImportanceWeighting::LINEAR;    break;
    case 2: rwSettings.importanceWeighting = utils::ImportanceWeighting::NORMAL;    break;
    case 3: rwSettings.importanceWeighting = utils::ImportanceWeighting::ONLYLAST;  break;
    case 4: rwSettings.importanceWeighting = utils::ImportanceWeighting::FIRST_VISIT;  break;
    }

    return rwSettings;
}

std::vector<uint32_t> SPHPlugin::getEnabledDimensions()
{
    Log::trace("InteractiveHsnePlugin:: enabledDimensions");

    std::vector<bool> enabledDimensions = _settingsAction.getDimensionSelectionAction().getPickerAction()->getEnabledDimensions();
    std::vector<uint32_t> enabledDimensionsIDs;
    for (uint32_t i = 0; i < _inputData->getNumDimensions(); i++)
        if (enabledDimensions[i])
            enabledDimensionsIDs.push_back(i);

    return enabledDimensionsIDs;
}


/// ////////////// ///
/// PLUGIN FACTORY ///
/// ////////////// ///

SPHPluginFactory::SPHPluginFactory() 
{
    setIconByName("grip-horizontal");
}

AnalysisPlugin* SPHPluginFactory::produce()
{
    // Return a new instance of this analysis plugin
    return new SPHPlugin(this);
}

PluginTriggerActions SPHPluginFactory::getPluginTriggerActions(const mv::Datasets& datasets) const
{
    PluginTriggerActions pluginTriggerActions;

    const auto getPluginInstance = [this](const Dataset<Points>& dataset) -> SPHPlugin* {
        return dynamic_cast<SPHPlugin*>(plugins().requestPlugin(getKind(), { dataset }));
    };

    if (PluginFactory::areAllDatasetsOfTheSameType(datasets, ImageType)) {
        if (datasets.count() >= 1) {
            auto pluginTriggerAction = new PluginTriggerAction(const_cast<SPHPluginFactory*>(this), this, "SPH", "Spatial Hierarchy", icon(), [this, getPluginInstance, datasets](PluginTriggerAction& pluginTriggerAction) -> void {
                for (const auto& dataset : datasets)
                    getPluginInstance(dataset);
                });

            pluginTriggerActions << pluginTriggerAction;
        }
    }

    return pluginTriggerActions;
}
