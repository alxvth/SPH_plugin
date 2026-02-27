#include "RefineAction.h"

#include "RefinedSelectionMapping.h"
#include "SettingsTsneAction.h"
#include "SphPlugin.h"
#include "Utils.h"

#include <ImageData/ImageData.h>
#include <ImageData/Images.h>
#include <PointData/InfoAction.h>

#include <sph/utils/Algorithms.hpp>
#include <sph/utils/CommonDefinitions.hpp>
#include <sph/utils/Data.hpp>
#include <sph/utils/HDILibHelper.hpp>
#include <sph/utils/Math.hpp>

#include <ankerl/unordered_dense.h>
#include <hdi/data/map_mem_eff.h>

#include <algorithm>
#include <limits>

using namespace sph;

using hash = ankerl::unordered_dense::hash<int64_t>;
using hashmap = ankerl::unordered_dense::map<int64_t, int64_t, hash>;

RefineAction::RefineAction(QObject* parent) :
    GroupAction(parent, "RefineAction", true),
    _refineAction(this, "Refine"),
    _exactRefinementAction(this, "Exact refinement", 0.f, 1.f, 1.f, 2)
{
    setText("Refine");
    setObjectName("Refine");

    addAction(&_refineAction);
    addAction(&_exactRefinementAction);

    _refineAction.setToolTip("Refine selection: selection in data image (pixels)\nwill be converted to superpixels on level");
    _exactRefinementAction.setToolTip("Lower values than 1 will include points outside the selection to create better embeddings");

    _exactRefinementAction.setSingleStep(0.01f);

    connect(&_refineAction, &mv::gui::TriggerAction::triggered, this, &RefineAction::refine);

    const auto updateReadOnly = [this]() -> void {
        auto enable = !isReadOnly();

        _refineAction.setEnabled(enable);
        };

    connect(this, &GroupAction::readOnlyChanged, this, [this, updateReadOnly](const bool& readOnly) {
        updateReadOnly();
        });

}

void RefineAction::refine()
{
    if (_sphPlugin == nullptr)
    {
        qWarning() << "RefineAction::refine: _sphPlugin not set";
        return;
    }

    if (_currentLevel == 0)
    {
        qWarning() << "RefineAction::refine: _currentLevel is data, doing nothing";
        return;
    }

    mv::Dataset<Points> inputDataset        = _sphPlugin->getInputDataSet();
    const sph::utils::DataView inputData    = _sphPlugin->getInputData();

    if (inputDataset->getSelectionIndices().size() == 0) {
        qWarning() << "RefineAction::refine: selected data size is 0, doing nothing";
        return;
    }

    const auto refinedLevel = _currentLevel - 1;
    const sph::vui64* mappingDataToRefinedLevel = _sphPlugin->getMappingDataToLevel(refinedLevel);
    const sph::vvui64* mappingRefinedLevelToData = _sphPlugin->getMappingLevelToData(refinedLevel);

    if (mappingDataToRefinedLevel == nullptr || mappingRefinedLevelToData == nullptr) {
        qWarning() << "RefineAction::refine: mappingDataToRefinedLevel or mappingRefinedLevelToData is not defined, doing nothing";
        return;
    }

    const auto expandedSelectionInData = expandPixelToSuperpixelSelection(inputDataset, mappingDataToRefinedLevel, mappingRefinedLevelToData);

    // get probDist for selection on refined level
    const auto& probDistOnRefinedLevel = _sphPlugin->getComputeHierarchy()->getProbDistOnLevel(refinedLevel);

    std::vector<uint64_t> newEmbIdsInRefinedLevelEmb;
    const bool exactRefinement = sph::utils::isBasicallyEqual(_exactRefinementAction.getValue(), 1.f, 0.001);

    std::vector<uint64_t> selectionSuperpixelsInRefinedLevel;
    selectionSuperpixelsInRefinedLevel.reserve(expandedSelectionInData.size());

    for (const auto selectionIdData : expandedSelectionInData)
        selectionSuperpixelsInRefinedLevel.push_back((*mappingDataToRefinedLevel)[selectionIdData]);

    selectionSuperpixelsInRefinedLevel.shrink_to_fit();

    sph::utils::sortAndUnique(selectionSuperpixelsInRefinedLevel);

    if (exactRefinement)
        sph::utils::extractSubGraph(probDistOnRefinedLevel, selectionSuperpixelsInRefinedLevel, _refinedTransitionMatrix, newEmbIdsInRefinedLevelEmb);
    else
        sph::utils::extractSubGraph(probDistOnRefinedLevel, selectionSuperpixelsInRefinedLevel, _refinedTransitionMatrix, newEmbIdsInRefinedLevelEmb, _exactRefinementAction.getValue()); // also extract connected vertices

    const size_t numNewEmbPoints = newEmbIdsInRefinedLevelEmb.size();

    hashmap currentToRefinedIDs;
    currentToRefinedIDs.reserve(numNewEmbPoints);

    for (size_t idRefinedEmb = 0; idRefinedEmb < numNewEmbPoints; idRefinedEmb++)
        currentToRefinedIDs[newEmbIdsInRefinedLevelEmb[idRefinedEmb]] = idRefinedEmb;

    qDebug() << "Selection in data: " << expandedSelectionInData.size();
    qDebug() << "Parent embedding selection: " << _parentEmbedding->getSelectionIndices().size();
    qDebug() << "Refined embedding size: " << numNewEmbPoints;

    // add new embedding data set
    auto& refinedEmbedding = _refinedEmbeddings.emplace_back(mv::data().createDataset<Points>("Points", QString("Refined (level %1)").arg(refinedLevel), _parentEmbedding));

    // helper used for meta data and potentially embedding init
    std::vector<float> avgDataSuperpixels = computeAveragePerDimensionForSuperpixels(inputData, *mappingRefinedLevelToData);

    // add selection maps between refined embedding and data and update meta data sets
    {
        const size_t numImagePoints = static_cast<size_t>(inputDataset->getNumPoints());
        sph::vvui64 mapLevelToData(numNewEmbPoints);
        sph::vui64 mapDataToLevel(numImagePoints, std::numeric_limits<uint64_t>::max());

        for (size_t idRefinedEmb = 0; idRefinedEmb < numNewEmbPoints; idRefinedEmb++)
            mapLevelToData[idRefinedEmb] = (*mappingRefinedLevelToData)[newEmbIdsInRefinedLevelEmb[idRefinedEmb]];

        if (exactRefinement)
        {
            // no extra mapping required
            for (const auto selectionIdData : expandedSelectionInData)
                mapDataToLevel[selectionIdData] = currentToRefinedIDs[(*mappingDataToRefinedLevel)[selectionIdData]];
        }
        else
        {
            // add point outside selection to mapping as well
            for (const auto selectionIdInRefinedLevel : newEmbIdsInRefinedLevelEmb)
            {
                const sph::vui64& selectionIDsInData = (*mappingRefinedLevelToData)[selectionIdInRefinedLevel];
                for (const auto selectionIdData : selectionIDsInData)
                    mapDataToLevel[selectionIdData] = currentToRefinedIDs[(*mappingDataToRefinedLevel)[selectionIdData]];
            }
        }

        constexpr size_t numInitialDataDimensions = 2;

        {
            std::vector<float> initialPointData(numNewEmbPoints * numInitialDataDimensions, 0);
            refinedEmbedding->setData(std::move(initialPointData), numInitialDataDimensions);
            mv::events().notifyDatasetDataChanged(refinedEmbedding);
        }

        // populate data sets: image recolored by embedding layout 
        // -> reuse embedding position and recolor in image viewer with same color map as in Scatterplot
        auto& imgColoredByEmb = _refinedRecolorData.emplace_back(mv::data().createDataset<Points>("Points", "Scatter colors", refinedEmbedding));
        
        {
            std::vector<float> initialData(numInitialDataDimensions * numImagePoints, 0.f);
            imgColoredByEmb->setData(std::move(initialData), numInitialDataDimensions);
            events().notifyDatasetDataChanged(imgColoredByEmb);
        }

        auto& refinedRecolorImage = _refinedRecolorImages.emplace_back(mv::data().createDataset<Images>("Images", "Scatter colors", imgColoredByEmb));

        refinedRecolorImage->setType(ImageData::Type::Stack);
        refinedRecolorImage->setNumberOfImages(numInitialDataDimensions);
        refinedRecolorImage->setImageSize(_sphPlugin->getImageSize());
        refinedRecolorImage->setNumberOfComponentsPerPixel(1);

        std::vector<std::uint8_t> imageMask(numImagePoints, 255);

        SPH_PARALLEL
        for (size_t imageID = 0; imageID < numImagePoints; imageID++) {
            if (mapDataToLevel[imageID] == std::numeric_limits<uint64_t>::max())
                imageMask[imageID] = 0;
        }

        refinedRecolorImage->setMaskData(imageMask);

        events().notifyDatasetDataChanged(refinedRecolorImage);

        // populate data sets: resized embedding by represented data points
        auto& refinedRepresentedSizeData = _refinedRepresentedSizes.emplace_back(mv::data().createDataset<Points>("Points", "Represented Data Size", refinedEmbedding));

        {
            std::vector<float> representedDataPoints(numNewEmbPoints);

            SPH_PARALLEL
            for (int64_t i = 0; i < static_cast<int64_t>(numNewEmbPoints); i++)
            {
                assert(mapLevelToData[i].size() > 0);
                const float representedDataSize = std::log(static_cast<float>(mapLevelToData[i].size() + 1));
                representedDataPoints[i] = std::clamp(representedDataSize, 0.f, 10.f);
            }

            refinedRepresentedSizeData->setData(std::move(representedDataPoints), 1);
            events().notifyDatasetDataChanged(refinedRepresentedSizeData);
        }

        // populate data set: non-zero redined transition matrix entries
        auto& refinedTransitionEntries = _refinedTransitionEntriess.emplace_back(mv::data().createDataset<Points>("Points", "Transition Neighbors", refinedEmbedding));

        {
            std::vector<float> transitionEntries(numNewEmbPoints);

            assert(_refinedTransitionMatrix.size() == numNewEmbPoints);

            SPH_PARALLEL
            for (int64_t i = 0; i < static_cast<int64_t>(numNewEmbPoints); i++)
            {
                const float nonzeroEntires = std::log(static_cast<float>(_refinedTransitionMatrix[i].size() + 1));
                transitionEntries[i] = std::clamp(nonzeroEntires, 0.f, 10.f);
            }

            refinedTransitionEntries->setData(std::move(transitionEntries), 1);
            events().notifyDatasetDataChanged(refinedTransitionEntries);
        }

        // populate data sets: average images
        auto& avgComponentDataSuper = _avgComponentDatasSuper.emplace_back(mv::data().createDataset<Points>("Points", "Average Data (Superpixel)", refinedEmbedding));
        auto& avgComponentDataPixel = _avgComponentDatasPixel.emplace_back(mv::data().createDataset<Points>("Points", "Average Data (Pixel)", refinedEmbedding));
        auto& avgComponentDataPixelImg = _avgComponentDatasPixelImg.emplace_back(mv::data().createDataset<Images>("Images", "Average Data (Image)", avgComponentDataPixel));

        {
            // Map (scatter) from superpixels to pixels
            std::vector<float> avgDataPixels = mapSuperpixelAverageToPixels(avgDataSuperpixels, inputData.getNumPoints(), *mappingRefinedLevelToData);

            std::vector<float> initialAvgData(inputData.getNumPoints() * inputData.getNumDimensions(), 0);
            avgComponentDataSuper->setData(avgDataSuperpixels, inputData.getNumDimensions());
            avgComponentDataSuper->setDimensionNames(inputDataset->getDimensionNames());
            events().notifyDatasetDataChanged(avgComponentDataSuper);

            avgComponentDataPixel->setData(std::move(avgDataPixels), inputData.getNumDimensions());
            avgComponentDataPixel->setDimensionNames(inputDataset->getDimensionNames());
            events().notifyDatasetDataChanged(avgComponentDataPixel);

            avgComponentDataPixelImg->setType(ImageData::Type::Stack);
            avgComponentDataPixelImg->setNumberOfImages(inputData.getNumDimensions());
            avgComponentDataPixelImg->setImageSize(_sphPlugin->getImageSize());
            avgComponentDataPixelImg->setNumberOfComponentsPerPixel(1);

            avgComponentDataPixelImg->setMaskData(imageMask);

            events().notifyDatasetDataChanged(avgComponentDataPixelImg);
        }

        // Add selection mappings
        RefinedSelectionMapping* refineMappingAction = _refinedRefinedSelectionMappings.emplace_back(new RefinedSelectionMapping(this));

        refineMappingAction->setInputData(_sphPlugin->getInputDataSet());
        refineMappingAction->setEmbeddingData(refinedEmbedding);
        refineMappingAction->setImgColoredByEmb(imgColoredByEmb);
        refineMappingAction->setAvgComponentDataPixel(avgComponentDataPixel);

        refineMappingAction->setMappingLevelToData(std::move(mapLevelToData));
        refineMappingAction->setMappingDataToLevel(std::move(mapDataToLevel));

        refinedEmbedding->addAction(*refineMappingAction);
    }

    // add refine action and TsneSettingsAction if refined level > data level
    if(refinedLevel > 0)
    {
        RefineAction* refineAction = _refinedScaleActions.emplace_back(new RefineAction(this));
        TsneSettingsAction* tsneSettingsAction = _refinedTsneSettingsActions.emplace_back(new TsneSettingsAction(this, "Refine t-SNE"));

        refineAction->setSPHPlugin(_sphPlugin);
        refineAction->setTsneSettingsAction(tsneSettingsAction);
        refineAction->setParentEmbedding(refinedEmbedding);
        refineAction->setCurrentLevel(refinedLevel);

        tsneSettingsAction->setExpanded(true);
        tsneSettingsAction->adjustToLowNumberOfPoints(numNewEmbPoints);
        tsneSettingsAction->getTsneComputeAction().setEnabled(false);

        refinedEmbedding->addAction(*refineAction);
        refinedEmbedding->addAction(*tsneSettingsAction);
        refinedEmbedding->_infoAction->collapse();
    }

    // compute t-sne

    auto& tSNEParams = _refineTsneSettingsAction->getTsneParameters();
    tSNEParams.symmetricProbDist = true;

    if (numNewEmbPoints < 1000) {
        tSNEParams.gradientDescentType = GradientDescentType::CPU;
        qDebug() << "Refined embedding: adjust gradient descent to CPU for small number of points";
    }
    else {
        tSNEParams.gradientDescentType = GradientDescentType::GPUcompute;
    }

    if (_refineTsneSettingsAction->getInitAction().getCurrentText() == "PCA") {
        const auto numDims = inputData.getNumDimensions();
        std::vector<float> avgDataRefinedSuperpixels(selectionSuperpixelsInRefinedLevel.size() * numDims);

        SPH_PARALLEL
        for (size_t i = 0; i < selectionSuperpixelsInRefinedLevel.size(); i++) {
            auto copy_from_start = avgDataSuperpixels.begin() + selectionSuperpixelsInRefinedLevel[i] * numDims;
            auto copy_from_end = copy_from_start + numDims;

            std::ranges::copy(
                copy_from_start,
                copy_from_end,
                avgDataRefinedSuperpixels.begin() + i * numDims);
        }

        size_t numPC = 2;
        std::vector<float> pca;
        bool success = false;

        std::tie(pca, success) = sph::utils::pca(avgDataRefinedSuperpixels, numDims, numPC);
        
        if (success) {
            _computeEmbedding.initEmbedding(refinedLevel, numNewEmbPoints, std::move(pca));
            qDebug() << "Refined embedding initialized with PCA";
        }
        else {
            _computeEmbedding.initEmbedding(refinedLevel, numNewEmbPoints);
        }
    }
    else {
        if (_refineTsneSettingsAction->getInitAction().getCurrentText() != "RANDOM") {
            qDebug() << "Not implemented: " << _refineTsneSettingsAction->getInitAction().getCurrentText();
        }
        _computeEmbedding.initEmbedding(refinedLevel, numNewEmbPoints);
    }

    disconnect(&_computeEmbedding, nullptr, this, nullptr);

    // Update embedding points when the TSNE analysis produces new data
    connect(&_computeEmbedding, &ComputeEmbeddingWrapper::embeddingUpdate, this, [this](const sph::vf32& emb) {

        auto& refineEmbedding = _refinedEmbeddings.back();
        refineEmbedding->setData(emb.data(), emb.size() / 2, 2);
        mv::events().notifyDatasetDataChanged(refineEmbedding);

        auto refinedRefinedSelectionMapping = _refinedRefinedSelectionMappings.back();
        extractEmbPositions(refineEmbedding, refinedRefinedSelectionMapping->getMappingLevelToData(), _sphPlugin->getImageSize(), refinedRefinedSelectionMapping->getImgColoredByEmb());
        });

    _computeEmbedding.setNumIterations(0);
    _computeEmbedding.startComputation(_refinedTransitionMatrix, tSNEParams);
}
