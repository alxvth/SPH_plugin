#include "Utils.h"

#include <sph/utils/Algorithms.hpp>
#include <sph/utils/CommonDefinitions.hpp>
#include <sph/utils/Embedding.hpp>

#include <CoreInterface.h>
#include <PointData/PointData.h>

#include <cassert>
#include <cstdint>
#include <type_traits>
#include <vector>

std::vector<uint32_t> mapPixelToSuperPixel(const mv::Dataset<Points>& selectionDataPixel, const std::vector<uint64_t>* selectionMapDataToLevel) {
    // to ensure only unique elements, sort and call std::unique 
    std::vector<uint32_t> selectionIndices;

    const auto& pixelSelectionIDs = selectionDataPixel->getSelection<Points>()->indices;
    selectionIndices.reserve(pixelSelectionIDs.size());

    uint32_t currentIndex = 0;

    // For all selected indices in the embedding, look up to which bottom level IDs they correspond
    for (const auto selectionIndex : pixelSelectionIDs)
    {
        currentIndex = (*selectionMapDataToLevel)[selectionIndex];
        if (currentIndex == std::numeric_limits<uint64_t>::max())    // marks empty
            continue;

        selectionIndices.push_back(static_cast<uint32_t>(currentIndex));
    }

    sph::utils::sortAndUnique(selectionIndices);

    return selectionIndices;
}

std::vector<uint32_t> mapSuperPixelToPixel(const mv::Dataset<Points>& selectionDataSuperpixel, const std::vector<std::vector<uint64_t>>* selectionMapLevelToData) {
    // to ensure only unique elements, sort and call std::unique 
    std::vector<uint32_t> selectionIndices;

    const auto& superpixelSelectionIDs = selectionDataSuperpixel->getSelection<Points>()->indices;
    selectionIndices.reserve(superpixelSelectionIDs.size()); // this is very conservative

    // For all selected indices in the embedding, look up to which bottom level IDs they correspond
    for (const auto selectionIndex : superpixelSelectionIDs)
    {
        const auto& selectionMapIndex = (*selectionMapLevelToData)[selectionIndex];
        if (selectionMapIndex.empty())
            continue;

        std::transform(selectionMapIndex.begin(),
            selectionMapIndex.end(),
            std::back_inserter(selectionIndices),
            [](const auto& val) { return static_cast<uint32_t>(val); }
        );
    }

    sph::utils::sortAndUnique(selectionIndices);

    return selectionIndices;
}

void copySelection(const mv::Dataset<Points>& selectionInput, mv::Dataset<Points>& selectionOutput)
{
    const auto& sel = selectionInput->getSelection<Points>()->indices;
    selectionOutput->getSelection<Points>()->indices.assign(sel.cbegin(), sel.cend());
    mv::events().notifyDatasetDataSelectionChanged(selectionOutput);
}

void selectionMapping(const mv::Dataset<Points>& selectionInputData, const std::vector<uint64_t>* selectionMap, mv::Dataset<Points> selectionOutputData)
{
    // if there is nothing to be mapped, don't do anything
    if (selectionMap->size() == 0)
        return;

    // Selection map is supposed to be of the same size as the selection input data
    assert(selectionMap->size() == selectionInputData->getNumPoints());

    std::vector<uint32_t> selectionIndices = mapPixelToSuperPixel(selectionInputData, selectionMap);

    selectionOutputData->getSelection<Points>()->indices = std::move(selectionIndices);
    mv::events().notifyDatasetDataSelectionChanged(selectionOutputData);
}

void selectionMapping(const mv::Dataset<Points>& selectionInputData, const std::vector<std::vector<uint64_t>>* selectionMap, mv::Dataset<Points> selectionOutputData)
{
    // if there is nothing to be mapped, don't do anything
    if (selectionMap->size() == 0)
        return;

    // Selection map is supposed to be of the same size as the selection input data
    assert(selectionMap->size() == selectionInputData->getNumPoints());

    std::vector<uint32_t> selectionIndices = mapSuperPixelToPixel(selectionInputData, selectionMap);

    selectionOutputData->getSelection<Points>()->indices = std::move(selectionIndices);
    mv::events().notifyDatasetDataSelectionChanged(selectionOutputData);
}

std::vector<uint32_t> expandPixelToSuperpixelSelection(const mv::Dataset<Points>& selectionInputData, const std::vector<uint64_t>* selectionMapDataToLevel, const std::vector<std::vector<uint64_t>>* selectionMaLevelToData) {
    std::vector<uint32_t> selectionIndices;

    // if there is nothing to be mapped, don't do anything
    if (selectionMapDataToLevel->size() == 0 || selectionMaLevelToData->size() == 0)
        return selectionIndices;

    // Selection map is supposed to be of the same size as the selection input data
    assert(selectionMapDataToLevel->size() == selectionInputData->getNumPoints());

    // First map from pixel to superpixels
    std::vector<uint32_t> selectionIndicesSuperPixel = mapPixelToSuperPixel(selectionInputData, selectionMapDataToLevel);

    // Then map back from superpixels to pixels
    {
        selectionIndices.reserve(selectionIndicesSuperPixel.size());

        // For all selected indices in the embedding, look up to which bottom level IDs they correspond
        for (const auto selectionIndex : selectionIndicesSuperPixel)
        {
            const auto& selectionMapIndex = (*selectionMaLevelToData)[selectionIndex];
            if (selectionMapIndex.empty())
                continue;

            std::transform(selectionMapIndex.begin(),
                selectionMapIndex.end(),
                std::back_inserter(selectionIndices),
                [](const auto& val) { return static_cast<uint32_t>(val); }
            );
        }

        sph::utils::sortAndUnique(selectionIndices);
    }

    return selectionIndices;
}

void extractEmbPositions(const mv::Dataset<Points>& embOnLevel, const sph::vvui64& mappingLevelToData, const QSize& imgSize, mv::Dataset<Points>& embPosOnLevel)
{
    const size_t numImagePoints = static_cast<size_t>(imgSize.height() * imgSize.width());
    const uint32_t numEmbPoints = embOnLevel->getNumPoints();
    const size_t numColorChannels = 2;

    std::vector<float> embData(static_cast<size_t>(numEmbPoints) * 2);
    std::vector<uint32_t> embDims{ 0, 1 };
    embOnLevel->populateDataForDimensions(embData, embDims);

    auto embExtends = sph::utils::computeExtends(embData);
    float xMin = embExtends.x_min();
    float yMin = embExtends.y_min();

    std::vector<float> embPos;
    embPos.resize(static_cast<size_t>(numImagePoints) * numColorChannels, 0.f);

    SPH_PARALLEL
    for (std::size_t i = 0; i < embPos.size(); ++i)
        embPos[i] = (i % 2 == 0) ? xMin : yMin;

    SPH_PARALLEL
    for (int64_t embID = 0; embID < static_cast<int64_t>(numEmbPoints); embID++)
    {
        // map the current color to all image points on which embID has the highest influence
        for (const auto& imgID : mappingLevelToData[embID])
        {
            embPos[numColorChannels * imgID] = embData[numColorChannels * embID];
            embPos[numColorChannels * imgID + 1u] = embData[numColorChannels * embID + 1u];
        }
    }

    embPosOnLevel->setData(std::move(embPos), numColorChannels);
    mv::events().notifyDatasetDataChanged(embPosOnLevel);
}

std::vector<float> computeAveragePerDimensionForSuperpixels(const sph::utils::DataView& data, const sph::vvui64& mappingLevelToData) {
    const size_t numSuperpixels = mappingLevelToData.size();
    const auto numDimensions    = data.getNumDimensions();

    std::vector<float> avgs(static_cast<size_t>(numSuperpixels) * numDimensions, 0.f);

    SPH_PARALLEL
    for (uint64_t superpixelID = 0; superpixelID < numSuperpixels; superpixelID++) {
        const auto& dataIDs = mappingLevelToData[superpixelID];
        std::vector<float> superpixelAvgs(numDimensions, 0.f);

        for (const auto dataID : dataIDs) {
            const auto dataValues = data.getValuesAt(dataID);

            assert(dataValues.size() == numDimensions);

            for (uint32_t dim = 0; dim < numDimensions; dim++) {
                superpixelAvgs[dim] += dataValues[dim];
            }
        }

        for (uint32_t dim = 0; dim < numDimensions; dim++) {
            avgs[superpixelID * numDimensions + dim] = superpixelAvgs[dim] / dataIDs.size();
        }
    }

    return avgs;
}

std::vector<float> mapSuperpixelAverageToPixels(const std::vector<float>& averagesSuperpixels, int64_t numDataPoints, const sph::vvui64& mappingLevelToData) {
    const size_t numSuperpixels = mappingLevelToData.size();
    const int64_t numDimensions = averagesSuperpixels.size() / numSuperpixels;

    std::vector<float> pixelAvgs(numDataPoints * numDimensions, 0.f);

    SPH_PARALLEL
    for (uint64_t superpixelID = 0; superpixelID < numSuperpixels; superpixelID++) {
        const auto& dataIDs = mappingLevelToData[superpixelID];

        for (const auto dataID : dataIDs) {
            for (int64_t dim = 0; dim < numDimensions; dim++) {
                pixelAvgs[dataID * numDimensions + dim] = averagesSuperpixels[superpixelID * numDimensions + dim];
            }
        }
    }

    return pixelAvgs;
}
