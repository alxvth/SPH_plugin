#pragma once

#include <sph/utils/CommonDefinitions.hpp>
#include <sph/utils/Data.hpp>

#include <PointData/PointData.h>

#include <cstdint>
#include <vector>

#include <QSize>

/// ///////// ///
/// SELECTION ///
/// ///////// ///

std::vector<uint32_t> mapPixelToSuperPixel(const mv::Dataset<Points>& selectionDataPixel, const std::vector<uint64_t>* selectionMapDataToLevel);
std::vector<uint32_t> mapSuperPixelToPixel(const mv::Dataset<Points>& selectionDataSuperpixel, const std::vector<std::vector<uint64_t>>* selectionMapLevelToData);

void copySelection(const mv::Dataset<Points>& selectionInput, mv::Dataset<Points>& selectionOutput);

void selectionMapping(const mv::Dataset<Points>& selectionInputData, const std::vector<uint64_t>* selectionMap, mv::Dataset<Points> selectionOutputData);

void selectionMapping(const mv::Dataset<Points>& selectionInputData, const std::vector<std::vector<uint64_t>>* selectionMap, mv::Dataset<Points> selectionOutputData);

// Returns a set of pixel that cover all superpixel which the input pixels are part of
std::vector<uint32_t> expandPixelToSuperpixelSelection(const mv::Dataset<Points>& selectionInputData, const std::vector<uint64_t>* selectionMapDataToLevel, const std::vector<std::vector<uint64_t>>* selectionMaLevelToData);

/// ///////// ///
/// EMBEDDING ///
/// ///////// ///

void extractEmbPositions(const mv::Dataset<Points>& embOnLevel, const sph::vvui64& mappingLevelToData, const QSize& imgSize, mv::Dataset<Points>& embPosOnLevel);

/// /////////////// ///
/// SUPERPIXEL DATA ///
/// /////////////// ///

std::vector<float> computeAveragePerDimensionForSuperpixels(const sph::utils::DataView& data, const sph::vvui64& mappingLevelToData);

static inline std::vector<float> computeAveragePerDimensionForSuperpixels(const sph::utils::Data& data, const sph::vvui64& mappingLevelToData) {
    return computeAveragePerDimensionForSuperpixels(data.getDataView(), mappingLevelToData);
}

std::vector<float> mapSuperpixelAverageToPixels(const std::vector<float>& averagesSuperpixels, int64_t numDataPoints, const sph::vvui64& mappingLevelToData);
