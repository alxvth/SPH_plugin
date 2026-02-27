#pragma once

#include <actions/WidgetAction.h>

#include <Dataset.h>
#include <PointData/PointData.h>

#include <sph/utils/CommonDefinitions.hpp>

#include <algorithm>
#include <array>
#include <cstdint>

#include <QSize>

/// ///////////////////////// ///
///  RefinedSelectionMapping ///
/// ///////////////////////// ///

class RefinedSelectionMapping : public mv::gui::WidgetAction
{
    Q_OBJECT

private:
    enum class SelectionDatasets : size_t {
        GLOBAL,
        INPUT,
        EMBEDDING,
        RECOLOR_IMAGE,
        AVERAGES,
    };

public:
    RefinedSelectionMapping(QObject* parent);

public: // Setter
    void setInputData(const mv::Dataset<Points>& input);
    void setEmbeddingData(const mv::Dataset<Points>& emb);
    void setImgColoredByEmb(const mv::Dataset<Points>& col);
    void setAvgComponentDataPixel(const mv::Dataset<Points>& avgs);
    
    void setMappingLevelToData(sph::vvui64&& map) { 
        _mappingLevelToData = std::move(map); 
    }
    void setMappingDataToLevel(sph::vui64&& map) { 
        _mappingDataToLevel = std::move(map); 
    }

public: // Getter
    const sph::vvui64& getMappingLevelToData() const { 
        return _mappingLevelToData; 
    }

    const sph::vui64& getMappingDataToLevel() const { 
        return _mappingDataToLevel; 
    }

    mv::Dataset<Points>& getImgColoredByEmb() { 
        return _dataColoredByLevelEmb; 
    }

    mv::Dataset<Points>& getAverageDataPixels() {
        return _avgComponentDataPixel;
    }

private:
    void onSelectionInInputData();
    void onSelectionInLevelEmbedding();
    void onSelectionInColoredByEmb();
    void onSelectionInPixelAverages();

private: // locking

    inline void markAsHandled(const SelectionDatasets& dataLock) {
        _selectionCounters[static_cast<size_t>(dataLock)]++;
    }

    inline bool isNotYetHandled(const SelectionDatasets& dataLock) const {
        return _selectionCounters[static_cast<size_t>(dataLock)] < _selectionCounters[static_cast<size_t>(SelectionDatasets::GLOBAL)];
    }

    inline bool areLocksInSync() const {
        return std::ranges::all_of(_selectionCounters, [&](uint64_t val) { return val == _selectionCounters[0]; });
    }

private:
    mv::Dataset<Points>     _levelEmbedding = { };
    mv::Dataset<Points>     _inputData = { };
    mv::Dataset<Points>     _dataColoredByLevelEmb = { };
    mv::Dataset<Points>     _avgComponentDataPixel = { };

    sph::vvui64             _mappingLevelToData = {};               /** Maps embedding indices to bottom indices (in image). The embedding indices refer to their position in the dataset vector */
    sph::vui64              _mappingDataToLevel = {};               /** Maps bottom indices (in image) to embedding indices. The embedding indices refer to their position in the dataset vector */

    std::array<uint64_t, 5> _selectionCounters = { 0, 0, 0, 0, 0 };      /** Prevents endless selection loop */
};
