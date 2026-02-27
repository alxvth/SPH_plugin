#pragma once

#include <AnalysisPlugin.h>
#include <PointData/PointData.h>

#include "ComputeEmbeddingWrapper.h"
#include "ComputeHierarchyWrapper.h"
#include "SettingsAction.h"

#include <sph/utils/CommonDefinitions.hpp>
#include <sph/utils/Data.hpp>

#include <array>
#include <cstdint>
#include <ranges>
#include <vector>

#include <QSize>

using namespace mv::plugin;
using namespace mv::gui;

class Images;

/// ////////// ///
/// SPH PLUGIN ///
/// ////////// ///
class SPHPlugin : public AnalysisPlugin
{
    Q_OBJECT

private:
    enum class SelectionDatasets : size_t {
            GLOBAL,
            INPUT,
            EMBEDDING,
            RECOLOR_IMAGE,
            SUPERPIXELS,
            AVERAGES
        };

public:

    /**
     * Constructor
     * @param factory Pointer to the plugin factory
     */
    SPHPlugin(const PluginFactory* factory);

    /** Destructor */
    ~SPHPlugin() override = default;

    /* This function is called by the core after the analysis plugin has been created, sets up init data */
    void init() override;

    /** Sets current level and computes a new embedding */
    void updateEmbedding(int64_t level);

public:
    mv::Dataset<Points> getInputDataSet() { return _inputData; }
    sph::utils::DataView getInputData() { return _data.getDataView(); }
    QSize getImageSize() const { return _imgSize; }
    ComputeHierarchyWrapper* getComputeHierarchy() { return &_computeHierarchy; }
    const sph::vui64* getMappingDataToLevel(uint64_t level) const { return &(_computeHierarchy.getHierarchy().mapFromPixelToLevel()[level]); }
    const sph::vvui64* getMappingLevelToData(uint64_t level) const { return &(_computeHierarchy.getHierarchy().mapFromLevelToPixel[level]); }

private: // selection handling
    // TODO: integrate with RefinedSelectionMapping class

    /** A selection in the image is mapped to a the other data sets */
    void onSelectionInInputData();

    /** A selection in the embedding is mapped to a the other data sets */
    void onSelectionInEmbedding();

    void onSelectionInImgColoredByEmb();

    void onSelectionInSuperPixelComponents();

    void onSelectionInPixelAverages();

private:
    /** When a single point in the embedding is selected, update _randomWalkPointSim **/
    void updateRandomWalkPointSimDataset();

    /** imgColors are not resized, scatterColors are resized*/
    void updateColorImage();

    void updateAverageDatasets();
    
    void updateMappingsAndTransitionsReferences();
    
    void updateInitEmbedding();

    void computeHierarchy();

    void computeEmbedding();

    void deselectAll();

    void setEmbeddingInManiVault(const std::vector<float>& emb);

private: // convenience
    sph::NearestNeighborsSettings getDataKnnSettings();
    sph::ImageHierarchySettings getImageHierarchySettings();
    sph::LevelSimilaritiesSettings getLevelSimilaritiesSettings();
    sph::utils::Scaler getDataNormalizationScheme();
    sph::utils::NormalizationScheme getNormalizationScheme();
    sph::utils::RandomWalkReduction getRandomWalkReductionSetting();
    sph::utils::RandomWalkSettings getRandomWalkSettings();

    std::vector<uint32_t> getEnabledDimensions();

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

    SettingsAction              _settingsAction         = {this};           /** General settings, contains other settings classes */

    sph::utils::Data            _data                   = {};

    const sph::SparseMatHDI*    _currentTransitionMatrix = nullptr;

    int64_t                     _currentLevel           = 0;
    bool                        _isInit                 = false;
    bool                        _isBusy                 = false;
    bool                        _updateMetaDataset      = false;

    mv::Dataset<Images>         _superpixelImage        = { };              /** Image layout for _superpixelComponents */
    mv::Dataset<Points>         _superpixelComponents   = { };              /** superpixel component IDs (random numbers) */
    mv::Dataset<Points>         _inputData              = { };
    QSize                       _imgSize                = { };

    const sph::vvui64*          _mappingLevelToData     = nullptr;          /** Maps embedding indices to bottom indices (in image). The embedding indices refer to their position in the dataset vector */
    const sph::vui64*           _mappingDataToLevel     = nullptr;          /** Maps bottom indices (in image) to embedding indices. The embedding indices refer to their position in the dataset vector */

    std::array<uint64_t, 6>     _selectionCounters      = { 0, 0, 0, 0, 0, 0 }; /** Prevents endless selection loop */

    ComputeEmbeddingWrapper     _computeEmbedding       = { "t-SNE Analysis" };
    ComputeHierarchyWrapper     _computeHierarchy       = { "Image Hierarchy Wrapper" };
    size_t                      _numCurrentEmbPoints    = 0;

    sph::vf32                   _dataLevelEmbInit       = {};

    mv::Dataset<Points>         _dataColoredByEmb       = { };              /** Re-color image with level embedding scatter colors (data) */
    mv::Dataset<Images>         _imgColoredByEmb        = { };              /** Re-color image with level embedding scatter colors */
    
    mv::Dataset<Points>         _representSizeDataset   = { };              /** Dataset that stores how many data point are represented by a component on the current level */
    mv::Dataset<Points>         _notMergedNotesDataset  = { };              /** Dataset that stores how if a point was merged */
    mv::Dataset<Points>         _randomWalkPointSim     = { };              /** For a selected point, show the random walk similarities with this helper data set */

    mv::Dataset<Points>         _avgComponentDataSuper  = { };              /** Average data of superpixels */
    mv::Dataset<Points>         _avgComponentDataPixel  = { };              /** Average data of superpixels mapped to pixels (data values) */
    mv::Dataset<Images>         _avgComponentDataPixelImg = { };            /** Average data of superpixels mapped to pixels (image) */

};

/// ////////////// ///
/// PLUGIN FACTORY ///
/// ////////////// ///
class SPHPluginFactory : public AnalysisPluginFactory
{
    Q_INTERFACES(mv::plugin::AnalysisPluginFactory mv::plugin::PluginFactory)
    Q_OBJECT
    Q_PLUGIN_METADATA(IID   "manivault.studio.SPHPlugin"
                      FILE  "PluginInfo.json")

public:

    /** Default constructor */
    SPHPluginFactory();

    /** Destructor */
    ~SPHPluginFactory() override {}

    /** Creates an instance of the example analysis plugin */
    AnalysisPlugin* produce() override;

    /**
     * Get plugin trigger actions given \p datasets
     * @param datasets Vector of input datasets
     * @return Vector of plugin trigger actions
     */
    PluginTriggerActions getPluginTriggerActions(const mv::Datasets& datasets) const override;
};