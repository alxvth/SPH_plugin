#pragma once

#include "ComputeEmbeddingWrapper.h"

#include <sph/utils/CommonDefinitions.hpp>

#include <actions/DecimalAction.h>
#include <actions/GroupAction.h>
#include <actions/TriggerAction.h>

#include <Dataset.h>
#include <PointData/PointData.h>

class SPHPlugin;
class TsneSettingsAction;
class RefinedSelectionMapping;
class Images;

/// ///////////////////// ///
///      RefineAction     ///
/// ///////////////////// ///

class RefineAction : public mv::gui::GroupAction
{
    Q_OBJECT

public:
    RefineAction(QObject* parent);

public: // Getter
    int64_t getCurrentLevel() const { return _currentLevel; }

public: // Setter
    void setCurrentLevel(int64_t l) { _currentLevel = l; }
    void setSPHPlugin(SPHPlugin* sph) { _sphPlugin = sph; }
    void setParentEmbedding(mv::Dataset<Points> data) { _parentEmbedding = data; }
    void setTsneSettingsAction(TsneSettingsAction* tset) { _refineTsneSettingsAction = tset; }

public: // Action getters

    mv::gui::TriggerAction& getRefineAction() { return _refineAction; };
    mv::gui::DecimalAction& getExactRefinementAction() { return _exactRefinementAction; };

private slots:
    void refine();
    
private:
    using Datasets = std::vector<mv::Dataset<Points>>;
    using ImageDatasets = std::vector<mv::Dataset<Images>>;
    using RefinedScaleActions = std::vector<RefineAction*>;
    using TsneSettingsActions = std::vector<TsneSettingsAction*>;
    using RefinedSelectionMappings = std::vector<RefinedSelectionMapping*>;

private: // UI elements
    mv::gui::TriggerAction      _refineAction;                  /** Refine button */
    mv::gui::DecimalAction      _exactRefinementAction;         /** Refine button */

private:
    SPHPlugin*                  _sphPlugin = nullptr;
    int64_t                     _currentLevel = 0;

    sph::SparseMatHDI           _refinedTransitionMatrix = {};
    ComputeEmbeddingWrapper     _computeEmbedding = { "Refine Embedding" };
    TsneSettingsAction*         _refineTsneSettingsAction = nullptr;
    mv::Dataset<Points>         _parentEmbedding = {};                          /** Parent embedding dataset references */
    Datasets                    _refinedEmbeddings = {};                        /** Refine embedding dataset references */
    Datasets                    _refinedRecolorData = {};                       /** Refine embedding recolor data based on scatter layout */
    ImageDatasets               _refinedRecolorImages = {};                     /** Refine embedding recolor images based on scatter layout */
    Datasets                    _refinedRepresentedSizes = {};                  /** Refine embedding represented data size dataset references */
    Datasets                    _refinedTransitionEntriess = {};                /** Refine embedding non-zero transition entries */
    RefinedScaleActions         _refinedScaleActions = {};                      /** Refine embedding scale actions  */
    TsneSettingsActions         _refinedTsneSettingsActions = {};               /** Refine embedding tsne settings actions */
    RefinedSelectionMappings    _refinedRefinedSelectionMappings = {};          /** Refine embedding selection maps */

    Datasets                    _avgComponentDatasSuper = { };                  /** Average data of superpixels */
    Datasets                    _avgComponentDatasPixel = { };                  /** Average data of superpixels mapped to pixels (data values) */
    ImageDatasets               _avgComponentDatasPixelImg = { };               /** Average data of superpixels mapped to pixels (image) */
};
