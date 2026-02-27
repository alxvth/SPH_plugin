#include "ComputeHierarchyWrapper.h"

#include <sph/utils/Logger.hpp>
#include <sph/utils/PrintHelper.hpp>
#include <sph/utils/ShortestPath.hpp>

using namespace sph;

/// /////////////// ///
/// HierarchyWorker ///
/// /////////////// ///

size_t HierarchyWorker::_workerCount = 0;

void HierarchyWorker::init(const utils::DataView& data, int64_t rows, int64_t cols, const ImageHierarchySettings& ihs, const LevelSimilaritiesSettings& lss,
                           const utils::RandomWalkSettings& rws, const NearestNeighborsSettings& nns, const std::optional<CacheSettings>& cs)
{
    _computeHierarchy->init(data, rows, cols, ihs, lss, rws, nns, cs);
}

void HierarchyWorker::compute()
{
    utils::printSettings(_computeHierarchy->getImageHierarchySettings(), _computeHierarchy->getLevelSimilaritiesSettings(), _computeHierarchy->getNearestNeighborsSettings(), _computeHierarchy->getRandomWalkSettings());

    // 1. Create knn graph on data level
    _computeHierarchy->computeKnnGraph();

    if (_shouldStop)
        return;

    // 2. Build image hierarchy based on knn graph
    _computeHierarchy->computeImageHierarchy();

    // 3. Publish image hierarchy to ManiVault core
    emit computedImageHierarchy();

    if (_shouldStop)
        return;

    // 4. Compute knn on each hierarchy level
    _computeHierarchy->computeLevelSimilarities();

    // 5. Start computing embedding
    emit computedKnnHierarchy();

    emit finished();
}

void HierarchyWorker::stop()
{
    _shouldStop = true;
}

/// //////////////// ///
/// ComputeHierarchy ///
/// //////////////// ///

ComputeHierarchyWrapper::ComputeHierarchyWrapper(const std::string& name)
{
    _analysisName = name;
}

ComputeHierarchyWrapper::~ComputeHierarchyWrapper()
{
    _workerThread.quit();
    _workerThread.wait();
    _workerThread.deleteLater();
}

void ComputeHierarchyWrapper::startComputation(const utils::DataView& data, int64_t rows, int64_t cols, const ImageHierarchySettings& ihs, const LevelSimilaritiesSettings& lss,
                                        const utils::RandomWalkSettings& rwSettings, const NearestNeighborsSettings& nns, 
                                        const std::string& path, const std::string& fileName, bool cacheActive)
{
    utils::printSettings(ihs, lss, nns, rwSettings);

    _hierarchyWorker->setName(_analysisName);

    CacheSettings cs = { path, fileName, cacheActive };
    _hierarchyWorker->init(data, rows, cols, ihs, lss, rwSettings, nns, cs);

    if (!_workerThread.isRunning())
    {
        _hierarchyWorker->moveToThread(&_workerThread);

        // To worker
        connect(this, &ComputeHierarchyWrapper::startWorker, _hierarchyWorker.get(), &HierarchyWorker::compute);
        connect(this, &ComputeHierarchyWrapper::stopWorker, _hierarchyWorker.get(), &HierarchyWorker::stop);

        // From worker
        connect(_hierarchyWorker.get(), &HierarchyWorker::finished, this, &ComputeHierarchyWrapper::finished);
        connect(_hierarchyWorker.get(), &HierarchyWorker::computedImageHierarchy, this, &ComputeHierarchyWrapper::computedImageHierarchy);
        connect(_hierarchyWorker.get(), &HierarchyWorker::computedKnnHierarchy, this, &ComputeHierarchyWrapper::computedKnnHierarchy);

        // Start thread
        _workerThread.start();
    }

    // Start computation in thread
    emit startWorker();
}

void ComputeHierarchyWrapper::stopComputation()
{
    emit stopWorker();
}
