#pragma once

#include <sph/ComputeHierarchy.hpp>

#include <sph/utils/CommonDefinitions.hpp>
#include <sph/utils/Data.hpp>
#include <sph/utils/Graph.hpp>
#include <sph/utils/Hierarchy.hpp>

#include <memory>
#include <optional>

#include <QThread>

/// /////////////// ///
/// HierarchyWorker ///
/// /////////////// ///

class HierarchyWorker : public QObject
{
    Q_OBJECT
public:
    HierarchyWorker() = default;
    ~HierarchyWorker() = default;

    // resets the compute classes and sets settings
    void init(const sph::utils::DataView& data, int64_t rows, int64_t cols, const sph::ImageHierarchySettings& ihs, const sph::LevelSimilaritiesSettings& lss,
             const sph::utils::RandomWalkSettings& rws, const sph::NearestNeighborsSettings& nns, const std::optional<sph::CacheSettings>& cs = std::nullopt);

public: // Setter
    void setName(const std::string& name) { _analysisParentName = name; }

public: // Getter
    std::string getName() const { return _analysisParentName; }
    const size_t getWorkerID() const { return _workerID; }

    const sph::NearestNeighbors* getKnnDataLevel() { return _computeHierarchy->getKnnDataLevel(); }
    const sph::ImageHierarchy* getImageHierarchy() { return _computeHierarchy->getImageHierarchy(); }
    const sph::LevelSimilarities* getLevelSimilarities() { return _computeHierarchy->getLevelSimilarities(); }

public slots:
    void compute();
    void stop();

signals:
    void computedImageHierarchy();
    void computedKnnHierarchy();
    void finished();

private:
    std::unique_ptr<sph::ComputeHierarchy>  _computeHierarchy = std::make_unique<sph::ComputeHierarchy>();

    // Flags and Utils
    static size_t                           _workerCount;
    size_t                                  _workerID = ++_workerCount;     // Debugging counter
    std::string                             _analysisParentName = "";       // Name for logging
    bool                                    _shouldStop = false;
};

/// /////////////////////// ///
/// ComputeHierarchyWrapper ///
/// /////////////////////// ///

class ComputeHierarchyWrapper : public QObject
{
    Q_OBJECT
public:
    ComputeHierarchyWrapper(const std::string& name = "");
    ~ComputeHierarchyWrapper();

    void startComputation(const sph::utils::DataView& data, int64_t rows, int64_t cols, const sph::ImageHierarchySettings& ihs, const sph::LevelSimilaritiesSettings& lss,
                          const sph::utils::RandomWalkSettings& rwSettings, const sph::NearestNeighborsSettings& nns,
                          const std::string& path, const std::string& fileName, bool cacheActive);
    void stopComputation();


public: // Getter
    const sph::utils::Hierarchy& getHierarchy() const { return _hierarchyWorker->getImageHierarchy()->getHierarchy(); }
    const sph::utils::GraphView getLevelSimilarities() const { return _hierarchyWorker->getLevelSimilarities()->getSimilaritiesGraphCurrent(); } // TODO: ensure consistent naming
    const sph::utils::GraphView getSimilaritiesOnLevel(int64_t level) const { return _hierarchyWorker->getLevelSimilarities()->getSimilaritiesGraph(level); } // TODO: ensure consistent naming
    const sph::SparseMatHDI& getProbDistOnLevel(int64_t level) const { return _hierarchyWorker->getLevelSimilarities()->getProbDist(level); }
    const sph::LevelSimilarities* getLevelSimComp() { return _hierarchyWorker->getLevelSimilarities(); }
    const sph::ImageHierarchy* getImageHierarchyComp() { return _hierarchyWorker->getImageHierarchy(); }

    bool threadIsRunning() const { return _workerThread.isRunning(); }

signals:
    // Local signals
    void startWorker();
    void stopWorker();

    // Outgoing signals
    void computedImageHierarchy();
    void computedKnnHierarchy();
    void finished();

private:
    QThread                             _workerThread       = QThread{};
    std::string                         _analysisName       = "";
    std::unique_ptr<HierarchyWorker>    _hierarchyWorker    = std::make_unique<HierarchyWorker>();
};
