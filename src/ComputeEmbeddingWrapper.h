#pragma once

#include <sph/EmbedTsne.hpp>
#include <sph/EmbedUmap.hpp>
#include <sph/utils/CommonDefinitions.hpp>
#include <sph/utils/Embedding.hpp>
#include <sph/utils/Graph.hpp>
#include <sph/utils/Settings.hpp>

#include <limits>
#include <memory>

#include <QOpenGLContext>
#include <QPointer>
#include <QThread>
#include <QWindow>

namespace sph {
    class LevelSimilarities;
}

class OffscreenBufferQt;

/// /////////// ///
/// EmbedWorker ///
/// /////////// ///

class EmbedWorker : public QObject
{
    Q_OBJECT
public:
    EmbedWorker() = default;
    ~EmbedWorker() = default;

public: // Setter
    void setName(const std::string& name) { _analysisParentName = name; }
    void setPublishExtendsIter(uint32_t publishExtendsIter) { _publishExtendsIter = publishExtendsIter; }
    void setNumIterations(uint32_t num) { _currentIteration = num; }
    void setNormScheme(sph::utils::NormalizationScheme scheme) { _normScheme = scheme; }

public: // Getter
    std::string getName() const { return _analysisParentName; }
    uint32_t getCurrentIterations() const { return _currentIteration; }
    uint32_t getPublishExtendsIter() const { return _publishExtendsIter; }
    sph::utils::NormalizationScheme getNormScheme() const { return _normScheme; }
    const size_t getWorkerID() const { return _workerID; }
    sph::TsneComputation& getTsneComp() { return _tsneComputation; }
    sph::UmapComputation& getUmapComp() { return _umapComputation; }

    inline constexpr uint32_t getUpdateStep() const { return _updateSteps; }

public slots:
    void compute(uint32_t iterations, bool init = true);
    void continueComputation(uint32_t iterations);
    void stop();
    void resetStop();

signals:
    void embeddingUpdate(const std::vector<float>& emb);
    void finished(sph::utils::EmbeddingExtends extends);
    void publishExtends(sph::utils::EmbeddingExtends extends);
    void started();
    void stopped();

private:
    void initGradientDescent(uint32_t iterations);
    void continueGradientDescent(uint32_t iterations);
    sph::utils::EmbeddingExtends computeExtends() const;

private:
    static size_t                       _workerCount;
    static constexpr uint32_t           _updateSteps = 10;

    sph::TsneComputation                _tsneComputation = {};
    sph::UmapComputation                _umapComputation = {};
    uint32_t                            _currentIteration = 0;          // Current gradient descent iteration
    uint32_t                            _publishExtendsIter = 0;        // Iteration at which to publish extends
    volatile bool                       _shouldStop = false;
    sph::utils::NormalizationScheme     _normScheme = sph::utils::NormalizationScheme::TSNE;

    size_t                              _workerID = ++_workerCount;     // Debugging counter
    std::string                         _analysisParentName = "";       // Name for logging

};

/// /////////////////////// ///
/// ComputeEmbeddingWrapper ///
/// /////////////////////// ///

class ComputeEmbeddingWrapper : public QObject
{
    Q_OBJECT
public:
    ComputeEmbeddingWrapper(const std::string& name = "");
    ~ComputeEmbeddingWrapper();

    void startComputation(const sph::SparseMatHDI& probDist, const sph::TsneEmbeddingParameters& params);
    void startComputation(const sph::utils::Graph& knnGraph, const sph::TsneEmbeddingParameters& params);
    
    void startComputation(const sph::SparseMatHDI& probDist, const sph::UmapEmbeddingParameters& params);
    void startComputation(const sph::utils::Graph& knnGraph, const sph::UmapEmbeddingParameters& params);
    
    void continueComputation(uint32_t iterations);
    void stopComputation();
    void restartComputation(const sph::TsneEmbeddingParameters& params);
    void restartComputation(const sph::UmapEmbeddingParameters& params);

    void initEmbedding(const uint64_t newLevel, uint64_t numEmbPoints, std::vector<float>&& embedding);    // for first time embedding
    void initEmbedding(const uint64_t newLevel, uint64_t numEmbPoints);                                    // for first time embedding
    void updateInitEmbedding(const uint64_t newLevel, const uint64_t levelSize);

public: // Setter

    void setCurrentLevel(uint64_t level) { _currentLevel = level; }
    void setNumIterations(uint32_t num) { _embedWorker->setNumIterations(num); }
    void setPublishExtendsIter(uint32_t num) { _embedWorker->setPublishExtendsIter(num); }
    void setNormScheme(sph::utils::NormalizationScheme scheme) { _embedWorker->setNormScheme(scheme); }

public: // Getter
    auto& getInitEmbedding() { return _initEmbedding; };
    const auto& getInitEmbedding() const { return _initEmbedding; };

    bool canContinue() const { return (_embedWorker == nullptr) ? false : _embedWorker->getCurrentIterations() >= 1; }
    uint32_t getCurrentIterations() const { return _embedWorker->getCurrentIterations(); }
    const std::vector<float>& getEmbedding() const { return _embedWorker->getTsneComp().getEmbedding().getContainer(); }
    bool threadIsRunning() const { return _workerThread.isRunning(); }

signals: // Outgoing signals
    void embeddingUpdate(const std::vector<float>& emb);
    void finished();
    void publishExtends(sph::utils::EmbeddingExtends extends);
    void workerStarted();
    void workerEnded();

signals: // Local signals
    void startWorker(uint32_t iterations, bool init = true);
    void continueWorker(uint32_t iterations);
    void stopWorker();

private:
    void compute(const sph::TsneEmbeddingParameters& params);
    void compute(const sph::UmapEmbeddingParameters& params);
    void resizeInitEmbedding(uint64_t numEmbPoints);

private:
    // Embedding Computation
    QThread                             _workerThread       = QThread{};
    std::string                         _analysisName       = "";
    std::unique_ptr<EmbedWorker>        _embedWorker        = std::make_unique<EmbedWorker>();
    std::unique_ptr<OffscreenBufferQt>  _offscreenBuffer    = std::make_unique<OffscreenBufferQt>();

    // Data
    std::vector<float>                  _embedding          = {};       /** current positions */
    std::vector<float>                  _initEmbedding      = {};       /** initialization positions */
    sph::utils::EmbeddingExtends        _emdExtendsTarget   = {};       /** Min and Max of each embedding dimension */
    sph::utils::EmbeddingExtends        _emdExtendsFinal    = {} ;      /** Min and Max of each embedding dimension */

    // Settings
    float                               _initRadius         = 1.f;
    uint64_t                            _currentLevel       = std::numeric_limits<uint64_t>::max();
};

/// ///////////////// ///
/// OffscreenBufferQt ///
/// ///////////////// ///

class OffscreenBufferQt : public QWindow, public sph::OffscreenBuffer
{
public:
    OffscreenBufferQt();

    QOpenGLContext* getContext() { return _context; }

    void initialize() override;
    void bindContext() override;
    void releaseContext() override;
    void destroyContext() override;

private:
    QPointer<QOpenGLContext> _context;
};
