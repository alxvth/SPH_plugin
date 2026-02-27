#include "ComputeEmbeddingWrapper.h"

#include <sph/utils/CommonDefinitions.hpp>
#include <sph/utils/Embedding.hpp>
#include <sph/utils/Logger.hpp>
#include <sph/utils/Progressbar.hpp>

#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

using namespace sph;

/// /////////// ///
/// EmbedWorker ///
/// /////////// ///

size_t EmbedWorker::_workerCount = 0;

void EmbedWorker::compute(uint32_t iterations, bool init)
{
    static_assert(_updateSteps > 0);

    if (iterations == 0)
        return;

    emit started();

    resetStop();

    auto checkPublishExtends = [this]() {
        if (_currentIteration >= _publishExtendsIter + _updateSteps)
            return;
        if (_currentIteration >= _publishExtendsIter)
            emit publishExtends(computeExtends());
    };

    utils::ProgressBar progress(iterations);

    int32_t remainingIter = iterations;

    if (init)
    {
        uint32_t firstUpdate = _updateSteps;
        if (iterations <= _updateSteps)
        {
            firstUpdate = iterations;
            remainingIter = 0;
        }
        else
            remainingIter = iterations - _updateSteps;

        initGradientDescent(firstUpdate);

        _currentIteration += firstUpdate;
        checkPublishExtends();
    }

    Log::info("ComputeEmbedding:: Gradient descent...");

    while(remainingIter > 0 && remainingIter >= _updateSteps)
    {
        if (_shouldStop)
            return;

        continueGradientDescent(_updateSteps);
        _currentIteration += _updateSteps;
        checkPublishExtends();
        remainingIter -= _updateSteps;

        progress.update(static_cast<uint64_t>(iterations) - remainingIter);
    }

    if (remainingIter != 0)
    {
        if (remainingIter < 0)
        {
            assert(iterations < _updateSteps);
            remainingIter = iterations;
        }

        if (_shouldStop)
            return;

        continueGradientDescent(remainingIter);
        _currentIteration += remainingIter;
        checkPublishExtends();
    }

    progress.finish();

    emit finished(computeExtends());
}

void EmbedWorker::continueComputation(uint32_t iterations)
{
    compute(iterations, /* init = */ false);
}

void EmbedWorker::stop()
{
    _shouldStop = true;

    if (_normScheme == utils::NormalizationScheme::TSNE) {
        _tsneComputation.stop();
    }
    else {
        _umapComputation.stop();
    }

    emit stopped();
}

void EmbedWorker::resetStop()
{
    _shouldStop = false;

    if (_normScheme == utils::NormalizationScheme::TSNE) {
        _tsneComputation.resetStop();
    }
    else {
        _umapComputation.resetStop();
    }
}

void EmbedWorker::initGradientDescent(uint32_t iterations)
{
    if (_normScheme == utils::NormalizationScheme::TSNE) {
        _tsneComputation.compute(iterations, false);
        emit embeddingUpdate(_tsneComputation.getEmbedding().getContainer());
    }
    else {
        _umapComputation.initProbabilityDistribution();
        _umapComputation.runGradientDescentForEpochs(iterations);
        emit embeddingUpdate(_umapComputation.getEmbedding());
    }
}

void EmbedWorker::continueGradientDescent(uint32_t iterations)
{
    if (_normScheme == utils::NormalizationScheme::TSNE) {
        _tsneComputation.continueGradientDescent(iterations, false);
        emit embeddingUpdate(_tsneComputation.getEmbedding().getContainer());
    }
    else {
        _umapComputation.runGradientDescentForEpochs(iterations);
        emit embeddingUpdate(_umapComputation.getEmbedding());
    }
}

sph::utils::EmbeddingExtends EmbedWorker::computeExtends() const
{
    if (_normScheme == utils::NormalizationScheme::TSNE) {
        auto& emb = _tsneComputation.getEmbedding().getContainer();
        return utils::computeExtends(emb);
    }
    else {
        auto& emb = _umapComputation.getEmbedding();
        return utils::computeExtends(emb);
    }
}

/// /////////////////////// ///
/// ComputeEmbeddingWrapper ///
/// /////////////////////// ///

ComputeEmbeddingWrapper::ComputeEmbeddingWrapper(const std::string& name)
{
    _analysisName = name;

    // Offscreen buffer must be created in the UI thread because it is a QWindow
    _offscreenBuffer->moveToThread(&_workerThread);
    _offscreenBuffer->getContext()->moveToThread(&_workerThread);
}

ComputeEmbeddingWrapper::~ComputeEmbeddingWrapper()
{
    _workerThread.quit();
    _workerThread.wait();
    _workerThread.deleteLater();
}

void ComputeEmbeddingWrapper::startComputation(const utils::Graph& knnGraph, const TsneEmbeddingParameters& params)
{
    _embedWorker->getTsneComp().setNeighborGraph(&knnGraph);

    compute(params);
}

void ComputeEmbeddingWrapper::startComputation(const SparseMatHDI& probDist, const TsneEmbeddingParameters& params)
{
    _embedWorker->getTsneComp().setProbabilityDistribution(&probDist);

    compute(params);
}

void ComputeEmbeddingWrapper::startComputation(const utils::Graph& knnGraph, const UmapEmbeddingParameters& params)
{
    _embedWorker->getUmapComp().setNeighborGraph(&knnGraph);

    compute(params);
}

void ComputeEmbeddingWrapper::startComputation(const SparseMatHDI& probDist, const UmapEmbeddingParameters& params)
{
    _embedWorker->getUmapComp().setNeighborMatrix(&probDist);

    compute(params);
}

void ComputeEmbeddingWrapper::resizeInitEmbedding(uint64_t numEmbPoints)
{
    _initEmbedding.clear();
    _initEmbedding.resize(numEmbPoints * 2, 0);
}

void ComputeEmbeddingWrapper::initEmbedding(const uint64_t newLevel, uint64_t numEmbPoints)
{
    setCurrentLevel(newLevel);
    resizeInitEmbedding(numEmbPoints);
    utils::randomEmbeddingInit(_initEmbedding, _initRadius, _initRadius);
}

void ComputeEmbeddingWrapper::initEmbedding(const uint64_t newLevel, uint64_t numEmbPoints, std::vector<float>&& embedding)
{
    assert(embedding.size() == numEmbPoints * 2);
    setCurrentLevel(newLevel);
    _initEmbedding = std::move(embedding);
}

void ComputeEmbeddingWrapper::updateInitEmbedding(const uint64_t newLevel, const uint64_t levelSize)
{
    assert(_currentLevel != std::numeric_limits<uint64_t>::max());  // call initEmbedding first

    setCurrentLevel(newLevel);
    _initEmbedding.resize(levelSize);
    utils::randomEmbeddingInit(_initEmbedding, _initRadius, _initRadius);
}

void ComputeEmbeddingWrapper::compute(const TsneEmbeddingParameters& params)
{
    _embedWorker->setNormScheme(utils::NormalizationScheme::TSNE);
    auto& tsneComputation = _embedWorker->getTsneComp();
    tsneComputation.setParams(params);
    tsneComputation.setInitialEmbedding(_initEmbedding);    // updates params.gradDescentParams._presetEmbedding, i.e. call after setParams()
    tsneComputation.setOffscreenBuffer(dynamic_cast<OffscreenBuffer*>(_offscreenBuffer.get()));

    if (!_workerThread.isRunning())
    {
        _embedWorker->setName(_analysisName);
        _embedWorker->moveToThread(&_workerThread);

        // To worker
        connect(this, &ComputeEmbeddingWrapper::startWorker, _embedWorker.get(), &EmbedWorker::compute);
        connect(this, &ComputeEmbeddingWrapper::continueWorker, _embedWorker.get(), &EmbedWorker::continueComputation);
        connect(this, &ComputeEmbeddingWrapper::stopWorker, _embedWorker.get(), &EmbedWorker::stop, Qt::DirectConnection);

        // From worker
        connect(_embedWorker.get(), &EmbedWorker::started, this, &ComputeEmbeddingWrapper::workerStarted);
        connect(_embedWorker.get(), &EmbedWorker::stopped, this, &ComputeEmbeddingWrapper::workerEnded);
        connect(_embedWorker.get(), &EmbedWorker::embeddingUpdate, this, &ComputeEmbeddingWrapper::embeddingUpdate);
        connect(_embedWorker.get(), &EmbedWorker::finished, this, [this](utils::EmbeddingExtends emdExtends) {
            _emdExtendsFinal = emdExtends;
            Log::info("ComputeEmbeddingWrapper::publishExtends: Embedding extends at iteration {0} are {1} ", _embedWorker->getCurrentIterations(), _emdExtendsFinal.getMinMaxString());
            emit finished();
            emit workerEnded();
            }); 
        connect(_embedWorker.get(), &EmbedWorker::publishExtends, this, [this](utils::EmbeddingExtends emdExtends) {
            _emdExtendsTarget = emdExtends;
            Log::info("ComputeEmbeddingWrapper::publishExtends: Embedding extends at iteration {0} are {1} ", _embedWorker->getCurrentIterations(), _emdExtendsTarget.getMinMaxString());
        });

        // Start thread
        _workerThread.start();
    }

    Log::info("ComputeEmbeddingWrapper::compute: start {0} t-SNE iterations", params.numIterations);

    // Update core with init embedding
    emit embeddingUpdate(_initEmbedding);

    // Start computation in thread
    emit startWorker(params.numIterations);
}

void ComputeEmbeddingWrapper::compute(const UmapEmbeddingParameters& params)
{
    _embedWorker->setNormScheme(utils::NormalizationScheme::UMAP);
    auto& umapComputation = _embedWorker->getUmapComp();
    umapComputation.setParams(params);
    umapComputation.setInitialEmbedding(_initEmbedding);    // updates params.gradDescentParams._presetEmbedding, i.e. call after setParams()

    if (!_workerThread.isRunning())
    {
        _embedWorker->setName(_analysisName);
        _embedWorker->moveToThread(&_workerThread);

        // To worker
        connect(this, &ComputeEmbeddingWrapper::startWorker, _embedWorker.get(), &EmbedWorker::compute);
        connect(this, &ComputeEmbeddingWrapper::continueWorker, _embedWorker.get(), &EmbedWorker::continueComputation);
        connect(this, &ComputeEmbeddingWrapper::stopWorker, _embedWorker.get(), &EmbedWorker::stop, Qt::DirectConnection);

        // From worker
        connect(_embedWorker.get(), &EmbedWorker::started, this, &ComputeEmbeddingWrapper::workerStarted);
        connect(_embedWorker.get(), &EmbedWorker::stopped, this, &ComputeEmbeddingWrapper::workerEnded);
        connect(_embedWorker.get(), &EmbedWorker::embeddingUpdate, this, &ComputeEmbeddingWrapper::embeddingUpdate);
        connect(_embedWorker.get(), &EmbedWorker::finished, this, [this](utils::EmbeddingExtends emdExtends) {
            _emdExtendsFinal = emdExtends;
            Log::info("ComputeEmbeddingWrapper::publishExtends: Embedding extends at iteration {0} are {1} ", _embedWorker->getCurrentIterations(), _emdExtendsFinal.getMinMaxString());
            emit finished();
            emit workerEnded();
            }); 
        connect(_embedWorker.get(), &EmbedWorker::publishExtends, this, [this](utils::EmbeddingExtends emdExtends) {
            _emdExtendsTarget = emdExtends;
            Log::info("ComputeEmbeddingWrapper::publishExtends: Embedding extends at iteration {0} are {1} ", _embedWorker->getCurrentIterations(), _emdExtendsTarget.getMinMaxString());
        });

        // Start thread
        _workerThread.start();
    }

    Log::info("ComputeEmbeddingWrapper::compute: start {0} UMAP iterations", params.numEpochs);

    // Update core with init embedding
    emit embeddingUpdate(_initEmbedding);

    // Start computation in thread
    emit startWorker(params.numEpochs);
}

void ComputeEmbeddingWrapper::continueComputation(uint32_t iterations)
{
    Log::info("ComputeEmbeddingWrapper::compute: continue {0} iterations", iterations);

    emit continueWorker(iterations);
}

void ComputeEmbeddingWrapper::restartComputation(const TsneEmbeddingParameters& params)
{
    Log::info("ComputeEmbeddingWrapper::restart: restart t-SNE with {0} iterations", params.numIterations);
    setNumIterations(0);
    compute(params);
}

void ComputeEmbeddingWrapper::restartComputation(const UmapEmbeddingParameters& params)
{
    Log::info("ComputeEmbeddingWrapper::restart: restart UMAP with {0} iterations", params.numEpochs);
    setNumIterations(0);
    compute(params);
}

void ComputeEmbeddingWrapper::stopComputation()
{
    emit stopWorker();
}

/// ///////////////// ///
/// OffscreenBufferQt ///
/// ///////////////// ///

OffscreenBufferQt::OffscreenBufferQt() : 
    OffscreenBuffer(),
    _context(new QOpenGLContext(this))
{
    setSurfaceType(QWindow::OpenGLSurface);
    create();
}

void OffscreenBufferQt::initialize()
{
    QOpenGLContext* globalContext = QOpenGLContext::globalShareContext();
    _context->setFormat(globalContext->format());

    if (!_context->create())
        qFatal("Cannot create requested OpenGL context.");

    bindContext();

#ifndef __APPLE__
    if (!gladLoadGL()) {
        qFatal("No OpenGL context is currently bound, therefore OpenGL function loading has failed.");
    }
#endif // Not __APPLE__

    _isInitialized = true;
}

void OffscreenBufferQt::bindContext()
{
    _context->makeCurrent(this);
}

void OffscreenBufferQt::releaseContext()
{
    _context->doneCurrent();
}

void OffscreenBufferQt::destroyContext()
{
    releaseContext();
    _context.clear();
    _isInitialized = false;
}
