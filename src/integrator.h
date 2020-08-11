#pragma once

#include "path.h"
#include "commondef.h"
#include "mutation.h"

struct PathFuncLib;

class Integrator {
public:
    std::string output_name = "";
    bool verbose = true;
    bool deterministic = false;
protected:
    const Scene * scene;
    const std::shared_ptr<const PathFuncLib> pathFuncLib;
public:
    Integrator(const Scene *scene, const std::shared_ptr<const PathFuncLib> pathFuncLib) : scene(scene), pathFuncLib(pathFuncLib){
    }
    virtual void Render() = 0;
    virtual Float PrePass(const MLTState &mltState, const int64_t numInitSamples, const int numChains, MarkovStates &initStates, std::shared_ptr<PiecewiseConstant1D> &lengthDist) = 0;
};

/**
 *  We implement a hybrid algorithm that combines Primary Sample Space MLT [Kelemen et al. 2002]
 *  and Multiplxed MLT (MMLT) [Hachisuka et al. 2014].  Specifically, the state of our Markov
 *  chain only represents one of the N^2 pairs connection as in MMLT.  During the "large
 *  step" mutations, instead of choosing the camera and light subpath lengths a priori as in
 *  MMLT, we sample all pairs of connections, and probabilistically pick one based on their
 *  contributions (similar to Multiple-try Metropolis).  During the "small step" mutations,
 *  we fix the camera and light subpath lengths of the state.
 */
inline void DirectLighting(const Scene *scene, SampleBuffer &buffer) {
    if (scene->options->minDepth > 2 || scene->options->maxDepth < 1) {
        return;
    }

    std::cout << "\nComputing direct lighting..." << std::endl;
    const Camera *camera = scene->camera.get();
    const int pixelHeight = GetPixelHeight(camera);
    const int pixelWidth = GetPixelWidth(camera);
    const int tileSize = 16;
    const int nXTiles = (pixelWidth + tileSize - 1) / tileSize;
    const int nYTiles = (pixelHeight + tileSize - 1) / tileSize;
    ProgressReporter reporter(nXTiles * nYTiles);

    Timer timer;
    Tick(timer);
    ParallelFor([&](const Vector2i tile) {
        const int seed = tile[1] * nXTiles + tile[0] + scene->options->seedOffset;
        RNG rng(seed);
        const int x0 = tile[0] * tileSize;
        const int x1 = std::min(x0 + tileSize, pixelWidth);
        const int y0 = tile[1] * tileSize;
        const int y1 = std::min(y0 + tileSize, pixelHeight);
        Path path;
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                for (int s = 0; s < scene->options->directSpp; s++) {
                    std::vector<SubpathContrib> spContribs;
                    path.clear();
                    GeneratePath(scene, Vector2i(x, y), std::min(scene->options->minDepth, 2), std::min(scene->options->maxDepth, 2), path, spContribs, rng);
                    for (const auto &spContrib : spContribs) {
                        Splat(buffer, spContrib.screenPos, spContrib.contrib);
                    }
                }
            }
        }
        reporter.Update(1, true);
    }, Vector2i(nXTiles, nYTiles));
    TerminateWorkerThreads();
    reporter.Done();
    Float elapsed = Tick(timer);
    std::cout << "Elapsed time: " << elapsed << " s\n" << std::endl;
}