#include "pathtrace.h"
#include "camera.h"
#include "image.h"
#include "path.h"
#include "progressreporter.h"
#include "parallel.h"
#include "timer.h"
#include "bsdf.h"

#include <algorithm>
#include <vector>
#include <unordered_map>

void PathTrace(const Scene *scene,
        const std::shared_ptr<const PathFuncLib> pathFuncLib,
        const std::string& output_name,
        bool deterministic) {
    SerializedSubpath ssubPath;

    int maxDervDepth = pathFuncLib->maxDepth;
    std::vector<Float> sceneParams(GetSceneSerializedSize());
    Serialize(scene, &sceneParams[0]);
    ssubPath.primary.resize(GetPrimaryParamSize(maxDervDepth, maxDervDepth));
    ssubPath.vertParams.resize(GetVertParamSize(maxDervDepth, maxDervDepth));

    const int spp = scene->options->spp;
    std::shared_ptr<const Camera> camera = scene->camera;
    std::shared_ptr<Image3> film = camera->film;
    film->Clear();
    const int pixelHeight = GetPixelHeight(camera.get());
    const int pixelWidth = GetPixelWidth(camera.get());
    const int tileSize = 16;
    const int nXTiles = (pixelWidth + tileSize - 1) / tileSize;
    const int nYTiles = (pixelHeight + tileSize - 1) / tileSize;

    auto pathFunc = scene->options->bidirectional ? GeneratePathBidir : GeneratePath;
    auto algo = scene->options->bidirectional ? "bdpt" : "pt";
    Timer timer;
    Tick(timer);

    // Create the temporary directory
    auto integrator = scene->options->integrator;
    fs::path partialDir = fs::path(scene->outputName).parent_path() / fs::path(integrator + "_" + algo + "_" + "partial");
    if(output_name != "") {
        // If a output name is provided, we rewrite the partial dir
        partialDir = fs::path(scene->outputName).parent_path() / fs::path(output_name + "_partial");
    }
    if (!fs::exists(partialDir))
        fs::create_directory(partialDir);

    // Global randomize
    int offsetRandom = 0;
    if(!deterministic) {
        std::random_device rd;
        offsetRandom =std::uniform_int_distribution<int>()(rd);
        std::cout << "Random offset: " << offsetRandom << "\n";
    }

    // Cumulative buffer
    SampleBuffer accumulate(pixelWidth, pixelHeight);
    const size_t offset_rng = nXTiles * nYTiles;
    size_t iteration = 0;
    while(iteration < 1000) {
        ProgressReporter reporter(nXTiles * nYTiles);
        SampleBuffer buffer(pixelWidth, pixelHeight);
        ParallelFor([&](const Vector2i tile) {
            const int seed = tile[1] * nXTiles + tile[0] + offset_rng * iteration + offsetRandom;
            RNG rng(seed);
            const int x0 = tile[0] * tileSize;
            const int x1 = std::min(x0 + tileSize, pixelWidth);
            const int y0 = tile[1] * tileSize;
            const int y1 = std::min(y0 + tileSize, pixelHeight);
            Path path;
            for (int y = y0; y < y1; y++) {
                for (int x = x0; x < x1; x++) {
                    for (int s = 0; s < spp; s++) {
                        path.clear();
                        std::vector<SubpathContrib> spContribs;
                        pathFunc(scene,
                                 Vector2i(x, y),
                                 scene->options->minDepth,
                                 scene->options->maxDepth,
                                 path,
                                 spContribs,
                                 rng);
                        for (const auto &spContrib : spContribs) {
                            if (Luminance(spContrib.contrib) <= Float(1e-10)) {
                                continue;
                            }
                            Vector3 contrib = spContrib.contrib / Float(spp);
                            Splat(buffer, spContrib.screenPos, contrib);
                        }
                    }
                }
            }
            reporter.Update(1, true);
        }, Vector2i(nXTiles, nYTiles));

        // Average the buffers
        SampleBuffer merged(pixelWidth, pixelHeight);
        MergeBuffer(accumulate, iteration / Float(iteration+1), buffer, 1.0 / Float(iteration + 1), merged);
        BufferToFilm(merged, film.get());
        CopyBuffer(merged, accumulate);

        // Update and print the time
        Float elapsed = Tick(timer);
        std::cout << "Elapsed time:" << elapsed << std::endl;

        // Write the information
        std::string base_name_fileout = integrator + "_" + algo;
        if(output_name != "") {
            base_name_fileout = output_name;
        }
        fs::path writePath = partialDir / fs::path(base_name_fileout + std::to_string(iteration + 1) + ".exr");
        auto timeFile = [&]() -> std::ofstream {
            fs::path filepath = partialDir / fs::path(base_name_fileout + "_time.csv");
            if (iteration == 0) {
                return std::ofstream(filepath.c_str(), std::ofstream::out | std::ofstream::trunc);
            } else {
                return std::ofstream(filepath.c_str(), std::ofstream::out | std::ofstream::app);
            }
        }();
        timeFile << elapsed << ",\n";
        WriteImage(writePath.c_str(), film.get());
        std::cout << " Writing to: " << writePath << std::endl;

        // Move to the next iteration
        iteration += 1;
        reporter.Done();
    }
    TerminateWorkerThreads();

}
