#include "mcmc.h"


/**
 * For comments, refer to ./delayed/[delayed.cpp AND drsmallstep.cpp]
*/

MCMCIntegrator::MCMCIntegrator(const Scene *scene, const std::shared_ptr<const PathFuncLib> pathFuncLib) : Integrator(scene, pathFuncLib) {
}

void MCMCIntegrator::Render(){

    const MLTState mltState{scene,
                            scene->options->bidirectional ? GeneratePathBidir : GeneratePath,
                            scene->options->bidirectional ? PerturbPathBidir : PerturbPath,
                            scene->options->bidirectional ? PerturbLensBidir : PerturbLens,
                            pathFuncLib->funcMap,
                            pathFuncLib->dervFuncMap,
                            pathFuncLib->staticFuncMap,
                            pathFuncLib->staticDervFuncMap,
                            pathFuncLib->lensFuncMap,
                            pathFuncLib->lensDervFuncMap};
    const int spp = scene->options->spp;
    std::shared_ptr<const Camera> camera = scene->camera;
    const Float largeStepProb = scene->options->largeStepProb;
    std::shared_ptr<Image3> film = camera->film;
    film->Clear();
    const int pixelHeight = GetPixelHeight(camera.get());
    const int pixelWidth = GetPixelWidth(camera.get());
    SampleBuffer directBuffer(pixelWidth, pixelHeight);
    DirectLighting(scene, directBuffer);

    const int64_t numPixels = int64_t(pixelWidth) * int64_t(pixelHeight);
    const int64_t totalSamples = int64_t(spp) * numPixels;
    const int64_t numChains = scene->options->numChains;
    const int64_t numSamplesPerChain = totalSamples / numChains;
    const int64_t chainsNeedExtraSamples = numSamplesPerChain % numChains;

    MarkovStates initStates;
    std::shared_ptr<PiecewiseConstant1D> lengthDist;
    const Float avgScore = this->PrePass(mltState, scene->options->numInitSamples, numChains, initStates, lengthDist);
    std::cout << "Average brightness: " << avgScore << std::endl;
    const Float normalization = avgScore;

    ProgressReporter reporter(totalSamples);
    const int reportInterval = 16384;
    const int reportIntervalStats = 163840;
    int intervalImgId = 1;

    // Intermediate outputs
    auto integrator = scene->options->integrator;
    fs::path partialDir = fs::path(scene->outputName).parent_path() / fs::path(integrator + "_" + "partial");
    if(output_name != "") {
        partialDir = fs::path(scene->outputName).parent_path() / fs::path(output_name + "_partial");
    }
    if (!fs::exists(partialDir))
        fs::create_directory(partialDir);

    struct Stats {
        size_t largeStep = 0;
        size_t largeStepAccepted = 0;
        size_t smallStep = 0;
        size_t smallStepAccepted = 0;
        size_t lensStep = 0;
        size_t lensStepAccepted = 0;

        Stats& operator +=(const Stats& b) {
            largeStep += b.largeStep;
            largeStepAccepted += b.largeStepAccepted;
            smallStep += b.smallStep;
            smallStepAccepted += b.smallStepAccepted;
            lensStep += b.lensStep;
            lensStepAccepted += b.lensStepAccepted;
            return *this;
        }

        void output(std::ostream& out) const {
            Float ratioLarge = (Float)largeStepAccepted/largeStep;
            Float ratioSmall = (Float)smallStepAccepted/smallStep;
            Float ratioLens = (Float)lensStepAccepted/lensStep;
            out << "\n";
            out << "Rendering Statistics \n";
            out << "    Large step mutations acceptance ratio       : " << ratioLarge << " (" << largeStep << ")\n";
            out << "    Small step mutations acceptance ratio       : " << ratioSmall << " (" << smallStep << ")\n";
            out << "    Lens mutations acceptance ratio    :                    " << ratioLens << " (" << lensStep << ")\n";
            out << "\n";
        }
        void print() const {
            output(std::cout);
        }
    };
    std::mutex mutexGlobalStats;
    Stats globalStats;

    SampleBuffer indirectBuffer(pixelWidth, pixelHeight);
    Timer timer;
    Tick(timer);

    int offsetRandom = 0;
    if(!deterministic) {
        std::random_device rd;
        offsetRandom =std::uniform_int_distribution<int>()(rd);
        std::cout << "Random offset: " << offsetRandom << "\n";
    }

    ParallelFor([&](const int chainId) {
        Stats currStats;
        const int seed = chainId + scene->options->seedOffset + offsetRandom;
        RNG rng(seed);
        std::uniform_real_distribution<Float> uniDist(Float(0.0), Float(1.0));

        int64_t numSamplesThisChain = numSamplesPerChain + ((chainId < chainsNeedExtraSamples) ? 1 : 0);
        std::vector<Float> contribCdf;
        MarkovState currentState = initStates.at(chainId);
        MarkovState proposalState;

        std::unique_ptr<LargeStep> largeStep = std::unique_ptr<LargeStep>(new LargeStep(lengthDist));

        std::unique_ptr<Mutation> smallStep;

        smallStep = std::unique_ptr<Mutation>(new SmallStep());

        for (int sampleIdx = 0; sampleIdx < numSamplesThisChain; sampleIdx++) {
            auto a = Float(1.0);
            bool isLargeStep = false;
            if (!currentState.valid || uniDist(rng) < largeStepProb) {
                isLargeStep = true;
                a = largeStep->Mutate(mltState, normalization, currentState, proposalState, rng);
            } else {
                a = smallStep->Mutate(mltState, normalization, currentState, proposalState, rng);
            }
            if (currentState.valid && a < Float(1.0)) {
                for (const auto &splat : currentState.toSplat) {
                    Splat(indirectBuffer, splat.screenPos, (Float(1.0) - a) * splat.contrib);
                }
            }
            if (a > Float(0.0)) {
                for (const auto &splat : proposalState.toSplat) {
                    Splat(indirectBuffer, splat.screenPos, a * splat.contrib);
                }
            }

            if (a > Float(0.0) && uniDist(rng) <= a) {
                proposalState.path.Initialize(proposalState.spContrib.camDepth, proposalState.spContrib.lightDepth);
                std::swap(currentState, proposalState);
                currentState.valid = true;
                if (isLargeStep) {
                    largeStep->lastScoreSum = currentState.scoreSum;
                    largeStep->lastScore = currentState.spContrib.lsScore;
                    currentState.lensGaussianInitialized = false;
                    currentState.gaussianInitialized = false;
                    currStats.largeStepAccepted++;
                } else {
                    if (smallStep->lastMutationType == MutationType::Small)
                        currStats.smallStepAccepted++;
                    else if (smallStep->lastMutationType == MutationType::Lens)
                        currStats.lensStepAccepted++;
                }
            }

            if (isLargeStep) {
                currStats.largeStep++;
            } else {
                if (smallStep->lastMutationType == MutationType::Small)
                    currStats.smallStep++;
                else if (smallStep->lastMutationType == MutationType::Lens)
                    currStats.lensStep++;
            }

            // Intermediate reports
            if (sampleIdx > 0 && (sampleIdx % reportInterval == 0)) {
                reporter.Update(reportInterval, verbose);
                if (threadIndex == 0 && scene->options->reportIntervalSeconds > 0) {
                    long interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now() - timer.last).count();
                    Float elapsed = Float(interval) / 1000.f;
                    if(verbose) {
                        std::cout << " Interval: " << elapsed << " s" << std::endl;
                    }
                    if (elapsed >= (scene->options->reportIntervalSeconds * intervalImgId)) {
                        BufferToFilm(indirectBuffer, film.get(), Float(numPixels) / Float(reporter.GetWorkDone()));
                        std::string base_name_fileout = integrator;
                        if(output_name != "") {
                            base_name_fileout = output_name;
                        }
                        fs::path writePath = partialDir / fs::path(base_name_fileout + "_" + std::to_string(intervalImgId) + ".exr");
                        auto timeFile = [&]() -> std::ofstream {
                            fs::path filepath = partialDir / fs::path(base_name_fileout + "_time.csv");
                            if (intervalImgId == 1) {
                                return std::ofstream(filepath.c_str(), std::ofstream::out | std::ofstream::trunc);
                            } else {
                                return std::ofstream(filepath.c_str(), std::ofstream::out | std::ofstream::app);
                            }
                        }();
                        timeFile << elapsed << ",\n";
                        WriteImage(writePath.c_str(), film.get());
                        std::cout << " Writing to: " << writePath << std::endl;
                        intervalImgId++;
                    }
                }
            }
            if(sampleIdx > 0 && sampleIdx % reportIntervalStats == 0 ) {
                std::lock_guard<std::mutex> lock(mutexGlobalStats);
                globalStats += currStats;
                currStats = Stats(); // Reset
                if(threadIndex == 0) {
                    globalStats.print();
                    std::string base_name_fileout = integrator;
                    if(output_name != "") {
                        base_name_fileout = output_name;
                    }
                    fs::path writePath = partialDir / fs::path(base_name_fileout + "_stats.txt");
                    std::ofstream outputFile(writePath.c_str(), std::ofstream::out | std::ofstream::trunc);
                    globalStats.output(outputFile);
                }
            }
        }
        reporter.Update(numSamplesThisChain % reportInterval, verbose);
        {
            std::lock_guard<std::mutex> lock(mutexGlobalStats);
            globalStats += currStats;
            if(threadIndex == 0) {
                globalStats.print();
                std::string base_name_fileout = integrator;
                if(output_name != "") {
                    base_name_fileout = output_name;
                }
                fs::path writePath = partialDir / fs::path(base_name_fileout + "_stats.txt");
                std::ofstream outputFile(writePath.c_str(), std::ofstream::out | std::ofstream::trunc);
                globalStats.output(outputFile);
            }
        }
    }, numChains);
    TerminateWorkerThreads();
    reporter.Done();
    Float elapsed = Tick(timer);
    std::cout << "Elapsed time:" << elapsed << std::endl;

    SampleBuffer buffer(pixelWidth, pixelHeight);
    Float directWeight = scene->options->directSpp > 0 ? inverse(Float(scene->options->directSpp)) : Float(0.0);
    Float indirectWeight = spp > 0 ? inverse(Float(spp)) : Float(0.0);
    MergeBuffer(directBuffer, directWeight, indirectBuffer, indirectWeight, buffer);
    BufferToFilm(buffer, film.get());
}

Float MCMCIntegrator::PrePass(const MLTState &mltState, const int64_t numInitSamples, const int numChains, MarkovStates &initStates, std::shared_ptr<PiecewiseConstant1D> &lengthDist)  {
    std::cout << "Initializing mlt" << std::endl;
    Timer timer;
    Tick(timer);

    const int64_t numSamplesPerThread = numInitSamples / NumSystemCores();
    const int64_t threadsNeedExtraSamples = numSamplesPerThread % NumSystemCores();
    const Scene *scene = mltState.scene;
    auto genPathFunc = mltState.genPathFunc;

    int offsetRandom = 0;
    if(!deterministic) {
        std::random_device rd;
        offsetRandom =std::uniform_int_distribution<int>()(rd);
        std::cout << "Random offset: " << offsetRandom << "\n";
    }

    std::mutex mStateMutex;
    struct LightMarkovState {
        RNG rng;
        int camDepth;
        int lightDepth;
        Float lsScore;
    };
    std::vector<LightMarkovState> mStates;
    auto totalScore(Float(0.0));
    std::vector<Float> lengthContrib;
    ParallelFor([&](const int threadId) {
        RNG rng(threadId + scene->options->seedOffset + offsetRandom);
        int64_t numSamplesThisThread = numSamplesPerThread + ((threadIndex < threadsNeedExtraSamples) ? 1 : 0);
        std::vector<SubpathContrib> spContribs;
        Path path;
        for (int sampleIdx = 0; sampleIdx < numSamplesThisThread; sampleIdx++) {
            spContribs.clear();
            RNG rngCheckpoint = rng;
            path.clear();
            const int minPathLength = std::max(scene->options->minDepth, 3);
            genPathFunc(scene,
                        Vector2i(-1, -1),
                        minPathLength,
                        scene->options->maxDepth,
                        path,
                        spContribs,
                        rng);

            std::lock_guard<std::mutex> lock(mStateMutex);
            for (const auto &spContrib : spContribs) {
                totalScore += spContrib.lsScore;
                const int pathLength = GetPathLength(spContrib.camDepth, spContrib.lightDepth);
                if (pathLength >= int(lengthContrib.size())) {
                    lengthContrib.resize(pathLength + 1, Float(0.0));
                }
                lengthContrib[pathLength] += spContrib.lsScore;
                mStates.emplace_back(LightMarkovState{rngCheckpoint, spContrib.camDepth, spContrib.lightDepth, spContrib.lsScore});
            }
        }
    }, NumSystemCores());

    lengthDist = std::make_shared<PiecewiseConstant1D>(&lengthContrib[0], lengthContrib.size());

    if (int(mStates.size()) < numChains) {
        Error(
                "MLT initialization failed, consider using a larger number of initial samples or "
                "smaller number of chains");
    }

    // Equal-spaced seeding (See p.340 in Veach's thesis)
    std::vector<Float> cdf(mStates.size() + 1);
    cdf[0] = Float(0.0);
    for (int i = 0; i < (int)mStates.size(); i++) {
        cdf[i + 1] = cdf[i] + mStates[i].lsScore;
    }
    const Float interval = cdf.back() / Float(numChains);
    std::uniform_real_distribution<Float> uniDist(Float(0.0), interval);
    RNG rng(mStates.size());
    Float pos = uniDist(rng);
    int cdfPos = 0;
    initStates.reserve(numChains);
    std::vector<SubpathContrib> spContribs;
    for (int i = 0; i < (int)numChains; i++) {
        while (pos > cdf[cdfPos]) {
            cdfPos = std::min(cdfPos + 1, int(mStates.size()) - 1);
        }
        MarkovState &state = initStates.add();
        state.setValid(true);

        spContribs.clear();
        state.path.clear();
        RNG rngCheckpoint = mStates[cdfPos - 1].rng;
        genPathFunc(scene,
                    Vector2i(-1, -1),
                    std::max(scene->options->minDepth, 3),
                    scene->options->maxDepth,
                    state.path,
                    spContribs,
                    rngCheckpoint);
        state.scoreSum = Float(0.0);
        for (const auto &spContrib : spContribs) {
            state.scoreSum += spContrib.lsScore;
            if (spContrib.camDepth == mStates[cdfPos - 1].camDepth &&
                spContrib.lightDepth == mStates[cdfPos - 1].lightDepth) {
                state.spContrib = spContrib;
            }
        }
        state.path.Initialize(state.spContrib.camDepth, state.spContrib.lightDepth);
        state.gaussianInitialized = false;
        state.lensGaussianInitialized = false;
        pos += interval;
    }

    Float invNumInitSamples = inverse(Float(numInitSamples));
    Float elapsed = Tick(timer);
    std::cout << "Elapsed time:" << elapsed << std::endl;
    return Float(totalScore) * invNumInitSamples;
}
