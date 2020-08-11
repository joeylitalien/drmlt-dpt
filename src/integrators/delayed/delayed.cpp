#include "delayed.h"


/**
 * Default constructor
 */
DRIntegrator::DRIntegrator(const Scene *scene, const std::shared_ptr<const PathFuncLib> pathFuncLib)
    : Integrator(scene, pathFuncLib) {}


/**
* MLT prepass to compute path candidates and approximate normalization constant
*/
Float DRIntegrator::PrePass(const MLTState &mltState,
                            const int64_t numInitSamples,
                            const int numChains,
                            MarkovStates &initStates,
                            std::shared_ptr<PiecewiseConstant1D> &lengthDist) {

    // Start clock
    std::cout << "Initializing MLT..." << std::endl;
    Timer timer;
    Tick(timer);

    const int64_t numSamplesPerThread = numInitSamples / NumSystemCores();
    const int64_t threadsNeedExtraSamples = numSamplesPerThread % NumSystemCores();
    const Scene *scene = mltState.scene;

    // Path sampler (like trace): path or bidir
    auto genPathFunc = mltState.genPathFunc;

    // Sync object to lock
    std::mutex mStateMutex;

    // Markov state
    struct LightMarkovState {
        RNG rng;
        int camDepth;
        int lightDepth;
        Float lsScore;
    };

    int offsetRandom = 0;
    if(!deterministic) {
        std::random_device rd;
        offsetRandom =std::uniform_int_distribution<int>()(rd);
        std::cout << "Random offset: " << offsetRandom << "\n";
    }

    // List of LightMarkovState
    std::vector<LightMarkovState> mStates;
    auto totalScore = Float(0.0);

    // Vector of length equal to the path length of all the paths
    // Each entry is the sum of the contribution of all paths with same length
    // This is use if multiplex is turned on to importance sample a strategy
    std::vector<Float> lengthContrib;

    // Dispatch a regular path tracer/bidir to threads
    ParallelFor([&](const int threadId) {
        // Random generator init with a unique seed
        RNG rng(threadId + scene->options->seedOffset + offsetRandom);

        // Number of sample for current thread
        int64_t numSamplesThisThread = numSamplesPerThread + ((threadIndex < threadsNeedExtraSamples) ? 1 : 0);

        // List of path contributions
        std::vector<SubpathContrib> spContribs;

        // Path (list of vertices)
        Path path;

        // Loop through all paths to generate
        for (int sampleIdx = 0; sampleIdx < numSamplesThisThread; sampleIdx++) {
            // Clear path contribution
            spContribs.clear();

            // Backup random number generator, we need to keep a copy of the original for replay
            RNG rngCheckpoint = rng;

            // Clear path
            path.clear();

            // Minimum path length for a valid path
            const int minPathLength = std::max(scene->options->minDepth, 3);

            // Call trace function to generate a path of length [minPathLength, maxDepth]
            // Here spContribs can have more than one contribs. For bidir it can go:
            // 0: no path
            // 1: if sensor path hit an emitter (s=0, t>0)
            // 2: if an explicit light connection is done (s=1, t>0)
            genPathFunc(scene, Vector2i(-1, -1), minPathLength, scene->options->maxDepth, path, spContribs, rng);

            // === LOCK START ===
            std::lock_guard<std::mutex> lock(mStateMutex);

            // Loop over all contributions
            for (const auto &spContrib : spContribs) {

                // Get total contribution
                totalScore += spContrib.lsScore;

                // Compute path length (camLength + lgtLength - 1)
                const int pathLength = GetPathLength(spContrib.camDepth, spContrib.lightDepth);

                // Resize lengthContrib if it's too short
                if (pathLength >= int(lengthContrib.size())) {
                    lengthContrib.resize(pathLength + 1, Float(0.0));
                }

                // Accumulate total contributions for this path length
                lengthContrib[pathLength] += spContrib.lsScore;

                // Store a replayable path (LightMarkovState) with some extra value (spContrib.lsScore) to check that replay is correct
                mStates.emplace_back(LightMarkovState{rngCheckpoint, spContrib.camDepth, spContrib.lightDepth,
                                                        spContrib.lsScore});
            }
            // === LOCK END ===
        }
    }, NumSystemCores());

    // Build CDF of path contributions by length
    lengthDist = std::make_shared<PiecewiseConstant1D>(&lengthContrib[0], lengthContrib.size());

    // Equal-spaced seeding (See p.340 in Veach's thesis)
    std::vector<Float> cdf(mStates.size() + 1);
    cdf[0] = Float(0.0);
    for (int i = 0; i < (int) mStates.size(); i++) {
        cdf[i + 1] = cdf[i] + mStates[i].lsScore;
    }
    const Float interval = cdf.back() / Float(numChains);
    std::uniform_real_distribution<Float> uniDist(Float(0.0), interval);
    RNG rng(mStates.size());
    Float pos = uniDist(rng);
    int cdfPos = 0;
    initStates.reserve(numChains);
    std::vector<SubpathContrib> spContribs;
    for (int i = 0; i < numChains; i++) {
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
    std::cout << "Elapsed time: " << elapsed << std::endl;
    return Float(totalScore) * invNumInitSamples;
}


/**
 * Actual rendering algorithm
 * Core of the Delayed Rejection Metropolis Light Transport DRMLT (2020).
 * this is the implementation of the cheap-then-expensive approach.
 */
void DRIntegrator::Render() {
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

    // Prepare film
    const int spp = scene->options->spp;
    std::shared_ptr<const Camera> camera = scene->camera;
    std::shared_ptr<Image3> film = camera->film;
    film->Clear();
    const int pixelHeight = GetPixelHeight(camera.get());
    const int pixelWidth = GetPixelWidth(camera.get());

    // Compute direct lighting
    SampleBuffer directBuffer(pixelWidth, pixelHeight);
    DirectLighting(scene, directBuffer);

    // Compute number of samples/chains
    const int64_t numPixels = int64_t(pixelWidth) * int64_t(pixelHeight);
    const int64_t totalSamples = int64_t(spp) * numPixels;
    const int64_t numChains = scene->options->numChains;
    const int64_t numSamplesPerChain = totalSamples / numChains;
    const int64_t chainsNeedExtraSamples = numSamplesPerChain % numChains;

    // List of seed path (MarkovChain) computed during prepass
    MarkovStates initStates;

    // Path contribution distributed by path lengths
    std::shared_ptr<PiecewiseConstant1D> lengthDist;

    // Return the 'b' normalization term and populate 'initStates' with Markov Chains generated during an MC pass
    const Float avgScore = PrePass(mltState, scene->options->numInitSamples, numChains, initStates, lengthDist);
    std::cout << "Average brightness: " << avgScore << std::endl;
    const Float normalization = avgScore;

    // Track progress
    ProgressReporter reporter(totalSamples);
    const int reportInterval = 16384;
    const int reportIntervalStats = 163840;
    int intervalImgId = 1;

    // Prepare buffer for indirect lighting
    SampleBuffer indirectBuffer(pixelWidth, pixelHeight);
    Timer timer;
    Tick(timer);

    // Intermediate outputs
    auto integrator = scene->options->integrator;
    fs::path partialDir = fs::path(scene->outputName).parent_path() / fs::path(integrator + "_" + "partial");
    if(output_name != "") {
        // If a output name is provided, we rewrite the partial dir
        partialDir = fs::path(scene->outputName).parent_path() / fs::path(output_name + "_partial");
    }
    if (!fs::exists(partialDir))
        fs::create_directory(partialDir);

    /* First and second stage acceptance colors (for visualization) 
    Red for first stage, Green for second*/
    bool drawAcceptanceMap = scene->options->acceptanceMap;
    std::pair<Vector3, Vector3> stageColors = {Vector3(1,0,0), Vector3(0,1,0)};

    // Useful lambdas
    auto flipCoin = [](Float x, Float d) { return (x == 1) || (d < x); };
    auto metropolisClamp = [](Float x) { return std::min((Float) 1.0f, x); };
    auto isInvalid = [](Float x) { return std::isnan(x) || std::isinf(x) || x < 0; };

    // Lambda to splat with some weight onto indirect film buffer
    auto WeightedSplat = [&](MarkovState &state, Float weight) {
        if (!drawAcceptanceMap) {
            for (const auto &splat : state.toSplat) {
                assert(std::isfinite(Luminance(splat.contrib)));
                Vector3 value = weight * splat.contrib;
                if (!value.isZero()) {
                    Splat(indirectBuffer, splat.screenPos, value);
                }
            }
        }
    };

    // Lambda to splat the acceptance map instead of the rendered image (build bin)
    auto splatAcceptanceOnly = [&](MarkovState &state, int stage) {
        if (drawAcceptanceMap) {
            for (const auto &splat : state.toSplat) {
                Vector3 value = stage == 0 ? stageColors.first : stageColors.second;
                Splat(indirectBuffer, splat.screenPos, value);
            }
        }
    };
    
	int offsetRandom = 0;
    if(!deterministic) {
        std::random_device rd;
        offsetRandom =std::uniform_int_distribution<int>()(rd);
        std::cout << "Random offset: " << offsetRandom << "\n";
    }

    /**
     * Rendering statistics utility
    */
    struct Stats {
        size_t largeStep = 0;
        size_t largeStepAccepted = 0;
        size_t stage1 = 0;
        size_t stage1Accepted = 0;
        size_t stage2Iso = 0;
        size_t stage2IsoAccepted = 0;
        size_t stage2H2MC = 0;
        size_t stage2H2MCAccepted =0;

        /// increment the counters
        Stats& operator +=(const Stats& b) {
            largeStep += b.largeStep;
            largeStepAccepted += b.largeStepAccepted;
            stage1 += b.stage1;
            stage1Accepted += b.stage1Accepted;
            stage2Iso += b.stage2Iso;
            stage2IsoAccepted += b.stage2IsoAccepted;
            stage2H2MC += b.stage2H2MC;
            stage2H2MCAccepted += b.stage2H2MCAccepted;
            return *this;
        }

        /// output the current statistics
        void output(std::ostream& out) const {
            Float ratioLarge = (Float)largeStepAccepted/largeStep;
            Float ratioStage1 = (Float)stage1Accepted/stage1;
            Float ratioStage2H2MC = (Float)stage2H2MCAccepted/stage2H2MC;
            Float ratioStage2Iso = (Float)stage2IsoAccepted/stage2Iso;
            out << "\n";
            out << "Rendering Statistics \n";
            out << "    Large step mutations acceptance ratio                        : " << ratioLarge << " (" << largeStep << ")\n";
            out << "    First stage mutations acceptance ratio                       : " << ratioStage1 << " (" << stage1 << ")\n";
            out << "    Second stage mutations acceptance ratio, isotropic component : " << ratioStage2H2MC << " (" << stage2H2MC << ")\n";
            out << "    Second stage mutations acceptance ratio, H2MC component      : " << ratioStage2Iso << " (" << stage2Iso << ")\n";
            out << "\n";
            
        }
        /// print the rendering statistics
        void print() const {
            output(std::cout);
        }
    };
    std::mutex mutexGlobalStats;
    Stats globalStats;

    // Run Markov chains
    ParallelFor([&](const int chainId) {
        Stats currStats;
        const int seed = chainId + scene->options->seedOffset + offsetRandom;
        RNG rng(seed);
        std::uniform_real_distribution<Float> uniDist(Float(0.0), Float(1.0));
        int64_t numSamplesThisChain = numSamplesPerChain + ((chainId < chainsNeedExtraSamples) ? 1 : 0);
        std::vector<Float> contribCdf;

        // States
        MarkovState currentState = initStates.at(chainId);   // current state
        std::pair<MarkovState, MarkovState> proposalStates;  // proposed state, both stage
        MarkovState proposalStateStar;                       // EGreen - y* = z-(y-x) intermediate state


        // Pointers to large step mutation routine
        auto largeStep = std::unique_ptr<LargeStep>(new LargeStep(lengthDist));

        // Set type of DR
        const DRIntegrator::EType type = [&]() -> DRIntegrator::EType {
            auto implementation = scene->options->type;
            if (implementation == "green") {
                return DRIntegrator::EGreen;
            } else if (implementation == "mira") {
                return DRIntegrator::EMira;
            } else {
                return DRIntegrator::EGreen;
                std::cout << "Unknown implementation type, set to green" << std::endl;
            }
        }();


        // Delayed rejection variables
        bool isLargeStep;                          // Is a largestep or smallstep mutation
        bool isAniso;                              // Is anisotropic or isotropic second stage
        bool doSecond;                             // Do a second stage or not

        std::pair<Float, Float> a;                 // acceptance probability of both stage
        Float aStar;                               // EGreen - acceptance probability of y* intermediate state

        std::pair<bool, bool> accept;              // accept a proposed state or not (both stage)
        std::pair<Float, Float> proposedWeights;   // splatting weight of proposed state (both stage)

        // Transition kernel hierarchy
        // first stage - iso gaussian
        // second stage - mixture of iso and aniso.
        std::pair<std::unique_ptr<SmallStep>, std::unique_ptr<SmallStep>> stages = {
            // first stage, isotropic gaussian. will be use for the second stage wehn sampling from iso
            // mutation.cpp -> SmallStep
            std::unique_ptr<SmallStep>(new SmallStep()),
            // second stage, used when sampling from th anisotropic component
            // drsmallstep.cpp -> PH2MCSmallStep
            std::unique_ptr<SmallStep>(new PH2MCSmallStep(scene, 
                                                        pathFuncLib->maxDepth, 
                                                        scene->options->perturbStdDev))
        };

        // Aliases/shorthands to simplify code  (splatting contribution)
        SubpathContrib &current = currentState.spContrib;
        SubpathContrib &first   = proposalStates.first.spContrib;
        SubpathContrib &second  = proposalStates.second.spContrib;

        // Loop over all samples for this chain
        for (int sampleIdx = 0; sampleIdx < numSamplesThisChain; sampleIdx++) {
            
            // Reset DR variables
            aStar = 0.f;
            a = {0.f, 0.f};
            accept = {false, false};
            proposedWeights = {0.f, 0.f};
            isAniso = false;
            doSecond = false;

            // Select if largestep of not
            isLargeStep = flipCoin(scene->options->largeStepProb, uniDist(rng));

            // FIRST STAGE

            // Large step case, uniform
            if (!currentState.valid || isLargeStep) {
                currStats.largeStep += 1;
                // large Step - trace y and compute the first stage acceptance. See DRMLT (2020), Eq. 5
                a.first = largeStep->Mutate(mltState, normalization, currentState, proposalStates.first, rng);
                
                // accept with probability a.first
                if (isInvalid(a.first) /*|| !proposalStates.first.valid*/) {
                    a.first = 0.0;
                    accept.first = false;
                } else {
                    accept.first = flipCoin(a.first, uniDist(rng));
                    if (accept.first) currStats.largeStepAccepted += 1;
                }
            // Small step case, isotropic, will do DR
            } else {
                currStats.stage1 += 1;
                // First stage - trace y and compute the first stage acceptance. See DRMLT (2020), Eq. 5
                a.first = stages.first->Mutate(mltState, normalization, currentState, proposalStates.first, rng);

                // accept with probability a.first
                if (isInvalid(a.first) /*|| !proposalStates.first.valid*/) {
                    a.first = 0.0;
                    accept.first = false;
                } else {
                    accept.first = flipCoin(a.first, uniDist(rng));
                    if (accept.first) currStats.stage1Accepted += 1;
                }

            }

            // if first stage is rejected, do a second stage
            doSecond = !accept.first;
            // unless its a large step, then skip
            doSecond = doSecond && !isLargeStep && currentState.valid;

            // SECOND STAGE
            if (doSecond) {

                // Sample component of the mixture
                isAniso = flipCoin(scene->options->anisoPerturbProb, uniDist(rng));
                
                // if aniso sample from the second stage anisotropic proposal using hessian
                if (isAniso) {
                    // Second Stage iso - trace z and comopute usual MH acceptance.
                    a.second = stages.second->Mutate(mltState, normalization, currentState, proposalStates.second, rng);
                    currStats.stage2H2MC += 1;
                } 
                // if iso sample from first stage proposal with smaller std dev
                else {
                    // Second Stage iso - trace z and comopute usual MH acceptance. since aniso Q_2(x,z)/Q_2(z,x) is computed
                    a.second = stages.first->Mutate(mltState, normalization, currentState, proposalStates.second, scene->options->anisoperturbstddev, rng);
                    currStats.stage2Iso += 1;
                }

                
                // Depending on the delayed rejection framework, accept or not
                if (isInvalid(a.second) /*|| !proposalStates.second.valid*/) {
                    a.second = 0.0;
                    accept.second = false;
                } else {
                     // Green & Mira (2001): use reverse path y^* to remove transition kernel ratio
                    if (type == DRIntegrator::EGreen) {

                        // generate offset vector to get y* = z + x - y
                        auto x = GetPathPos(currentState.path);
                        auto y = GetPathPos(proposalStates.first.path);
                        Vector offset = x - y;
                        assert(x.size() == y.size() && y.size()  == offset.size());

                        // trace y* and compute the reverse acceptance from z to y*. See DRMLT (2020), Eq. 13
                        aStar = stages.first->Trace(mltState, offset, proposalStates.second, proposalStateStar, rng);
                        aStar = isInvalid(aStar)? 0.0 : aStar;

                        // Accept through Green's second stage acceptance formula. See DRMLT (2020), Eq. 14 
                        if (aStar == 1.0) {
                                a.second = Float(0.0);
                                accept.second = false;
                        } else {
                            a.second = metropolisClamp(a.second * (1.0f - aStar) / (1.0f - a.first));
                            accept.second = flipCoin(a.second, uniDist(rng));
                            if (accept.second) {
                                if (isAniso) {
                                    currStats.stage2H2MCAccepted += 1;
                                } else {
                                    currStats.stage2IsoAccepted += 1;
                                }
                            }
                        }
                    }
                    // Tierney & Mira (1999): Naive approach, low acceptance rate at second stage
                    else if (type == DRIntegrator::EMira){
                        // compute the first stage reverse acceptance from z to y. See DRMLT (2020), Eq. 5 with x = z 
                        aStar = metropolisClamp(first.ssScore / second.ssScore);
                        aStar = isInvalid(aStar)? 1.0 : aStar;
                        
                        if (aStar == 1.0) {
                            a.second = Float(0.0);
                            accept.second = false;
                        } else {

                            /* Compute transition kernels ratio Q_1(z,y)/Q_1(x,y) */
                            Float transRatio = 1.0;
                            auto x = GetPathPos(currentState.path);
                            auto y = GetPathPos(proposalStates.first.path);
                            auto z = GetPathPos(proposalStates.second.path);
                            Vector offsetxy = x-y;
                            Vector offsetzy = z-y;
                            assert(x.size() == y.size() && y.size() == z.size());
                            Gaussian gaussIso;
                            IsotropicGaussian(y.size(), scene->options->perturbStdDev, gaussIso);
                            transRatio = std::exp(
                                    GaussianLogPdf(offsetzy, gaussIso)-GaussianLogPdf(offsetxy, gaussIso));

                            // Accept with the naive second stage acceptance formula. See DRMLT (2020), Eq. 7 
                            if (isInvalid(transRatio)) {
                                a.second = 0.0f;
                                accept.second = false;
                            } else {
                                a.second = metropolisClamp(a.second * transRatio * (1.0f - aStar) / (1.0f - a.first));
                                accept.second = flipCoin(a.second, uniDist(rng));
                                if (accept.second) {
                                    if (isAniso) {
                                        currStats.stage2H2MCAccepted += 1;
                                    } else {
                                        currStats.stage2IsoAccepted += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            

            // Compute acceptance weights for splatting. See DRMLT (2020), Fig. 10
            Float currentWeight;
            proposedWeights.first = a.first;
            proposedWeights.second = (1.0f - proposedWeights.first) * a.second;
            currentWeight = 1.0f - proposedWeights.first - proposedWeights.second;

            // Splat current
            if (currentState.valid && currentWeight > 0)
                WeightedSplat(currentState, currentWeight);
            // Splat first stage proposal
            if (proposedWeights.first > 0)
                WeightedSplat(proposalStates.first, proposedWeights.first);
            // Splat second stage proposal
            if (proposedWeights.second > 0)
                WeightedSplat(proposalStates.second, proposedWeights.second);

            bool acceptedFirst;
            // Either accept one of the proposed states 
            if (accept.first || accept.second) {
                /* Either first stage was accepted */
                if (accept.first) {
                    acceptedFirst = true;

                    proposalStates.first.path.Initialize(first.camDepth, first.lightDepth);
                    std::swap(currentState, proposalStates.first);
                    currentState.gaussianInitialized = false;

                    // Add contribution to acceptance map 
                    if (!isLargeStep) {
                        splatAcceptanceOnly(proposalStates.first, 0);
                    }

                } 
                // Or the second
                else {
                    acceptedFirst = false;
                    
                    proposalStates.second.path.Initialize(second.camDepth, second.lightDepth);
                    std::swap(currentState, proposalStates.second);
                    /* Sanity Check: If the stage was isotropic, we did not compute the gradient and 
                    hessian of this state. If in the futur, we need to do an anisotropic mutation
                    from this state, we will need to compute them to do so. 
                    */
                    if (!isAniso) {
                        currentState.gaussianInitialized = false;
                    }

                    // Add contribution to acceptance map
                    splatAcceptanceOnly(proposalStates.second, 1);
                }

                // Accepted, therefore new state is a valid one
                currentState.valid = true;

                // Stats
                if (isLargeStep) {
                    largeStep->lastScoreSum = currentState.scoreSum;
                    largeStep->lastScore = current.lsScore;
                }
            }
            // or do nothing and reject

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

    // Show statistics
    reporter.Done();
    Float elapsed = Tick(timer);

    // Save film to image
    SampleBuffer buffer(pixelWidth, pixelHeight);
    Float directWeight = scene->options->directSpp > 0 ? inverse(Float(scene->options->directSpp)) : Float(0.0);
    Float indirectWeight = spp > 0 ? inverse(Float(spp)) : Float(0.0);
    MergeBuffer(directBuffer, directWeight, indirectBuffer, indirectWeight, buffer);
    BufferToFilm(buffer, film.get());
}