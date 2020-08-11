#include "h2mc.h"

/**
 * For comments, refer to ./delayed/[delayed.cpp AND drsmallstep.cpp]
*/

#define H2MC_COMPUTEGAUSSIAN(S_) \
    case S_: {\
    ComputeGaussian<S_>(param, grad, hess, invSigmaSq, gaussian);\
    break;\
    }

template <int dim>
void ComputeGaussian(const H2MCIntegrator::H2MCParam &param,
                    const Eigen::Matrix<Float, dim, 1> &grad,
                    const Eigen::Matrix<Float, dim, dim> &hess,
                    const Eigen::Matrix<Float, dim, 1> &invSigmaSq,
                    Gaussian &gaussian) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Float, dim, dim>> eigenSolver;
    eigenSolver.compute(hess);

    const auto &hEigenvector = eigenSolver.eigenvectors();
    const auto &hEigenvalues = eigenSolver.eigenvalues();
    Eigen::Matrix<Float, dim, 1> eigenBuff;
    Eigen::Matrix<Float, dim, 1> offsetBuff;
    Eigen::Matrix<Float, dim, 1> postInvCovEigenvalues;
    int dimension = (dim == -1 ? int(grad.size()) : dim);
    eigenBuff.resize(dimension);
    offsetBuff.resize(dimension);
    postInvCovEigenvalues.resize(dimension);

    // σ = Small step perturbation size
    Float sigma = param.sigma;
    Float sigmaSq = sigma * sigma;
    Float invSigmaSqScal = Float(1.0) / sigmaSq;

    for (int i = 0; i < dimension; i++) {
        if (fabs(hEigenvalues(i)) > Float(1e-10)) {
            eigenBuff(i) = 1.0 / fabs(hEigenvalues(i));
        } else {
            eigenBuff(i) = 0.0;
        }
    }

    offsetBuff.noalias() = eigenBuff.asDiagonal() * (hEigenvector.transpose() * grad);
    for (int i = 0; i < dimension; i++) {
        Float s2 = Float(1.0);
        Float o = Float(0.0);
        if (fabs(hEigenvalues(i)) > Float(1e-10)) {
            o = offsetBuff(i);
            if (hEigenvalues(i) > 0.0) {
                s2 = param.posScaleFactor;   // Float(0.5) * (exp(c_L) - exp(-c_L)) ^ 2;
                o *= param.posOffsetFactor;  //(Float(0.5) * (exp(c_L) + exp(-c_L)) - Float(1.0));
            } else {                         //<= 0.0
                s2 = param.negScaleFactor;   // sin(c_L) ^ 2;
                o *= param.negOffsetFactor;  //-(cos(c_L) - Float(1.0));
            }
        } else {
            s2 = param.L * param.L;
            o = Float(0.5) * offsetBuff(i) * param.L * param.L;
        }
        eigenBuff(i) *= (s2);
        if (eigenBuff(i) > Float(1e-10)) {
            eigenBuff(i) = Float(1.0) / eigenBuff(i);
        } else {
            eigenBuff(i) = Float(0.0);
        }
        offsetBuff(i) = o;

        postInvCovEigenvalues(i) = eigenBuff(i) + invSigmaSqScal;
    }

    gaussian.invCov.noalias() = hEigenvector * postInvCovEigenvalues.asDiagonal() * hEigenvector.transpose();
    gaussian.mean.noalias() = hEigenvector * (eigenBuff.cwiseQuotient(postInvCovEigenvalues).asDiagonal() * offsetBuff);
    gaussian.covL.noalias() = hEigenvector * postInvCovEigenvalues.cwiseInverse().cwiseSqrt().asDiagonal();
    gaussian.logDet = Float(0.0);

    // Here, there was a mistake in the orignal code, dim was used and could potentially be set to -1...                                                                                           // log of determinant of covariance
    for (int i = 0; i < dimension; i++) {
        gaussian.logDet += log(postInvCovEigenvalues(i));
    }
    gaussian.maxAniRatio = postInvCovEigenvalues.cwiseInverse().maxCoeff()/postInvCovEigenvalues.cwiseInverse().minCoeff();
}

H2MCIntegrator::H2MCIntegrator(const Scene *scene, const std::shared_ptr<const PathFuncLib> pathFuncLib) : Integrator(scene, pathFuncLib) {
}

void H2MCIntegrator::Render() {
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
    std::cout << "Average brightness:" << avgScore << std::endl;
    const Float normalization = avgScore;

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

    ProgressReporter reporter(totalSamples);
    const int reportInterval = 16384;
    const int reportIntervalStats = 163840;
    int intervalImgId = 1;

    // Intermediate outputs
    auto integrator = scene->options->integrator;
    fs::path partialDir = fs::path(scene->outputName).parent_path() / fs::path(integrator + "_" + "partial");
    if(output_name != "") {
        // If a output name is provided, we rewrite the partial dir
        partialDir = fs::path(scene->outputName).parent_path() / fs::path(output_name + "_partial");
    }
    if (!fs::exists(partialDir))
        fs::create_directory(partialDir);

	int offsetRandom = 0;
    if(!deterministic) {
        std::random_device rd;
        offsetRandom =std::uniform_int_distribution<int>()(rd);
        std::cout << "Random offset: " << offsetRandom << "\n";
    }
    
    SampleBuffer indirectBuffer(pixelWidth, pixelHeight);
    Timer timer;
    Tick(timer);
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

        smallStep = std::unique_ptr<Mutation>(new H2MCSmallStep(scene, pathFuncLib->maxDepth, scene->options->perturbStdDev, scene->options->lensPerturbStdDev));

        for (int sampleIdx = 0; sampleIdx < numSamplesThisChain; sampleIdx++) {
            Float a = Float(1.0);
            bool isLargeStep = false;
            if (!currentState.valid || uniDist(rng) < largeStepProb) {
                isLargeStep = true;
                a = largeStep->Mutate(mltState, normalization, currentState, proposalState, rng);
            } else {
                a = smallStep->Mutate(mltState, normalization, currentState, proposalState, rng);
            }
            if (currentState.valid && a < Float(1.0)) {
                for (const auto splat : currentState.toSplat) {
                    Splat(indirectBuffer, splat.screenPos, (Float(1.0) - a) * splat.contrib);
                }
            }
            if (a > Float(0.0)) {
                for (const auto splat : proposalState.toSplat) {
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
    reporter.Done();
    Float elapsed = Tick(timer);
    std::cout << "Elapsed time:" << elapsed << std::endl;

    SampleBuffer buffer(pixelWidth, pixelHeight);
    Float directWeight = scene->options->directSpp > 0 ? inverse(Float(scene->options->directSpp)) : Float(0.0);
    Float indirectWeight = spp > 0 ? inverse(Float(spp)) : Float(0.0);
    MergeBuffer(directBuffer, directWeight, indirectBuffer, indirectWeight, buffer);
    BufferToFilm(buffer, film.get());
}

Float H2MCIntegrator::PrePass(const MLTState &mltState, const int64_t numInitSamples, const int numChains, MarkovStates &initStates, std::shared_ptr<PiecewiseConstant1D> &lengthDist)  {
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
    Float totalScore(Float(0.0));
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

H2MCSmallStep::H2MCSmallStep(const Scene *scene, const int maxDervDepth, const Float sigma, const Float lensSigma)
        : param(sigma), lensParam(lensSigma) {
    sceneParams.resize(GetSceneSerializedSize());
    Serialize(scene, &sceneParams[0]);
    ssubPath.primary.resize(GetPrimaryParamSize(maxDervDepth, maxDervDepth));
    ssubPath.vertParams.resize(GetVertParamSize(maxDervDepth, maxDervDepth));
}

Float H2MCSmallStep::Mutate(const MLTState &mltState, const Float normalization, MarkovState &currentState, MarkovState &proposalState, RNG &rng) {
    const Scene *scene = mltState.scene;
    // Sometimes the derivatives are noisy so that the light paths
    // will "stuck" in some regions, we probabilistically switch to
    // uniform sampling to avoid stucking
    std::uniform_real_distribution<Float> uniDist(Float(0.0), Float(1.0));
    if (uniDist(rng) < scene->options->uniformMixingProbability) {
        Float a =
                isotropicSmallStep.Mutate(mltState, normalization, currentState, proposalState, rng);
        lastMutationType = isotropicSmallStep.lastMutationType;
        return a;
    }

    spContribs.clear();

    const Float lensPerturbProb = scene->options->lensPerturbProb;
    Float a = Float(1.0);
    assert(currentState.valid);
    if (currentState.spContrib.lensScore > Float(0.0) && uniDist(rng) < lensPerturbProb) {
        lastMutationType = MutationType::Lens;
        auto initLensGaussian = [&](MarkovState &state) {
            const SubpathContrib &cspContrib = state.spContrib;
            auto funcIt =
                    mltState.lensFuncDervMap.find({cspContrib.camDepth, cspContrib.lightDepth});
            if (funcIt != mltState.lensFuncDervMap.end()) {
                vGrad.resize(2, Float(0.0));
                vHess.resize(2 * 2, Float(0.0));
                if (cspContrib.lensScore > Float(1e-15)) {
                    PathFuncDerv dervFunc = funcIt->second;
                    Serialize(scene, state.path, ssubPath);
                    dervFunc(&cspContrib.screenPos[0],
                            &ssubPath.primary[0],
                            &sceneParams[0],
                            &ssubPath.vertParams[0],
                            &vGrad[0],
                            &vHess[0]);
                    if (!IsFinite(vGrad) || !IsFinite(vHess)) {
                        // Usually caused by floating point round-off error
                        // (or, of course, bugs)
                        std::fill(vGrad.begin(), vGrad.end(), Float(0.0));
                        std::fill(vHess.begin(), vHess.end(), Float(0.0));
                    }

                    assert(IsFinite(vGrad));
                    assert(IsFinite(vHess));
                }
                ComputeGaussian(lensParam, cspContrib.ssScore, vGrad, vHess, state.lensGaussian);
            } else {
                IsotropicGaussian(2, lensParam.sigma, state.lensGaussian);
            }
            state.lensGaussianInitialized = true;
        };

        if (!currentState.lensGaussianInitialized) {
            initLensGaussian(currentState);
        }

        assert(currentState.lensGaussianInitialized);

        Vector offset(2);
        GenerateSample(currentState.lensGaussian, offset, rng);
        Vector screenPos(2);
        screenPos[0] = Modulo(currentState.spContrib.screenPos[0] + offset[0], Float(1.0));
        screenPos[1] = Modulo(currentState.spContrib.screenPos[1] + offset[1], Float(1.0));
        proposalState.path = currentState.path;
        const auto perturbLensFunc = mltState.perturbLensFunc;
        perturbLensFunc(scene, screenPos, proposalState.path, spContribs);
        proposalState.gaussianInitialized = false;
        if (spContribs.size() > 0) {
            assert(spContribs.size() == 1);
            proposalState.spContrib = spContribs[0];
            initLensGaussian(proposalState);
            const Float py = GaussianLogPdf(offset, currentState.lensGaussian);
            const Float px = GaussianLogPdf(-offset, proposalState.lensGaussian);
            a = Clamp(expf(px - py) * proposalState.spContrib.lensScore /
                        currentState.spContrib.lensScore,
                        Float(0.0),
                        Float(1.0));
            proposalState.toSplat.clear();
            for (const auto &spContrib : spContribs) {
                proposalState.toSplat.push_back(SplatSample{spContrib.screenPos, spContrib.contrib * (normalization / spContrib.lsScore)});
            }
        } else {
            a = Float(0.0);
        }
    } else {
        lastMutationType = MutationType::Small;
        const auto perturbPathFunc = mltState.perturbPathFunc;
        const int dim = GetDimension(currentState.path);
        auto initGaussian = [&](MarkovState &state) {
            const SubpathContrib &cspContrib = state.spContrib;
            const auto &fmap =
                    state.path.isMoving ? mltState.funcDervMap : mltState.staticFuncDervMap;
            auto funcIt = fmap.find({cspContrib.camDepth, cspContrib.lightDepth});
            const int dim = GetDimension(state.path);
            if (funcIt != fmap.end()) {
                vGrad.resize(dim, Float(0.0));
                vHess.resize(dim * dim, Float(0.0));
                if (cspContrib.ssScore > Float(1e-15)) {
                    PathFuncDerv dervFunc = funcIt->second;
                    assert(dervFunc != nullptr);
                    Serialize(scene, state.path, ssubPath);
                    dervFunc(&cspContrib.screenPos[0],
                            &ssubPath.primary[0],
                            &sceneParams[0],
                            &ssubPath.vertParams[0],
                            &vGrad[0],
                            &vHess[0]);
                    if (!IsFinite(vGrad) || !IsFinite(vHess)) {
                        // Usually caused by floating point round-off error
                        // (or, of course, bugs)
                        std::fill(vGrad.begin(), vGrad.end(), Float(0.0));
                        std::fill(vHess.begin(), vHess.end(), Float(0.0));
                    }
                    assert(IsFinite(vGrad));
                    assert(IsFinite(vHess));
                }
                ComputeGaussian(param, cspContrib.ssScore, vGrad, vHess, state.gaussian);
            } else {
                IsotropicGaussian(dim, param.sigma, state.gaussian);
            }
            state.gaussianInitialized = true;
        };

        if (!currentState.gaussianInitialized) {
            initGaussian(currentState);
        }

        assert(currentState.gaussianInitialized);

        Vector offset(dim);
        GenerateSample(currentState.gaussian, offset, rng);
        proposalState.path = currentState.path;
        perturbPathFunc(scene, offset, proposalState.path, spContribs, rng);

        proposalState.lensGaussianInitialized = false;
        if (spContribs.size() > 0) {
            assert(spContribs.size() == 1);
            proposalState.spContrib = spContribs[0];
            initGaussian(proposalState);
            Float py = GaussianLogPdf(offset, currentState.gaussian);
            Float px = GaussianLogPdf(-offset, proposalState.gaussian);
            a = Clamp(std::exp(px - py) * proposalState.spContrib.ssScore /
                        currentState.spContrib.ssScore,
                        Float(0.0),
                        Float(1.0));
            proposalState.toSplat.clear();
            for (const auto &spContrib : spContribs) {
                proposalState.toSplat.push_back(SplatSample{
                        spContrib.screenPos, spContrib.contrib * (normalization / spContrib.lsScore)});
            }
        } else {
            a = Float(0.0);
        }
    }

    return a;
}

void ComputeGaussian(const H2MCIntegrator::H2MCParam &param, const Float sc, const AlignedStdVector &vGrad, const AlignedStdVector &vHess, Gaussian &gaussian) {
    int dim = int(vGrad.size());
    Eigen::Map<const Vector, Eigen::Aligned> grad(&vGrad[0], dim);
    Eigen::Map<const Matrix, Eigen::Aligned> hess(&vHess[0], dim, dim);

    Vector sigma = Vector::Constant(dim, param.sigma);
    Float sigmaMax = sigma.maxCoeff();
    auto sigmaSq = sigma.cwiseProduct(sigma);
    Vector invSigmaSq = sigmaSq.cwiseInverse();
    if (sc <= Float(1e-15) || hess.norm() < Float(0.5) / (sigmaMax * sigmaMax)) {
        gaussian.mean = Vector::Zero(dim);
        gaussian.covL = sigma.asDiagonal();
        gaussian.invCov = invSigmaSq.asDiagonal();
        gaussian.logDet = Float(0.0);
        for (int i = 0; i < dim; i++) {
            gaussian.logDet += log(invSigmaSq[i]);
        }
    } else {
#if DYN_GAUSSIAN
        ComputeGaussian<-1>(param, grad, hess, invSigmaSq, gaussian);
#else
        switch (dim) {
            H2MC_COMPUTEGAUSSIAN(2)
            H2MC_COMPUTEGAUSSIAN(3)
            H2MC_COMPUTEGAUSSIAN(4)
            H2MC_COMPUTEGAUSSIAN(5)
            H2MC_COMPUTEGAUSSIAN(6)
            H2MC_COMPUTEGAUSSIAN(7)
            H2MC_COMPUTEGAUSSIAN(8)
            H2MC_COMPUTEGAUSSIAN(9)
            H2MC_COMPUTEGAUSSIAN(10)
            H2MC_COMPUTEGAUSSIAN(11)
            H2MC_COMPUTEGAUSSIAN(12)
            default: { ComputeGaussian<-1>(param, grad, hess, invSigmaSq, gaussian); }
        }
#endif
    }
}
