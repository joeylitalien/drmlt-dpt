#include "mutation.h"


/**
 * Large step mutation
 */
Float LargeStep::Mutate(const MLTState &mltState,
                        const Float normalization,
                        MarkovState &currentState,
                        MarkovState &proposalState,
                        RNG &rng) {

    // Store the mutation type
    proposalState.mutationType = MutationType::Large;
    lastMutationType = MutationType::Large;

    std::uniform_real_distribution<Float> uniDist(Float(0.0), Float(1.0));
    const Scene *scene = mltState.scene;

    // Pointer to the function used to generate a path (same as trace)
    const auto genPathFunc = mltState.genPathFunc;

    auto a = Float(0.0);
    spContribs.clear();
    proposalState.path.clear();

    // Should we perform multiplexed mutation?
    if (scene->options->largeStepMultiplexed) {
        // Select path length
        int length = lengthDist->SampleDiscrete(uniDist(rng), nullptr);

        // Light path length: if bidir [0, length]; if pathtracing [0, 1]
        int lgtLength = scene->options->bidirectional
            ? Clamp(int(uniDist(rng) * (length + 1)), 0, length)
            : Clamp(int(uniDist(rng) * 2), 0, 1);
        int camLength = length - lgtLength + 1;

        // Construct new full path
        GenerateSubpath(scene,
                        Vector2i(-1, -1),
                        camLength,
                        lgtLength,
                        scene->options->bidirectional,
                        proposalState.path,
                        spContribs,
                        rng);

        // Multiplexed only allows for one strategy
        assert(spContribs.size() <= 1);
    } else {
        // Otherwise we want all techniques
        genPathFunc(scene,
                    Vector2i(-1, -1),
                    std::max(scene->options->minDepth, 3),
                    scene->options->maxDepth,
                    proposalState.path,
                    spContribs,
                    rng);
    }

    proposalState.gaussianInitialized = false;
    proposalState.lensGaussianInitialized = false;

    if (!spContribs.empty()) {
        // Create a CDF of all contribs and normalize it
        contribCdf.clear();
        contribCdf.push_back(Float(0.0));
        for (const auto &spContrib : spContribs) {
            contribCdf.push_back(contribCdf.back() + spContrib.lsScore);
        }
        const Float scoreSum = contribCdf.back();
        const Float invSc = inverse(scoreSum);
        std::for_each(contribCdf.begin(), contribCdf.end(), [invSc](Float &cdf) { cdf *= invSc; });

        // Randomly select a contribution
        const auto it = std::upper_bound(contribCdf.begin(), contribCdf.end(), uniDist(rng));
        int64_t contribId = Clamp(int64_t(it - contribCdf.begin() - 1), int64_t(0), int64_t(spContribs.size() - 1));

        // Copy selected contribution into proposal contribution
        proposalState.spContrib = spContribs[contribId];

        // Copy also sum of all contributions
        proposalState.scoreSum = scoreSum;

        if (currentState.valid) {
            if (scene->options->largeStepMultiplexed) {
                int currentLength = GetPathLength(currentState.spContrib.camDepth, currentState.spContrib.lightDepth);
                int proposalLength =
                    GetPathLength(proposalState.spContrib.camDepth, proposalState.spContrib.lightDepth);
                Float invProposalTechniquesPmf =
                    scene->options->bidirectional ? (Float(proposalLength) + Float(1.0)) : Float(2.0);
                Float invCurrentTechniquesPmf =
                    scene->options->bidirectional ? (Float(currentLength) + Float(1.0)) : Float(2.0);
                a = Clamp(
                    (invProposalTechniquesPmf * proposalState.spContrib.lsScore / lengthDist->Pmf(proposalLength))
                        /
                            (invCurrentTechniquesPmf * currentState.spContrib.lsScore / lengthDist->Pmf(currentLength)),
                    Float(0.0), Float(1.0));
            } else {
                // In general, we do not have the "scoreSum" of currentState, since small steps only mutate one subpath
                // To address this, we introduce an augmented space that only contains large step states
                const Float probProposal = (proposalState.spContrib.lsScore / proposalState.scoreSum);

                const auto probLast = lastScore / lastScoreSum;

                a = Clamp((proposalState.spContrib.lsScore * probLast) / (currentState.spContrib.lsScore * probProposal), Float(0.0), Float(1.0));
            }
        }

        // Push all contributions to proposals splat list
        proposalState.toSplat.clear();
        for (const auto &spContrib : spContribs) {
            proposalState.toSplat.push_back(SplatSample{spContrib.screenPos,
                                                        spContrib.contrib * (normalization / scoreSum)});
        }

    }

    return a;
}


/**
 * Small step mutation
 */
Float SmallStep::Mutate(const MLTState &mltState,
                        const Float normalization,
                        MarkovState &currentState,
                        MarkovState &proposalState,
                        RNG &rng) {
    const Scene *scene = mltState.scene;
    spContribs.clear();

    auto a = Float(0.0);
    assert(currentState.valid);
    proposalState.path = currentState.path;
    std::uniform_real_distribution<Float> uniDist(Float(0.0), Float(1.0));

    const Float stdDev = scene->options->perturbStdDev;
    std::normal_distribution<Float> normDist(Float(0.0), stdDev);
    proposalState.mutationType = MutationType::Small;
    lastMutationType = MutationType::Small;
    const auto perturbPathFunc = mltState.perturbPathFunc;
    Vector offset(GetDimension(currentState.path));
    for (int i = 0; i < offset.size(); i++) {
        offset[i] = normDist(rng);
    }
    perturbPathFunc(scene, offset, proposalState.path, spContribs, rng);
    proposalState.path.SetIsMoving();
    proposalState.gaussianInitialized = false;

    if (!spContribs.empty()) {
        assert(spContribs.size() == 1);
        proposalState.spContrib = spContribs[0];
        a = Clamp(proposalState.spContrib.ssScore / currentState.spContrib.ssScore, Float(0.0), Float(1.0));
        proposalState.toSplat.clear();

        for (const auto &spContrib : spContribs) {
            proposalState.toSplat.push_back(
                SplatSample{spContrib.screenPos, spContrib.contrib * (normalization / spContrib.lsScore)}
            );
        }
    }

    return a;
}
Float SmallStep::Mutate(const MLTState &mltState,
                        const Float normalization,
                        MarkovState &currentState,
                        MarkovState &proposalState,
                        const Float sigma,
                        RNG &rng) {
    const Scene *scene = mltState.scene;
    spContribs.clear();

    auto a = Float(0.0);
    assert(currentState.valid);
    proposalState.path = currentState.path;
    std::uniform_real_distribution<Float> uniDist(Float(0.0), Float(1.0));

    const Float stdDev = sigma;
    std::normal_distribution<Float> normDist(Float(0.0), stdDev);
    proposalState.mutationType = MutationType::Small;
    lastMutationType = MutationType::Small;
    const auto perturbPathFunc = mltState.perturbPathFunc;
    Vector offset(GetDimension(currentState.path));
    for (int i = 0; i < offset.size(); i++) {
        offset[i] = normDist(rng);
    }
    perturbPathFunc(scene, offset, proposalState.path, spContribs, rng);
    proposalState.path.SetIsMoving();
    proposalState.gaussianInitialized = false;

    if (!spContribs.empty()) {
        assert(spContribs.size() == 1);
        proposalState.spContrib = spContribs[0];
        a = Clamp(proposalState.spContrib.ssScore / currentState.spContrib.ssScore, Float(0.0), Float(1.0));
        proposalState.toSplat.clear();

        for (const auto &spContrib : spContribs) {
            proposalState.toSplat.push_back(
                    SplatSample{spContrib.screenPos, spContrib.contrib * (normalization / spContrib.lsScore)}
            );
        }
    }

    return a;
}

Float SmallStep::Trace(const MLTState &mltState,
                        const Vector &offset,
                        MarkovState &proposalState,
                        MarkovState &proposalStateStar,
                        RNG &rng) {
    const Scene *scene = mltState.scene;
    spContribs.clear();

    auto a = Float(0.0);
    if(!proposalState.valid) {
        return 0.0;
    }
    proposalStateStar.path = proposalState.path;

    proposalStateStar.mutationType = MutationType::Small;
    lastMutationType = MutationType::Small;
    const auto perturbPathFunc = mltState.perturbPathFunc;
    assert(offset.size() == GetDimension(proposalStateStar.path));
    perturbPathFunc(scene, offset, proposalStateStar.path, spContribs, rng);
    proposalStateStar.path.SetIsMoving();
    proposalStateStar.gaussianInitialized = false;
    if (!spContribs.empty()) {
        assert(spContribs.size() == 1);
        proposalStateStar.spContrib = spContribs[0];
        a = Clamp(proposalStateStar.spContrib.ssScore / proposalState.spContrib.ssScore, Float(0.0), Float(1.0));
        proposalStateStar.toSplat.clear();
    }
    return a;
}