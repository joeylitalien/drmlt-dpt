#pragma once

#include "path.h"
#include "commondef.h"
#include "image.h"
#include "camera.h"
#include "gaussian.h"
#include "progressreporter.h"
#include "parallel.h"
#include "timer.h"
#include "alignedallocator.h"
#include "distribution.h"
#include "alignedallocator.h"


struct PathFuncLib;

enum class MutationType { Large, Small, Lens };

struct MLTState {
    const Scene *scene;
    const decltype(&GeneratePath) genPathFunc;
    const decltype(&PerturbPath) perturbPathFunc;
    const decltype(&PerturbLens) perturbLensFunc;
    PathFuncMap funcMap;
    PathFuncDervMap funcDervMap;
    PathFuncMap staticFuncMap;
    PathFuncDervMap staticFuncDervMap;
    PathFuncMap lensFuncMap;
    PathFuncDervMap lensFuncDervMap;
};

struct SplatSample {
    Vector2 screenPos;
    Vector3 contrib;
};

struct MarkovState {
    MarkovState() {
        valid = false;
        scoreSum = 0.0;
        gaussianInitialized = false;
        lensGaussianInitialized = false;
    }
    bool valid;
    SubpathContrib spContrib;   // contribution of this technique
    Path path;                  // Path (vertex...)
    Float scoreSum;             // sum of many path technique contribution

    bool gaussianInitialized;
    Gaussian gaussian;
    bool lensGaussianInitialized;
    Gaussian lensGaussian;

    MutationType mutationType;

    std::vector<SplatSample> toSplat;
    void setValid(bool valid_) { valid = valid_; }
};


struct MarkovStates {
    std::vector<MarkovState> initStates;
    const MarkovState &at(int i) const {
        assert(i < initStates.size());
        return initStates[i];
    }
    void reserve(int s) {
        initStates.reserve(s);
    }
    MarkovState &add() {
        initStates.push_back(MarkovState());
        return initStates.back();
    }
};

/**
 * Mutation Abstract Class
*/

struct Mutation {
    virtual Float Mutate(const MLTState &mltState,
                         const Float normalization,
                         MarkovState &currentState,
                         MarkovState &proposalState,
                         RNG &rng) { return 0.f; };

    virtual Float Mutate(const MLTState &mltState,
                         const Float normalization,
                         MarkovState &currentState,
                         MarkovState &proposalState,
                         const Float sigma,
                         RNG &rng) { return 0.f; };

    MutationType lastMutationType;
    std::vector<SubpathContrib> spContribsCurrent;
    std::vector<SubpathContrib> spContribs;
};

/**
 * Largestep Mutation Prototype (isotropic gaussian)
*/

struct LargeStep : public Mutation {
    explicit LargeStep(std::shared_ptr<PiecewiseConstant1D> lengthDist) : lengthDist(lengthDist) {}
    Float Mutate(const MLTState &mltState,
                 const Float normalization,
                 MarkovState &currentState,
                 MarkovState &proposalState,
                 RNG &rng) override;

    std::shared_ptr<PiecewiseConstant1D> lengthDist;
    std::vector<Float> contribCdf;
    Float lastScoreSum = Float(1.0);
    Float lastScore = Float(1.0);
};

/**
 * Smallstep Mutation Prototype (isotropic gaussian)
*/

struct SmallStep : public Mutation {
    explicit SmallStep() {}

    /**
     * Mutate from isotropic gaussian
    */
    Float Mutate(const MLTState &mltState,
                 const Float normalization,
                 MarkovState &currentState,
                 MarkovState &proposalState,
                 RNG &rng) override;

    /**
     * Mutate from isotropic gaussian given standard deviation, (second stage mixture)
    */
    Float Mutate(const MLTState &mltState,
                 const Float normalization,
                 MarkovState &currentState,
                 MarkovState &proposalState,
                 const Float sigma,
                 RNG &rng) override;
    /**
     * EGreen, Trace proposalStateStar and compute reverse acceptance given proposalState
    */
    Float Trace(const MLTState &mltState,
                 const Vector &offset,
                 MarkovState &proposalState,
                 MarkovState &proposalStateStar,
                 RNG &rng);
};

