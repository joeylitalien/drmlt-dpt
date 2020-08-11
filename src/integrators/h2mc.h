#pragma once

#include "commondef.h"
#include "gaussian.h"
#include "alignedallocator.h"
#include "utils.h"
#include "integrator.h"
#include <vector>

class H2MCIntegrator : public Integrator{
public:
    struct H2MCParam {
        H2MCParam(const Float sigma = 0.01, const Float L = Float(M_PI / 2.0)) : sigma(sigma), L(L) {
            posScaleFactor = Float(0.5) * (exp(L) - exp(-L)) * Float(0.5) * (exp(L) - exp(-L));
            posOffsetFactor = Float(0.5) * (exp(L) + exp(-L) - Float(1.0));
            negScaleFactor = sin(L) * sin(L);
            negOffsetFactor = -(cos(L) - Float(1.0)); //1
        }

        Float sigma;
        Float posScaleFactor; // sinh(L)^2 = 5.3
        Float posOffsetFactor;
        Float negScaleFactor; // sin(L)^2 = 1
        Float negOffsetFactor;
        Float L; // L^2 = 2.46
    };

    H2MCIntegrator(const Scene *scene, const std::shared_ptr<const PathFuncLib> pathFuncLib);
    void Render() override;
    Float PrePass(const MLTState &mltState, const int64_t numInitSamples, const int numChains, MarkovStates &initStates, std::shared_ptr<PiecewiseConstant1D> &lengthDist) override;
};

struct H2MCSmallStep : public Mutation {
    H2MCSmallStep(const Scene *scene, const int maxDervDepth, const Float sigma, const Float lensSigma);
    Float Mutate(const MLTState &mltState, const Float normalization, MarkovState &currentState, MarkovState &proposalState, RNG &rng) override;

    std::vector<SubpathContrib> spContribs;
    H2MCIntegrator::H2MCParam param, lensParam;
    AlignedStdVector sceneParams;
    SerializedSubpath ssubPath;
    SmallStep isotropicSmallStep;

    AlignedStdVector vGrad;
    AlignedStdVector vHess;
};

void ComputeGaussian(const H2MCIntegrator::H2MCParam &param, const Float sc, const AlignedStdVector &vGrad, const AlignedStdVector &vHess, Gaussian &gaussian);