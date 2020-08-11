#pragma once

#include "commondef.h"
#include "gaussian.h"
#include "alignedallocator.h"
#include "utils.h"
#include "integrator.h"
#include <vector>

class MCMCIntegrator : public Integrator{
public:
    MCMCIntegrator(const Scene *scene, const std::shared_ptr<const PathFuncLib> pathFuncLib);
    void Render() override;
    Float PrePass(const MLTState &mltState, const int64_t numInitSamples, const int numChains, MarkovStates &initStates, std::shared_ptr<PiecewiseConstant1D> &lengthDist) override;
};
