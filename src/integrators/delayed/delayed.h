#pragma once

#include "commondef.h"
#include "gaussian.h"
#include "alignedallocator.h"
#include "utils.h"
#include "integrator.h"
#include <vector>
#include "mltstats.h"
#include "drsmallstep.h"

/**
 * Delayed rejection (DR)
 * Isotropic 1st-stage, mixture of iso and H2MC 2nd-stage
 */
class DRIntegrator : public Integrator {
public:

    /**
     * Specifies the sampling algorithm that is internally used
    */
    enum EType {
        EGreen,   // uses Green & Mira (2001) algorithm
        EMira     // uses Tierney & Mira (1999) algorithm
    };




    /// Constructor
    DRIntegrator(const Scene *scene, std::shared_ptr<const PathFuncLib> pathFuncLib);

    /// Prepass for MLT bootstrap
    Float PrePass(const MLTState &mltState,
                  const int64_t numInitSamples,
                  const int numChains,
                  MarkovStates &initStates,
                  std::shared_ptr<PiecewiseConstant1D> &lengthDist) override;


    /// Main rendering loop
    void Render() override;
};