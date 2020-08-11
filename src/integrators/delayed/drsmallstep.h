#pragma once

#include "commondef.h"
#include "gaussian.h"
#include "alignedallocator.h"
#include "utils.h"
#include <vector>
#include <list>
#include "mutation.h"

/**
 * Pure H2MC small step mutation
 */
struct PH2MCSmallStep : public SmallStep {
    /// Constructor
    PH2MCSmallStep(const Scene *scene, int maxDervDepth, Float sigma);

    /// Mutation routine
    Float Mutate(const MLTState &mltState,
                 Float normalization,
                 MarkovState &currentState,
                 MarkovState &proposalState,
                 RNG &) override;

    /// Algorithm parameter (see H2MC paper for more info)
    struct Param {
        explicit Param(const Float sigma_ = 0.01,
                       const Float L_ = Float(M_PI / 2.0))
            : sigma(sigma_), L(L_) {
            posScaleFactor = Float(0.5) * (exp(L) - exp(-L)) * Float(0.5) * (exp(L) - exp(-L));
            posOffsetFactor = Float(0.5) * (exp(L) + exp(-L) - Float(1.0));
            negScaleFactor = sin(L) * sin(L);
            negOffsetFactor = -(cos(L) - Float(1.0));
        }

        Float sigma;
        Float posScaleFactor;
        Float posOffsetFactor;
        Float negScaleFactor;
        Float negOffsetFactor;
        Float L;
    };

    std::vector<SubpathContrib> spContribs;
    Param param;
    AlignedStdVector sceneParams;
    SerializedSubpath ssubPath;

    AlignedStdVector vGrad; // Gradient of throughput
    AlignedStdVector vHess; // Hessian of throughput

    /// Check if Hessian is flat; compute Gaussian if necessary
    void ComputeGaussian(const Param &param,
                         const Float sc,
                         const AlignedStdVector &vGrad,
                         const AlignedStdVector &vHess,
                         Gaussian &gaussian);

    /// Diagonalize and sample from anisotropic Gaussian
    template<int dim>
    void ComputeGaussian(const Param &param,
                         const Eigen::Matrix<Float, dim, 1> &grad,
                         const Eigen::Matrix<Float, dim, dim> &hess,
                         Gaussian &gaussian);
};