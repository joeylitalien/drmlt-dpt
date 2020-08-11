#include "drsmallstep.h"


// Shorthand for templating
#define PH2MC_COMPUTEGAUSSIAN(S_) \
    case S_: {\
    ComputeGaussian<S_>(param, grad, hess, gaussian);\
    break;\
    }

/**
 * Templated version. Compute the H2MC anisotropic Gaussian.
 * To do so the Hessian is diagonalized, its eigenvalues are 
 * bounded and inversed to construct the covariance matrix.
 * See Li & al 2015 (H2MC) and the original DPT for further 
 * references
 */
template<int dim>
void PH2MCSmallStep::ComputeGaussian(const PH2MCSmallStep::Param &param,
                                    const Eigen::Matrix<Float, dim, 1> &grad,
                                    const Eigen::Matrix<Float, dim, dim> &hess,
                                    Gaussian &gaussian) {
    
    // Diagonalize the Hessian using an adjoint eigensolver
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Float, dim, dim>> eigenSolver;
    eigenSolver.compute(hess);

    // Set array buffer for eigen computation
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

    // Invert
    for (int i = 0; i < dimension; i++) {
        if (fabs(hEigenvalues(i)) > Float(1e-10)) {
            eigenBuff(i) = 1.0 / fabs(hEigenvalues(i));
        } else {
            eigenBuff(i) = 0.0;
        }
    }

    // Good old H2MC formulas
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
        // add prior to bound covariance
        postInvCovEigenvalues(i) = eigenBuff(i) + invSigmaSqScal;
    }

    // Compute useful Gaussian quantities for sampling and pdf
    gaussian.invCov.noalias() = hEigenvector * postInvCovEigenvalues.asDiagonal() * hEigenvector.transpose();                // inverse of covariance
    gaussian.mean.noalias() = hEigenvector * (eigenBuff.cwiseQuotient(postInvCovEigenvalues).asDiagonal() * offsetBuff);     // drifted mean
    gaussian.covL.noalias() = hEigenvector * postInvCovEigenvalues.cwiseInverse().cwiseSqrt().asDiagonal();                  // Cholesky decomposition of covariance
    gaussian.logDet = Float(0.0); 
    
    // Here, there was a mistake in the orignal code, dim was used and could potentially be set to -1...                                                                                           // log of determinant of covariance
    for (int i = 0; i < dimension; i++) {
        gaussian.logDet += log(postInvCovEigenvalues(i));
    }
    gaussian.maxAniRatio = postInvCovEigenvalues.cwiseInverse().maxCoeff()/postInvCovEigenvalues.cwiseInverse().minCoeff();  //anisotropy ratio
}

/**
 * Small step constructor
 */
PH2MCSmallStep::PH2MCSmallStep(const Scene *scene,
                                const int maxDervDepth,
                                const Float sigma)
    : param(sigma) {
    sceneParams.resize(GetSceneSerializedSize());
    Serialize(scene, &sceneParams[0]);
    ssubPath.primary.resize(GetPrimaryParamSize(maxDervDepth, maxDervDepth));
    ssubPath.vertParams.resize(GetVertParamSize(maxDervDepth, maxDervDepth));
}

/**
 * Propose new state, return the usual MH acceptance ratio
 * For simplicity, we removed some feature of the original
 * DPT implementation such as motion blur and lens mutation
 */
Float PH2MCSmallStep::Mutate(const MLTState &mltState,
                            const Float normalization,
                            MarkovState &currentState,
                            MarkovState &proposalState,
                            RNG &rng) {

    const Scene *scene = mltState.scene;
    spContribs.clear();
    Float a = Float(1.0);

    assert(currentState.valid);
    lastMutationType = MutationType::Small;
    const auto perturbPathFunc = mltState.perturbPathFunc;
    const int dim = GetDimension(currentState.path);

    // Lambda to compute the gradient, Hessian and Gaussian of the state. 
    auto initGaussian = [&](MarkovState &state) {
        const SubpathContrib &cspContrib = state.spContrib;
        const auto &fmap =
                state.path.isMoving ? mltState.funcDervMap : mltState.staticFuncDervMap;
        auto funcIt = fmap.find({cspContrib.camDepth, cspContrib.lightDepth});
        const int dim = GetDimension(state.path);
        if (funcIt != fmap.end()) {
            // Compute Hessian and Gradient
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
            // Compute the associated Gaussian
            ComputeGaussian(param, cspContrib.ssScore, vGrad, vHess, state.gaussian);
        } else {
            // if no AD, compute iso
            IsotropicGaussian(dim, param.sigma, state.gaussian);
        }
        // set this flag to not recompute it uppon acceptance
        state.gaussianInitialized = true;
    };

    // If current state has never computed its anisotropic gaussian, do so.
    if (!currentState.gaussianInitialized) {
        initGaussian(currentState);
    }
    assert(currentState.gaussianInitialized);

    // Do the mutation
    Vector offset(dim);
    GenerateSample(currentState.gaussian, offset, rng);
    proposalState.path = currentState.path;
    perturbPathFunc(scene, offset, proposalState.path, spContribs, rng);

    // If zero contribution, reject automatically
    if (spContribs.size() > 0) {
        assert(spContribs.size() == 1);

        // Set contribution
        proposalState.spContrib = spContribs[0];

        // Compute second stage Gaussian for MH kernel ratio
        initGaussian(proposalState);

        // Compute Kernel Ratio
        Float py = GaussianLogPdf(offset, currentState.gaussian);
        Float px = GaussianLogPdf(-offset, proposalState.gaussian);

        // Usual MH acceptance 
        a = std::exp(px - py) * proposalState.spContrib.ssScore /
            currentState.spContrib.ssScore;
        
        // Add to splatlist
        proposalState.toSplat.clear();
        for (const auto &spContrib : spContribs) {
            if (spContrib.lsScore > 0.0f && std::isfinite(spContrib.lsScore)){
                proposalState.toSplat.push_back(SplatSample{
                        spContrib.screenPos, spContrib.contrib * (normalization / spContrib.lsScore)});
            }
        }
    } else {
        a = Float(0.0);
    }

    // return MH acceeptance
    return a;
}

/**
 * Compute anisotropic Gaussian proposal from gradient and Hessian.
 * Depending on the value it is either approximated by 
 * the bounding isotropic prior or diagonalized.
 */
void PH2MCSmallStep::ComputeGaussian(const PH2MCSmallStep::Param &param,
                                    const Float sc,
                                    const AlignedStdVector &vGrad,
                                    const AlignedStdVector &vHess,
                                    Gaussian &gaussian) {
    int dim = int(vGrad.size());
    
    // align in memory for eigen
    Eigen::Map<const Vector, Eigen::Aligned> grad(&vGrad[0], dim, 1);
    Eigen::Map<const Matrix, Eigen::Aligned> hess(&vHess[0], dim, dim);

    // bounding prior variance
    Vector sigma = Vector::Constant(dim, param.sigma);
    Float sigmaMax = sigma.maxCoeff();
    auto sigmaSq = sigma.cwiseProduct(sigma);
    Vector invSigmaSq = sigmaSq.cwiseInverse();

    // trick from Li & al, (2015) to fall back to iso when resulting Kernel would be larger than prior
    if (sc <= Float(1e-15) || hess.norm() < Float(0.5) / (sigmaMax * sigmaMax)) {
        gaussian.mean = Vector::Zero(dim);
        gaussian.covL = sigma.asDiagonal();
        gaussian.invCov = invSigmaSq.asDiagonal();
        gaussian.logDet = Float(0.0);
        for (int i = 0; i < dim; i++) {
            gaussian.logDet += log(invSigmaSq[i]);
        }
    }
    // Otherwise, diagonalize and sample from the anisotropic gaussian
    else {
#if DYN_GAUSSIAN
        ComputeGaussian<-1>(param, grad, hess, gaussian);
#else
        switch (dim) {
            PH2MC_COMPUTEGAUSSIAN(2)
            PH2MC_COMPUTEGAUSSIAN(3)
            PH2MC_COMPUTEGAUSSIAN(4)
            PH2MC_COMPUTEGAUSSIAN(5)
            PH2MC_COMPUTEGAUSSIAN(6)
            PH2MC_COMPUTEGAUSSIAN(7)
            PH2MC_COMPUTEGAUSSIAN(8)
            PH2MC_COMPUTEGAUSSIAN(9)
            PH2MC_COMPUTEGAUSSIAN(10)
            PH2MC_COMPUTEGAUSSIAN(11)
            PH2MC_COMPUTEGAUSSIAN(12)
            default: { ComputeGaussian<-1>(param, grad, hess, gaussian); }
        }
#endif
    }
}