#pragma once

#include "commondef.h"
#include <algorithm>


/**
 * Gaussian Kernel. This struct contains the nessessary quantities to
 * sample and evaluate a multidimensional anisotropic gaussian
 * distribution
*/
struct Gaussian {
    Matrix covL;        // Cholesky decomposition from the Covariance, to sample
    Matrix invCov;      // Inverse of Covariance, to evaluate the pdf
    Vector mean;        // Mean
    Float logDet;       // Logarithm of the determinant of the Covariance, to evaluate the pdf
    bool isIso;         // Flag to si if isotropic
    Float maxEigen;     // largest eigenvalue of Covariance
    Float maxAniRatio;  // Ratio of largest and smallest eigenvalue. give an Idea about the anisotropy
};

/**
 * Get dimension of the gaussian
*/
inline int GetDimension(const Gaussian &gaussian) {
    return gaussian.mean.size();
}

/**
 * Isotropic Gaussian constructor
*/
void IsotropicGaussian(const int dim, const Float sigma, Gaussian &gaussian);

/**
 * Return log_pdf of gaussian given offset, Vector version
*/
Float GaussianLogPdf(const Vector &offset, const Gaussian &gaussian);

/**
 * Return log_pdf of gaussian given offset, AlignedStdVector version
*/
Float GaussianLogPdf(const AlignedStdVector &offset, const Gaussian &gaussian);

/**
 * Return sample from gaussian centered at x
*/
void GenerateSample(const Gaussian &gaussian, Vector &x, RNG &rng);
