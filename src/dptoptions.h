#pragma once

#include "commondef.h"

#include <string>

struct DptOptions {

    // General
    std::string integrator = "drmlt";
    bool bidirectional = true;
    int spp = 16;
    int numInitSamples = 100000;
    int minDepth = -1;
    int maxDepth = -1;
    int directSpp = 16;
    Float largeStepProb = Float(0.3);
    Float perturbStdDev = Float(0.01);
    Float roughnessThreshold = Float(0.03);
    Float lensPerturbProb = Float(0.3);
    Float lensPerturbStdDev = Float(0.01);
    int numChains = 1024;
    int seedOffset = 0;
    int reportIntervalSpp = 0;
    int reportIntervalSeconds = 0;
    bool useLightCoordinateSampling = true;
    Float discreteStdDev = Float(0.01);
    bool largeStepMultiplexed = true;
    Float uniformMixingProbability = Float(0.2);
    

    // Delayed Rejection Specific
    std::string type = "green"; 
    Float anisoPerturbProb = Float(0.5);
    Float anisoperturbstddev = Float(0.0025);
    bool acceptanceMap = false;

};

inline std::string GetLibPath() {
    return fs::current_path();
}
