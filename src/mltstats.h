#pragma once

#include "commondef.h"

struct MLTStats {
    std::atomic<int64_t> smallStepAccepted, smallStepTotal;
    std::atomic<int64_t> secondSmallStepAccepted, secondSmallStepTotal;
    std::atomic<int64_t> largeStepAccepted, largeStepTotal;
    std::atomic<int64_t> lensStepAccepted, lensStepTotal;

    MLTStats() {
        smallStepAccepted = 0, smallStepTotal = 0;
        secondSmallStepAccepted = 0, secondSmallStepTotal = 0;
        largeStepAccepted = 0, largeStepTotal = 0;
        lensStepAccepted = 0, lensStepTotal = 0;
    }

    void ShowReport(const Float elapsed) {
        std::cout << "\nElapsed time: " << elapsed << " s" << std::endl;

        std::cout << "Small step acceptance rate: "
                  << Float(smallStepAccepted)/Float(smallStepTotal) << " ("
                  << int64_t(smallStepAccepted) << " / " << int64_t(smallStepTotal) << ")"
                  << std::endl;

        if (secondSmallStepTotal > 0) {
            std::cout << "2nd stage small step acceptance rate: "
                      << Float(secondSmallStepAccepted) / Float(secondSmallStepTotal) << " ("
                      << int64_t(secondSmallStepAccepted) << " / " << int64_t(secondSmallStepTotal) << ")"
                      << std::endl;
        }

        std::cout << "Large step acceptance rate: " << Float(largeStepAccepted)/Float(largeStepTotal)
                  << " (" << int64_t(largeStepAccepted) << " / " << int64_t(largeStepTotal) << ")"
                  << std::endl;

        if (lensStepTotal > 0) {
            std::cout << "Lens step acceptance rate: " << Float(lensStepAccepted) / Float(lensStepTotal)
                      << " (" << int64_t(lensStepAccepted) << " / " << int64_t(lensStepTotal) << ")"
                      << std::endl;
        }
    }
};