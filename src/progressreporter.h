#pragma once

#include <mutex>

class ProgressReporter {
    public:
    ProgressReporter(uint64_t totalWork) : totalWork(totalWork), workDone(0) {
    }
    void Update(uint64_t num, bool verbose) {
        std::lock_guard<std::mutex> lock(mutex);
        workDone += num;
        Float workRatio = (Float)workDone / (Float)totalWork;
        if(verbose) {
            fprintf(stdout,
                    "\r%.2f %% (%llu / %llu)",
                    workRatio * Float(100.0),
                    (unsigned long long) workDone,
                    (unsigned long long) totalWork);
        }
    }
    void Done() {
        workDone = totalWork;
        fprintf(stdout,
                "\r%.2f %% (%llu / %llu)\n",
                Float(100.0),
                (unsigned long long)workDone,
                (unsigned long long)totalWork);
    }
    uint64_t GetWorkDone() const {
        return workDone;
    }

    private:
    const uint64_t totalWork;
    uint64_t workDone;
    std::mutex mutex;
};
