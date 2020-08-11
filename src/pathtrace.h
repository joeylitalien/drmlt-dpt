#pragma once

#include "scene.h"

#include <memory>

struct PathFuncLib;

void PathTrace(const Scene *scene,
        const std::shared_ptr<const PathFuncLib> pathFuncLib,
        const std::string& output_name,
        bool deterministic);
