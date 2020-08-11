#include "args.h"
#include "parsescene.h"
#include "pathtrace.h"
#include "integrators/h2mc.h"
#include "integrators/mcmc.h"
#include "integrators/delayed/delayed.h"
#include "image.h"
#include "camera.h"
#include "texturesystem.h"
#include "parallel.h"
#include "path.h"
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <thread>

void DptInit() {
    TextureSystem::Init();
}

void DptCleanup() {
    TextureSystem::Destroy();
    TerminateWorkerThreads();
}

int main(int argc, char *argv[]) {
    args::ArgumentParser parser("DPT with super powers", "");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> output(parser, "output", "Output name", {'o'});
    args::Positional<std::string> scene(parser, "scene", "scene file");
    args::ValueFlag<int> flagnumberThreads(parser, "numberThreads", "number of threads", {"p"});

    // Auto-diff configuration
    args::Flag flagcompilePathLib(parser, "compilePathLib", "force compile path lib", {"compile-pathlib"});
    args::Flag flagcompileBidirPathLib(parser, "compileBidirPath", "force compile path lib (bdpt)", {"compile-bidirpathlib"});
    args::ValueFlag<int> maxDerivativesDepth(parser, "maxDerivativesDepth", "set maximum depth chad is applied", {"max-derivatives-depth"});

    // Debug options
    args::Flag flagNoMT(parser, "noMT", "disabling multi-threading", {"nomt"});
    args::Flag flagNoProgress(parser, "noProgress", "disabling progress bar", {"z"});
    args::Flag flagDeterministic(parser, "deterministic", "enable deterministic sampling", {'d'});

    // Look to exterior variable
    try {
        DptInit();

        // Parsing the command
        try
        {
            parser.ParseCLI(argc, argv);
        }
        catch (args::Help)
        {
            std::cout << parser;
            return 0;
        }
        catch (args::ParseError e)
        {
            std::cerr << e.what() << std::endl;
            std::cerr << parser;
            return 1;
        }
        catch (args::ValidationError e)
        {
            std::cerr << e.what() << std::endl;
            std::cerr << parser;
            return 1;
        }

        // Default options
        bool compilePathLib = false;
        bool compileBidirPathLib = false;
        bool hessian = true;
        bool noProgressBar = false;
        int maxDervDepth = 9;
        bool deterministic = false;
        numberThreads = 0; // Automatic
        std::vector<std::string> filenames;
        std::string output_name = "";

        // Reading the users args values
        if (output) { output_name = args::get(output); std::cout << "output: " << output_name << "\n"; }
        if (scene) { filenames.push_back(args::get(scene)); }
        if (flagcompilePathLib) { compilePathLib = true; }
        if (flagcompileBidirPathLib) { compileBidirPathLib = true; }
        if (maxDerivativesDepth) { maxDervDepth = args::get(maxDerivativesDepth); }
        if (flagnumberThreads) { numberThreads = args::get(flagnumberThreads); }
        if (flagNoProgress) { noProgressBar = true; }
        if (flagDeterministic) { deterministic = true; }

        std::cout << "Running with " << NumSystemCores() << " threads." << std::endl;

        if (compilePathLib) {
            CompilePathFuncLibrary(false, maxDervDepth, hessian);
        }
        if (compileBidirPathLib) {
            CompilePathFuncLibrary(true, maxDervDepth, hessian);
        }
        std::string cwd = getcwd(NULL, 0);
        for (const std::string &filename : filenames) {
            std::unique_ptr<Integrator> integratorPtr;
            std::unique_ptr<Scene> scene = ParseScene(filename);
            std::string integrator = scene->options->integrator;
            scene->options->bidirectional = true;
            if (integrator == "mc") {
                std::shared_ptr<const PathFuncLib> library = BuildPathFuncLibrary(scene->options->bidirectional, maxDervDepth, hessian);
                PathTrace(scene.get(), library, output_name, deterministic);
            } else if (integrator == "mcmc") {
                std::shared_ptr<const PathFuncLib> library = BuildPathFuncLibrary(scene->options->bidirectional, maxDervDepth, hessian);
                integratorPtr = std::unique_ptr<MCMCIntegrator>(new MCMCIntegrator(scene.get(), library));
            } else if (integrator == "h2mc") {
                std::shared_ptr<const PathFuncLib> library = BuildPathFuncLibrary(scene->options->bidirectional, maxDervDepth, hessian);
                integratorPtr = std::unique_ptr<H2MCIntegrator>(new H2MCIntegrator(scene.get(), library));
            } else if (integrator == "drmlt") {
                std::shared_ptr<const PathFuncLib> library = BuildPathFuncLibrary(scene->options->bidirectional, maxDervDepth, hessian);
                integratorPtr = std::unique_ptr<DRIntegrator>(new DRIntegrator(scene.get(), library));
            } else {
                Error("Unknown integrator");
            }
            if(integrator != "mc") {
                integratorPtr->output_name = output_name;
                integratorPtr->verbose = !noProgressBar;
                integratorPtr->deterministic = deterministic;
            }

            if (integrator != "mc") {
                integratorPtr->Render();
            }
            WriteImage(scene->outputName, GetFilm(scene->camera.get()).get());
            std::cout <<  "Final image written to: " << scene->outputName << std::endl;
            if (chdir(cwd.c_str()) != 0) {
                Error("chdir failed");
            }
        }
        DptCleanup();
    } catch (std::exception &ex) {
        std::cerr << ex.what() << std::endl;
    }

    return 0;
}
