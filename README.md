# Delayed Rejection Metropolis Light Transport

This repository contains the *cheap-then-expensive* implementation of [Delayed Rejection Metropolis Light Transport](https://joeylitalien.github.io/publications/drmlt) based on the differentiable path tracer [DPT](https://github.com/BachiLi/dpt) which implements [Anisotropic Gaussian Mutations for Metropolis Light Transport through Hessian-Hamiltonian Dynamics](https://people.csail.mit.edu/tzumao/h2mc/).

If you want to understand the algorithm by looking at the code, you should start with:

  - __Delayed Rejection MLT__
    - `src/integrators/delayed/delayed.*`
  - __H2MC__
    - `src/integrators/h2mc.*`

In case of problems/questions/comments, do not hesitate to contact the authors directly.
 

## Change from Original DPT

Note that the original code was refactored a bit in order to facilitate our experiments. Among the most important changes are:
- The program now uses [CMake](https://cmake.org/) as a build system instead of [tup](http://gittup.org/tup/)
- We removed the dependency on [ispc](https://ispc.github.io/ispc.html) to compile the derivative code it generates, it now uses [gcc](https://gcc.gnu.org/).
- The dynamic library compilation is now done in parallel.
- We removed the need to set the environment variable `DPT\_LIBPATH` to specify the path to the generated dynamic library. It is now assumed to be in the same directory as the one where the program is executed from.
- The original `dpt/src/[mlt.* & h2mc.*]` files where split into different integrators found in `src/integrator/*` and mutations type / Markov states found in `src/mutation.*`
- A minor mistake in the original `dpt/src/h2mc/ComputeGaussian(...)` function was fixed. 
- Minor tweaks to statistics and outputs were added.
- We did not implement the *motion blur* and *lens mutations* functionalities in the delayed rejection integrator.

For more information, you can refer to the original [DPT](https://github.com/BachiLi/dpt) repository. 

## Requirements

### DPT
- [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page), already in external
- [glog](https://github.com/google/glog), already in external
- [embree](https://embree.github.io/), already in external
- [OpenImageIO v2](https://github.com/OpenImageIO/oiio)
- [zlib](http://www.zlib.net/)
- [pugixml](http://pugixml.org/), already in src
- [PCG](http://www.pcg-random.org/), already in src


### Python
- [PyEXR](https://github.com/tvogels/pyexr)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)


## Compiling

We provided installation instructions for ArchLinux only, but the installation procedure is similar on Ubuntu using `apt install`. Configure and build the project using [Cmake](https://cmake.org/): 
```bash
mkdir build && cd "$_" 
cmake ../ 
make -j
```

To install the tooling dependencies, run:
```bash
pacman -Sy python3 python-pip
pip3 install numpy matplotlib pyexr
```

### Dynamic Library

To create/compile the dynamic library of path derivatives, add the argument `--compile-bidirpathlib` when running the program. 
Below is an example using 10 threads:
```bash
cd <PATH_TO_DPT_EXECUTABLE>     
./dpt --p=10 --compile-bidirpathlib                       
```
Doing so will save the dynamic library `pathlibbidir.so` in the `<PATH_TO_DPT_BIN>` directory. Note that this will need to be built from scratch the first time the program is compiled, when changes in code have an impact on the path structure (e.g. adding a BRDF) or when deeper paths are needed.


## Integrators

We modified and added the following integrators:

 - `src/integrator/mcmc`: A variation of MMLT using isotropic Gaussian mutations.
 - `src/integrator/h2mc`:  A variation of MMLT using H2MC aniotropic Gaussian mutations (H2MC).
 - `src/integrator/delayed/delayed`: This is the core of our DRMLT algorithm.

You can select between them by setting the `integrator=[mcmc, h2mc,drmlt]` parameters in the scene `.xml`.

### `drmlt`

| Parameter | Description | Requirement |
|:----------|:------------|:--|
| `type` | Delayed rejection framework | Optional (Default: `green`)   (Options: `mira, green`) |
| `acceptanceMap` | Output acceptance map | Optional (Default: `false`)  |
| `anisoperturbprob` | Probability of doing an isotropic or anisotropic move at the second stage | Optional (Default: `0.5`)  |
| `anisoperturbstddev` | Standard deviation of second stage isotropic proposal | Optional (Default: `0.0025`)  |

## Delayed Rejection Framework

This implementation of delayed rejection support two types of frameworks:

 - __Original Framework:__ Proposed by [Tierney & Mira [1999]](https://www.researchgate.net/publication/2767014_Some_Adaptive_Monte_Carlo_Methods_for_Bayesian_Inference). Suffers from vanishing acceptance at the second stage.
 - __Generalized Framework:__ Proposed by [Green & Mira [2001]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.20.7698&rep=rep1&type=pdf). Uses reversible jump to solve the vanishing acceptance problematic by sampling an expensive intermediate state. 

You can select between them by setting the `type=[green, mira]` parameters in the scene `.xml`.


## Acceptance Map

When using the `drmlt` integrator, you can generate an acceptance map by setting the `acceptanceMap=true`  parameters in the scene `.xml`. Doing so will generate an RGB image such that the R-channel corresponds to the number of accepted samples at the first stage and the G-channel is the same for the second stage. To convert this image to a heatmap, use the standalone script `./tools/stages_heatmap.py`. For example, the following command saves the acceptance map during rendering:

```bash
python <PATH_TO_DPT__ROOT>/tools/stages_heatmap.py \
          -t <PATH_TO_ACCEPTANCE_MAP>/acceptance_map.exr \ 
          -c [0.2,0.8]
```

| Parameter | Description | Requirement |
|:----------|:------------|:--|
| `t` | Acceptance map | Required |
| `c` | Pixel range (clip) for heatmap images | Optional (Default: `[0,1]`) |


## Scenes

- [Torus](http://adrien-gruson.com/research/2020_DRMLT/scenes/torus_dpt.zip)
- [Chess](http://adrien-gruson.com/research/2020_DRMLT/scenes/chess_dpt.zip)
- [Veach Door](http://adrien-gruson.com/research/2020_DRMLT/scenes/veach-door_dpt.zip)

With each scene, we provide the following preset parameters used to generate the renders in our paper:

- `scene_h2mc.xml`: run with the `h2mc` integrator.
- `scene_mcmc.xml`: run with the `mcmc` integrator.
- `scene_drmlt.xml`: run with the `drmlt` integrator using `green` framework.
- `scene_drmlt_map.xml`: Output the acceptance map using `drmlt` and `green`.
  
  
## Change Llogs

- 2020/07/29: Initial code release


## License

This code is released under the The MIT License (MIT).
