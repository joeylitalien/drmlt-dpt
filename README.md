# Delayed Rejection Metropolis Light Transport
-------------------------------------------------------------------------

![](./static/teaser.png)

  This repository contains the author's *cheap-then-expensive* implementation of [Delayed Rejection Metropolis Light Transport](https://joeylitalien.github.io/publications/drmlt) based on the differentiable path tracer [DPT](https://github.com/BachiLi/dpt) which implements [Anisotropic Gaussian Mutations for Metropolis Light Transport through Hessian-Hamiltonian Dynamics](https://people.csail.mit.edu/tzumao/h2mc/).

  If you want to understand the algorithm by looking at the code, You should start with:

  - Delayed Rejection MLT core algorithm
    - `src/integrator/delayed/delayed.*`
  - Delayed Rejection MLT core algorithm
    - `src/integrator/delayed/delayed.*`
  
  In case of problems/questions/comments don't hesitate to contact us
  directly: <damien.rioux-lavoie@mail.mcgill.ca> or <joey.litalien@mail.mcgill.ca>.
 

## Change from original DPT

Note that the original code was refactored a bit in order to facilitate our experimentations. 

As a notable case,  
- The program now use [CMake](https://cmake.org/) as build system instead of [tup](http://gittup.org/tup/)
- We removed the dependence on [ispc](https://ispc.github.io/ispc.html) to compile the derivative code it generates, it now use [gcc](https://gcc.gnu.org/).
- The dynamic library compilation is now done in parallel.
- We removed the need to set the environment variable  `DPT\_LIBPATH` to specify the path to the generated dynamic library. It is now assume to be in the same directory as the one where the program is executed from.
- The original `dpt/src/[mlt.* AND h2mc.*]` files where split into different integrator found in `src/integrator/*` and mutations type / markov states found in `src/mutation.*`
- A minor mistake in the original `dpt/src/h2mc/ComputeGaussian(...)` function was fixed. 
- Minor tweaks to statistics and outputs were added.
- We did not implement the *motion blur* and *lens mutations* functionalities in the delayed rejection integrator.

For more information, you can refer to the original [DPT](https://github.com/BachiLi/dpt) repository. 

## Requirement

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

You can use configure and build the project using [cmake](https://cmake.org/): 
```bash
mkdir build && cd "$_" 
cmake ../ 
make -j
```

To instal the tools dependence:
```bash
pacman -Sy  python3 python-pip 
pip3 install numpy matplotlib pyexr
```

### Dynamic Library

To create/compile the dynamic library of path derivatives, add the argument *--compile-bidirpathlib* when running the program. 
Here's an example using 10 threads:
```bash
cd <PATH_TO_DPT_EXECUTABLE>     
./dpt --p=10                    \
      --compile-bidirpathlib                       
```
doing so will save the dynamic library `pathlibbidir.so` in the `<PATH_TO_DPT_BIN>` directory.
Note that this will need to be build from scratch the first time the program is compiled, when changes in code have an impact on the path structure or when
deeper paths are needed.


## Integrators

We modified and added the following integrators:

 - `src/integrator/mcmc`: A variation of MMLT using isotropic Gaussian mutations
 - `src/integrator/h2mc`:  A variation of MMLT using H2MC aniotropic Gaussian mutations
 - `src/integrator/delayed/delayed`: This is the core our Delayed Rejection Metropolis Light Transport.

You can select between them by setting the `integrator=[mcmc, h2mc,drmlt]` parameters in the scene `.xml`.

### `drmlt` Parameters

| Parameter | Description | Requirement |
|:----------|:------------|:--|
| `type` | Delayed rejection framework | Optional (Default: `green`)   (Options: `mira, green`) |
| `acceptanceMap` | Output acceptance map | Optional (Default: `false`)  |
| `anisoperturbprob` | Probability of doing an isotropic or anisotropic move at the second stage | Optional (Default: `0.5`)  |
| `anisoperturbstddev` | Standard deviation of second stage isotropic proposal | Optional (Default: `0.0025`)  |

## Delayed Rejection Framework

This implementation of delayed rejection support 2 types of frameworks:

 - Original Framework: Proposed by [Tierney & Mira [1999]](https://www.researchgate.net/publication/2767014_Some_Adaptive_Monte_Carlo_Methods_for_Bayesian_Inference). Suffers from vanishing acceptance at the second stage.
 - Generalized Framework: Proposed by [Green & Mira [2001]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.20.7698&rep=rep1&type=pdf). Uses reversible jump to solve the vanishing acceptance problematic by sampling a costly intermediate state. 

You can select between them by setting the `type=[green, mira]` parameters in the scene `.xml`.


## Acceptance Map

When using the `drmlt` integrator, you can generate an acceptance map by setting the `acceptanceMap=true`  parameters in the scene `.xml`. Doing so will generate an RGB image such that the R channel correspond to the number of accepted samples at the first stage and the Green one to the second stage. You can convert this image to a heat map using the standalone script `tool/stages_heatmap.py`. You can then apply a heat map to the generated image by using:

```bash
python <PATH_TO_DPT__ROOT>/tools/stages_heatmap.py \
          -t <PATH_TO_ACCEPTANCE_MAP>/acceptance_map.exr \ 
          -c [0.2,0.8]
```

| Parameter | Description | Requirement |
|:----------|:------------|:--|
| `t` | Acceptance map | Required |
| `c` | Pixel range for heatmap images | Optional (Default: `[0,1]`) |


## Scenes



**TODO: change for the right links**
- [Torus](http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/research/2020_DRMLT/scenes/torus_dpt.zip)
- [Chess](http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/research/2020_DRMLT/scenes/chess_dpt.zip)
- [Veach Door](http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/research/2020_DRMLT/scenes/veach-door_dpt.zip)

With each scene we provide the following preset parameters used in our paper

- `scene_h2mc.xml`: run with the `h2mc` integrator.
- `scene_mcmc.xml`: run with the `mcmc` integrator.
- `scene_drmlt.xml`: run with the `drmlt` integrator using `green` framework.
- `scene_drmlt_map.xml`: Output the acceptance map using `drmlt` and `green`.
  
## Change logs

  2020/07/29: Initial code release


## License

This code is released under the The MIT License (MIT).