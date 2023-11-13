# ACE1.jl

<!--
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ACEsuit.github.io/ACE1docs.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ACEsuit.github.io/ACE1docs.jl/dev)

 [![Build Status](https://travis-ci.com/JuliaMolSim/ACE.jl.svg?branch=master)](https://travis-ci.com/JuliaMolSim/ACE.jl)

[![Codecov](https://codecov.io/gh/JuliaMolSim/ACE.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaMolSim/ACE.jl) -->

--- 

**WARNING:** This package now acts mostly as a (to be retired) backend to other packages. If you have come to this repository to fit interatomic potentials, then please go to the user-facing companion package [`ACEpotentials.jl`](https://github.com/ACEsuit/ACEpotentials.jl) which also provides documentation and tutorials on the usage of `ACE1.jl`.

---

This package implements a flavour of the *Atomic Cluster Expansion*; i.e., parameterisation schemes for permutation and isometry invariant functions, primarily for the purpose of modelling invariant atomic properties. It provides constructions of symmetric polynomial bases, imposing permutation and isometry invariance. 

## References

When using this software, please cite the following references

* Drautz, R.: Atomic cluster expansion for accurate and transferable interatomic potentials. Phys. Rev. B Condens. Matter. 99, 014104 (2019). [[html]](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.014104) 

* M. Bachmayr, G. Csanyi, G. Dusson, R. Drautz, S. Etter, C. van der Oord, and C. Ortner. Atomic cluster expansion: Completeness, efficiency and stability. J. Comp. Phys. 454 (2022). [[html]](https://www.sciencedirect.com/science/article/pii/S0021999122000080?via%3Dihub) [[arxiv]](https://arxiv.org/abs/1911.03550)


## License

`ACE1.jl` is © 2019, Christoph Ortner

`ACE1.jl` is published and distributed under the [Academic Software License v1.0 (ASL).](ASL.md)

`ACE1.jl` is distributed in the hope that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the ASL for more details.

You should have received a copy of the ASL along with this program; if not, write to Christoph Ortner, christophortner0@gmail.com. It is also published at [https://github.com/gabor1/ASL/blob/main/ASL.md](https://github.com/gabor1/ASL/blob/main/ASL.md).

You may contact the original licensor at `christophortner0@gmail.com`.
