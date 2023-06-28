
# --------------------------------------------------------------------------
# ACE1.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


module ACE1

using Reexport
@reexport using JuLIP

import Pkg 
import Base.pkgdir
pkgproject(m::Core.Module) = Pkg.Operations.read_project(Pkg.Types.projectfile_path(pkgdir(m)))
pkgversion(m::Core.Module) = pkgproject(m).version
# const ACE1_VERSION_NUMBER = pkgversion(@__MODULE__())

# external imports that are useful for all submodules
include("extimports.jl")


include("auxiliary.jl")
include("prototypes.jl")


# basic polynomial building blocks
include("polynomials/sphericalharmonics.jl")
include("polynomials/transforms.jl"); @reexport using ACE1.Transforms
include("polynomials/orthpolys.jl"); @reexport using ACE1.OrthPolys
include("polynomials/splines.jl"); @reexport using ACE1.Splines

# The One-particle basis is the first proper building block
include("oneparticlebasis.jl")

include("grapheval.jl")

# the permutation-invariant basis: this is a key building block
# for other bases but can also be a useful export itself
include("pibasis.jl")
include("pipot.jl")

# rotation-invariant site potentials (incl the ACE model)
include("rpi/rpi.jl")
@reexport using ACE1.RPI

# pair potentials + repulsion
include("pairpots/pair.jl");
@reexport using ACE1.PairPotentials

include("committee.jl")
include("descriptors.jl")
include("cleanup.jl")


# lots of stuff related to random samples:
#  - random configurations
#  - random potentials
#  ...
include("random.jl")
@reexport using ACE1.Random


include("utils.jl")
@reexport using ACE1.Utils


include("compat/compat.jl")

include("fio.jl")



include("testing/testing.jl")

# - bond model
# - pure basis
# - real basis
# - regularisers
# - descriptors
# - random potentials


end # module
