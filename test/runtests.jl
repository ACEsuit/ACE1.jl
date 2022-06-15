
# --------------------------------------------------------------------------
# ACE1.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------

##
using ACE1, Test, Printf, LinearAlgebra, StaticArrays, BenchmarkTools,
      JuLIP, JuLIP.Testing

##
@testset "ACE1.jl" begin
    # ------------------------------------------
    #   basic polynomial basis building blocks
    @testset "Ylm" begin include("polynomials/test_ylm.jl") end 
    @testset "Real  Ylm" begin include("polynomials/test_rylm.jl") end 
    @testset "Transforms" begin include("polynomials/test_transforms.jl") end 
    @testset "OrthogonalPolynomials" begin include("polynomials/test_orthpolys.jl") end 

    # --------------------------------------------
    # core permutation-invariant functionality
    @testset "1-Particle Basis"  begin include("test_1pbasis.jl") end 
    include("test_pibasis.jl")
    include("test_pipot.jl")

    # ------------------------
    #   rotation_invariance
    # TODO: implement a check whether Sympy is available and run test conditionally 
    include("rpi/test_cg.jl")
    include("rpi/test_rpibasis.jl")

    # ----------------------
    #   pair potentials
    include("pair/test_pair_basis.jl")
    include("pair/test_pair_pot.jl")
    include("pair/test_repulsion.jl")

    # ----------------------
    #   miscallaneous ...
    # TODO: These tests are current failing - should be re-examined and fixed 
    # include("compat/test_compat.jl")

    include("test_any.jl")
    include("polynomials/test_multitrans.jl")
end
