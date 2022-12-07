
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
    @testset "PIBasis"  begin include("test_pibasis.jl") end 
    @testset "PIPotential"  begin include("test_pipot.jl") end 

    # ------------------------
    #   rotation_invariance
    # TODO: implement a check whether Sympy is available and run test conditionally 
    include("rpi/test_cg.jl")
    @testset "RPIBasis"  begin include("rpi/test_rpibasis.jl") end 

    # ----------------------
    #   pair potentials
    @testset "PolyPairBasis" begin include("pair/test_pair_basis.jl") end 
    @testset "PolyPairPot" begin include("pair/test_pair_pot.jl") end 
    @testset "RepulsiveCore" begin include("pair/test_repulsion.jl") end 

    # ----------------------
    #   miscellaneous ...
    # TODO: These tests are current failing - should be re-examined and fixed 
    # include("compat/test_compat.jl")

    @testset "ACE-Committee" begin include("test_committee.jl") end 
    @testset "Pair-Committee" begin include("test_committee_pair.jl") end 
    @testset "Any"  begin include("test_any.jl") end 
    @testset "Multi-Transform" begin include("polynomials/test_multitrans.jl") end 
end
