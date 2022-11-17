
# --------------------------------------------------------------------------
# ACE1.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------

using ACE1
using Random, Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d, evaluate_ed


##

@info("Basic test of RPIBasis construction and evaluation")
maxdeg = 8
N = 3
r0 = 1.0
rcut = 3.0
species = [:Al, :Ti, :Si]
trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = SparsePSHDegree()
P1 = BasicPSH1pBasis(Pr; species = species, D = D)
basis = RPIBasis(P1, N, D, maxdeg)

##

at = bulk(:Al, cubic=true) * 3 
set_pbc!(at, false)
at.Z[2:3:end] .= AtomicNumber(:Ti)
at.Z[5:5:end] .= AtomicNumber(:Si)
rattle!(at, 0.1)

# nfeature x natoms matrix
X = ACE1.Descriptors.descriptors(basis, at)
# nfeature x natoms x natoms tensor with 
#   dX[i, j, :] the gradient of X[i, j] w.r.t. positions.
dX = ACE1.Descriptors.descriptors_d(basis, at)

##

using Printf 
U = randn(JVecF, length(at))

function _at(t) 
   at1 = deepcopy(at)
   X1 = at1.X + t * U 
   set_positions!(at1, X1)
   return at1
end

F(t) = ACE1.Descriptors.descriptors(basis, _at(t))

function dF(t)
   dX = ACE1.Descriptors.descriptors_d(basis, _at(t))
   return [ dot(dX[i, j, :], U) for i = 1:size(dX, 1), j = 1:size(dX, 2) ]
end

##

dX0 = dF(0.0)
@printf("   h   |  error \n")
for h in (0.1).^(2:10)
   dXh = (F(h) - F(0.0)) / h
   @printf(" %.0e |  %.2e \n", h, norm(dXh - dX0, Inf))
end
