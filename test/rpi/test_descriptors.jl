
# --------------------------------------------------------------------------
# ACE1.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------




##

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
rpibasis = RPIBasis(P1, N, D, maxdeg)

##

at = bulk(:Al, cubic=true) * 3 
at.Z[2:3:end] .= AtomicNumber(:Ti)
at.Z[5:5:end] .= AtomicNumber(:Si)
rattle!(at, 0.1)

X = ACE1.Descriptors.descriptors(rpibasis, at)
dX = ACE1.Descriptors.descriptors_d(rpibasis, at)

##

using Printf 
U = randn(JVecF, length(at))
_at(t) = ( at1 = deepcopy(at); at1.X[:] = at.X[:] + t*U; at1 )

F(t) = ACE1.Descriptors.descriptors(rpibasis, _at(t))
dF(t) = (
   dX = ACE1.Descriptors.descriptors_d(rpibasis, _at(t));
   return [ dot(dX[i, j, :], U) for i = 1:size(dX, 1), j = 1:size(dX, 2) ]
)

##

dX0 = dF(0.0)
@printf("   h   |  error \n")
for h in (0.1).^(2:10)
   dXh = (F(h) - F(0.0)) / h
   @printf(" %.0e |  %.2e \n", h, norm(dXh - dX0, Inf))
end