

##


using ACE1, ACE1.Testing
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing, Random
using JuLIP: evaluate, evaluate_d, evaluate_ed
using ACE1: combine

##

@info("Basic test of PIPotential construction and evaluation")
maxdeg = 10
r0 = 1.0
rcut = 3.0
basis = ACE1.Utils.rpi_basis(; species=:X, N=3, r0=r0, maxdeg=maxdeg, rcut=rcut)
Pr = V.pibasis.basis1p.J
len = length(basis) 
NCO = 12 
co_c = Diagonal(1 ./ (1:len).^2) * randn(len, 12)
c = (sum(co_c, dims=(2,)) / NCO )[:]

V = combine(basis, c)
co_V = ACE1.committee_potential(basis, c, co_c)

Nat = 15
Rs, Zs, z0 = ACE1.rand_nhd(Nat, Pr, :X)

val1 = evaluate(V, Rs, Zs, z0) 
val2 = evaluate(co_V, Rs, Zs, z0) 
val3, co_val = ACE1.co_evaluate(co_V, Rs, Zs, z0)

@info("check the three `evaluate` calls are consistent")
println_slim(@test(val1 ≈ val2 ≈ val3))

@info("check the co_evaluate is consistent")
println_slim(@test(val3 ≈ sum(co_val)/NCO))


