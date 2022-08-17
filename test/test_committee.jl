

##

@info("""-------------------------------------------
         Testing Committee Implementation
-------------------------------------------------""")

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
len = length(basis) 
NCO = 12 
co_c = Diagonal(1 ./ (1:len).^2) * randn(len, 12)
c = (sum(co_c, dims=(2,)) / NCO )[:]

V = combine(basis, c)
co_V = ACE1.committee_potential(basis, c, co_c)
Pr = V.pibasis.basis1p.J

Nat = 15
Rs, Zs, z0 = ACE1.rand_nhd(Nat, Pr, :X)

val1 = evaluate(V, Rs, Zs, z0) 
val2 = evaluate(co_V, Rs, Zs, z0) 
val3, co_val = ACE1.co_evaluate(co_V, Rs, Zs, z0)

@info("check the three `evaluate` calls are consistent")
println_slim(@test(val1 ≈ val2 ≈ val3))

@info("check the co_evaluate is consistent")
println_slim(@test(val3 ≈ sum(co_val)/NCO))

@info("check co_evaluate_d")

dEs1 = evaluate_d(V, Rs, Zs, z0)
dEs2 = evaluate_d(co_V, Rs, Zs, z0)
dEs3, co_dEs = ACE1.co_evaluate_d(co_V, Rs, Zs, z0)

@info("check consistency of three evaluate_d")
println_slim(@test(dEs1 ≈ dEs2 ≈ dEs3))

@info("check consistency of evaluate_d with committee")
println_slim(@test(dEs3 ≈ sum(co_dEs) / NCO ))

@info("check individual committee members")
iz0 = ACE1.z2i(V, z0)
for i in 1:NCO
   co_V.coeffs[iz0][:] .= ACE1.get_committee_coeffs(co_V, z0, i)
   print_tf(@test( evaluate(co_V, Rs, Zs, z0) ≈ co_val[i] ))
   print_tf(@test( evaluate_d(co_V, Rs, Zs, z0) ≈ co_dEs[i] ))
end

## next up total energy, forces, virials 

