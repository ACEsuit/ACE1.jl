

##

@info("""-------------------------------------------
         Testing Committee Implementation
-------------------------------------------------""")

using ACE1, ACE1.Testing
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing, Random
using JuLIP: evaluate, evaluate_d, evaluate_ed
using ACE1: combine
using Statistics: mean 

##

@info("Basic test of PIPotential construction and evaluation")
maxdeg = 10
r0 = rnn(:Cu)
rcut = 2.5 * r0 
basis = ACE1.Utils.rpi_basis(; species=:Cu, N=3, r0=r0, maxdeg=maxdeg, rcut=rcut)
len = length(basis) 
NCO = 12 
co_c = Diagonal(1 ./ (1:len).^2) * randn(len, 12)
c = mean(co_c, dims=(2,))[:]

V = combine(basis, c)
co_V = ACE1.committee_potential(basis, c, co_c)
Pr = V.pibasis.basis1p.J

Nat = 15
Rs, Zs, z0 = ACE1.rand_nhd(Nat, Pr, :Cu)

val1 = evaluate(V, Rs, Zs, z0) 
val2 = evaluate(co_V, Rs, Zs, z0) 
val3, co_val = ACE1.co_evaluate(co_V, Rs, Zs, z0)

@info("check the three `evaluate` calls are consistent")
println_slim(@test(val1 ≈ val2 ≈ val3))

@info("check the co_evaluate is consistent")
println_slim(@test(val3 ≈ mean(co_val)))

@info("check co_evaluate_d")

dEs1 = evaluate_d(V, Rs, Zs, z0)
dEs2 = evaluate_d(co_V, Rs, Zs, z0)
dEs3, co_dEs = ACE1.co_evaluate_d(co_V, Rs, Zs, z0)

@info("check consistency of three evaluate_d")
println_slim(@test(dEs1 ≈ dEs2 ≈ dEs3))

@info("check consistency of evaluate_d with committee")
println_slim(@test(dEs3 ≈ mean(co_dEs) ))

@info("check individual committee members")
iz0 = ACE1.z2i(V, z0)
_c0 = copy(co_V.coeffs[iz0])
for i in 1:NCO
   co_V.coeffs[iz0][:] .= ACE1.get_committee_coeffs(co_V, z0, i)
   print_tf(@test( evaluate(co_V, Rs, Zs, z0) ≈ co_val[i] ))
   print_tf(@test( evaluate_d(co_V, Rs, Zs, z0) ≈ co_dEs[i] ))
end
co_V.coeffs[iz0][:] .= _c0[:]

## next up total energy, forces, virials 

using JuLIP
at = rattle!(bulk(:Cu, cubic=true) * 2, 0.2)

@info("check energy")
E1 = energy(V, at)
E2, co_E = ACE1.co_energy(co_V, at)
println_slim(@test E1 ≈ E2 ≈ mean(co_E))

@info("check forces")
F1 = forces(V, at)
F2, co_F = ACE1.co_forces(co_V, at)
println_slim(@test (F1 ≈ F2 ≈ mean(co_F)))

##

@info("check virial")
vir1 = virial(V, at)
vir2, co_vir = ACE1.co_virial(co_V, at)
println_slim(@test (vir1 ≈ vir2 ≈ mean(co_vir)))

##

@info("check individual committee members: energy, ...")
iz0 = ACE1.z2i(V, z0)
_c0 = copy(co_V.coeffs[iz0])
for i in 1:NCO
   co_V.coeffs[iz0][:] .= ACE1.get_committee_coeffs(co_V, z0, i)
   print_tf(@test( energy(co_V, at) ≈ co_E[i] ))
   print_tf(@test( forces(co_V, at) ≈ co_F[i] ))
   print_tf(@test( virial(co_V, at) ≈ co_vir[i] ))
end
co_V.coeffs[iz0][:] .= _c0[:]
