

##

@info("""-------------------------------------------
         Testing Pair Committee Implementation
-------------------------------------------------""")

##

using ACE1
using Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d
using JuLIP.Potentials: i2z, numz
using JuLIP.MLIPs: combine
using Statistics: mean 
using ACE1.Testing: println_slim


randr() = 1.0 + rand()
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)

##

maxdeg = 8
r0 = 1.0
rcut = 3.0

trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
pB = ACE1.PairPotentials.PolyPairBasis(Pr, :W)

NCO = 12 
co_c = hcat([randcoeffs(pB) for i = 1:NCO]...)
coeffs = vec(sum(co_c, dims=2) / NCO)

V = combine(pB, coeffs)
co_V = ACE1.committee_potential(pB, coeffs, co_c)


##
Nat = 15
Rs, Zs, z0 = ACE1.rand_nhd(Nat, Pr, :W)
rr = norm.(Rs)

tmp = ACE1.alloc_temp(V, 100)
val1 = [ ACE1.evaluate!(tmp, V, r, z, z0) for (r, z) in zip(rr, Zs) ]
val2 = [ ACE1.evaluate!(tmp, co_V, r, z, z0) for (r, z) in zip(rr, Zs) ]
val3 = [ ACE1.co_evaluate!(tmp, co_V, r, z, z0) for (r, z) in zip(rr, Zs) ]

@info("check the three `evaluate` calls are consistent")
println_slim(@test( val1 ≈ val2 ≈ [ v[1] for v in val3] ))

@info("check the co_evaluate is consistent")
println_slim(@test all( [ v[2] isa AbstractVector && v[1] ≈ mean(v[2]) for v in val3] ) )


## 

at = rattle!(bulk(:W, cubic=true) * 2, 0.2)

@info("check energy")
E1 = energy(V, at)
E2, co_E = ACE1.co_energy(co_V, at)
println_slim(@test E1 ≈ E2 ≈ mean(co_E))



## --------------------------------------------------------------------------


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

## 

@info("testing FIO ... ")
@info("   ... without committee")
println_slim(@test all( JuLIP.Testing.test_fio(V) ))
@info("   ... with committee")
println_slim(@test all( JuLIP.Testing.test_fio(co_V) ))