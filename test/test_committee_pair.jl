

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


_randcoeffs_pair(B) = rand(length(B)) .* (1:length(B)).^(-2)

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

@info("check forces")
F1 = forces(V, at)
F2, co_F = ACE1.co_forces(co_V, at)
println_slim(@test (F1 ≈ F2 ≈ mean(co_F)))


@info("check virial")
vir1 = virial(V, at)
vir2, co_vir = ACE1.co_virial(co_V, at)
println_slim(@test (vir1 ≈ vir2 ≈ mean(co_vir)))


##

@info("check individual committee members: energy, ...")
_c0 = copy(co_V.coeffs)
for i in 1:NCO
   co_V.coeffs[:] .= ACE1.get_committee_coeffs(co_V, i)
   print_tf(@test( energy(co_V, at) ≈ co_E[i] ))
   print_tf(@test( forces(co_V, at) ≈ co_F[i] ))
   print_tf(@test( virial(co_V, at) ≈ co_vir[i] ))
end
co_V.coeffs[:] .= _c0

## 

@info("testing FIO ... ")
@info("   ... without committee")
println_slim(@test all( JuLIP.Testing.test_fio(V) ))
@info("   ... with committee")
println_slim(@test all( JuLIP.Testing.test_fio(co_V) ))


## 

@info(" Testing the committee for combined ACE + Pair")

Vace, co_Vace, Pr_ace = let 
   maxdeg = 10
   r0 = rnn(:W)
   rcut = 2.5 * r0 
   basis = ACE1.Utils.rpi_basis(; species=:W, N=3, r0=r0, maxdeg=maxdeg, rcut=rcut)
   len = length(basis) 
   co_c = Diagonal(1 ./ (1:len).^2) * randn(len, NCO)
   c = mean(co_c, dims=(2,))[:]
   V = combine(basis, c)
   co_V = ACE1.committee_potential(basis, c, co_c)
   Pr = V.pibasis.basis1p.J
   (V, co_V, Pr)
end



##

using JuLIP.MLIPs: combine
V_comb = JuLIP.MLIPs.SumIP(Vace, V)
co_V_comb = JuLIP.MLIPs.SumIP(co_Vace, co_V)


## 

at = rattle!(bulk(:W, cubic=true) * 2, 0.2)

@info("check energy")
E1 = energy(V_comb, at)
E2, co_E = ACE1.co_energy(co_V_comb, at)
println_slim(@test E1 ≈ E2 ≈ mean(co_E))
println_slim(@test co_E ≈ ACE1.co_energy(co_V, at)[2] + ACE1.co_energy(co_Vace, at)[2])

@info("check forces")
F1 = forces(V_comb, at)
F2, co_F = ACE1.co_forces(co_V_comb, at)
println_slim(@test (F1 ≈ F2 ≈ mean(co_F)))
println_slim(@test co_F ≈ ACE1.co_forces(co_V, at)[2] + ACE1.co_forces(co_Vace, at)[2])

@info("check virial")
vir1 = virial(V_comb, at)
vir2, co_vir = ACE1.co_virial(co_V_comb, at)
println_slim(@test (vir1 ≈ vir2 ≈ mean(co_vir)))
println_slim(@test co_vir ≈ ACE1.co_virial(co_V, at)[2] + ACE1.co_virial(co_Vace, at)[2])


## 

@info("check OneBody committee behaviour") 

V1 = JuLIP.OneBody(:W => randn())
co_V_comb1 = JuLIP.MLIPs.SumIP(V1, co_Vace, co_V)

@info("    ... energy")
E1, co_E1 = ACE1.co_energy(co_V_comb, at)
E2, co_E2 = ACE1.co_energy(co_V_comb1, at)
println_slim(@test E2 ≈ E1 + energy(V1, at))
println_slim(@test co_E1 .+ energy(V1, at) ≈ co_E2)

@info("    ... forces")
F1, co_F1 = ACE1.co_forces(co_V_comb, at)
F2, co_F2 = ACE1.co_forces(co_V_comb1, at)
println_slim(@test F2 ≈ F1)
println_slim(@test co_F1 ≈ co_F2)

@info("    ... virials")
vir1, co_vir1 = ACE1.co_virial(co_V_comb, at)
vir2, co_vir2 = ACE1.co_virial(co_V_comb1, at)
println_slim(@test vir2 ≈ vir1)
println_slim(@test co_vir1 ≈ co_vir2)
