

using JuLIP, Test, ACEbase, LinearAlgebra
using ACEbase.Testing: println_slim 

@info(" ------ Limited sanity tests for site_energy ------- ")

##

B1 = pair_basis(; species = [:Cu], r0 = 2.55, maxdeg = 4, rcut = 5.5)
B2 = rpi_basis(; species = [:Cu], N = 3, r0 = 2.55, rcut = 5.5)             
basis = JuLIP.MLIPs.IPSuperBasis(B1, B2)

at = rattle!(bulk(:Cu, cubic=true, pbc=true) * 2, 0.1)

##
# basic tests 

se = site_energy(basis, at, 1)

@test B1 == basis.BB[1]
dse1 = site_energy_d(B1, at, 1)
println_slim(@test length(dse1) == length(B1))

@test B2 == basis.BB[2]
dse2 = site_energy_d(B2, at, 1)
println_slim(@test length(dse2) == length(B2))

dse = site_energy_d(basis, at, 1)
println_slim(@test vcat(dse1, dse2) ≈ dse)
println_slim(@test all(d -> length(d) == length(at), dse))

##
# consistency of energy with site energies 

println_slim(@test (
      energy(basis, at) ≈ sum( site_energy(basis, at, i)
                                 for i = 1:length(at) ) ) )

## 
# consistency of site energy with gradient 

_basis = basis

u = randn(length(_basis))
dX = randn(JVecF, length(at)) 
X0 = deepcopy(at.X)
at_t = t -> set_positions!(at, X0 + t * dX)
F = t -> dot(u, site_energy(_basis, at_t(t), 1))
dF = t -> begin
      dEs = site_energy_d(_basis, at_t(t), 1)
      u_dEs = sum(u[n] * dEs[n] for n = 1:length(_basis))
      return dot(dX, u_dEs)
   end
ACEbase.Testing.fdtest(F, dF, 0.0)

