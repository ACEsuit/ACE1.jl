
using ACE1, JuLIP, LinearAlgebra, Statistics, StaticArrays, BenchmarkTools, Base.Threads
using JuLIP.MLIPs: combine 

@show nthreads() 

maxdeg = 18
N = 3

r0 = rnn(:Cu)
rcut = 2.5 * r0 
basis = ACE1.Utils.rpi_basis(; species=:Cu, N=N, r0=r0, maxdeg=maxdeg, rcut=rcut)
@show len = length(basis) 
NCO = 32
@show NCO 

co_c = Diagonal(1 ./ (1:len).^2) * randn(len, NCO)
c = mean(co_c, dims=(2,))[:]

V = combine(basis, c)
co_V = ACE1.committee_potential(basis, c, co_c)
Pr = V.pibasis.basis1p.J


using JuLIP
at = rattle!(bulk(:Cu, cubic=true) * 4, 0.2)
@show length(at)

nlist = neighbourlist(at, cutoff(V))
maxneigs = JuLIP.maxneigs(nlist)

##
#warmup 
energy(V, at)
ACE1.co_energy(co_V, at)

@info("Energy - allocating")
print("Potential : "); @time energy(V, at)
print("Committee : "); @time ACE1.co_energy(co_V, at)

E, co_E, tmp = ACE1.co_energy_alloc(co_V, at)
print("Committee!: "); @time ACE1.co_energy!(E, co_E, tmp, co_V, at)

##
#warmup 
forces(V, at)
ACE1.co_forces(co_V, at)

@info("Forces")
print("Potential : "); @time forces(V, at)
print("Committee : "); @time ACE1.co_forces(co_V, at)

F, co_F, tmp_d = ACE1.co_forces_alloc(co_V, at)
print("Committee!: "); @time ACE1.co_forces!(F, co_F, tmp_d, co_V, at)

## 

virial(V, at)
ACE1.co_virial(co_V, at)

@info("Virial")
print("Potential: "); @time virial(V, at)
print("Committee: "); @time ACE1.co_virial(co_V, at)

vir, co_vir, tmp_v = ACE1.co_virial_alloc(co_V, at)
print("Committee!: "); @time ACE1.co_virial!(vir, co_vir, tmp_v, co_V, at)

##

# @profview let vir = vir, co_vir = co_vir, tmp_v = tmp_v, co_V = co_V, at = at
#    for _ = 1:20 
#       ACE1.co_virial!(vir, co_vir, tmp_v, co_V, at)
#       ACE1.co_energy!(E, co_E, tmp, co_V, at)
#    end
# end
