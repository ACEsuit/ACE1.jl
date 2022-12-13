
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
co_c = Diagonal(1 ./ (1:len).^2) * randn(len, NCO)
c = mean(co_c, dims=(2,))[:]

V = combine(basis, c)
co_V = ACE1.committee_potential(basis, c, co_c)
Pr = V.pibasis.basis1p.J


using JuLIP
at = rattle!(bulk(:Cu, cubic=true) * 4, 0.2)
@show length(at)

##
#warmup 
energy(V, at)
ACE1.co_energy(co_V, at)

@info("Energy")
print("Potential: "); @time energy(V, at)
print("Committee: "); @time ACE1.co_energy(co_V, at)

##
#warmup 
forces(V, at)
ACE1.co_forces(co_V, at)

@info("Forces")
print("Potential: "); @time forces(V, at)
print("Committee: "); @time ACE1.co_forces(co_V, at)

## 

# using Base.Threads
# nt = nthreads()
# NCO = ACE1.ncommittee(co_V)
# T = ACE1.fltype(co_V)
# tmp_d = [ ACE1.alloc_temp_d(co_V, at) for _ in 1:nt ]
# F0 = zeros(JVec{T}, length(at))
# F = [ copy(F0) for _ in 1:nt ]
# co_F = [ SVector(ntuple(_ -> copy(F0), NCO)...) for _ in 1:nt ]

# ACE1.co_forces!(F, co_F, tmp_d, co_V, at)

# print("Committe!: "); @time ACE1.co_forces!(F, co_F, tmp_d, co_V, at)

# ##

# @profview let co_V = co_V, at=at 
#    for _ = 1:40
#       ACE1.co_forces(co_V, at)
#    end
# end

# ##

# @profview let F = F, co_F = co_F, tmp_d = tmp_d, co_V = co_V, at = at
#    for _ = 1:30
#       ACE1.co_forces!(F, co_F, tmp_d, co_V, at)
#    end
# end

##

virial(V, at)
ACE1.co_virial(co_V, at)

@info("Virial")
print("Potential: "); @time virial(V, at)
print("Committee: "); @time ACE1.co_virial(co_V, at)


##

# @profview let co_V = co_V, at = at
#    for _ = 1:10
#       ACE1.co_virial(co_V, at)
#    end
# end