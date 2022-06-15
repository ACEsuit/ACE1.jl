using ACE1pack, ACE1, IPFitting, StaticArrays, Plots
using ACE1: transformed_jacobi, transformed_jacobi_env
using LinearAlgebra: Diagonal

data_file = joinpath(ACE1pack.artifact("TiAl_tutorial"), "TiAl_tutorial.xyz")
data = IPFitting.Data.read_xyz(data_file, energy_key="energy", force_key="force", virial_key="virial")
train = data[1:2:end]
test = data[2:2:end]

##

species = [:Ti, :Al]
r0 = 2.88
rcut = 5.5 

# standard ace basis 

maxdeg = 6 
trans = AgnesiTransform(r0)
# envelope = ACE1.TwoSidedEnvelope(trans(rcut), trans(0.66*r0), 2, 2)
# rbasis = transformed_jacobi_env(maxdeg, trans, envelope, rcut)

ACE_B = ace_basis(species = species,
                  N = 3,
                  r0 = r0,
                  rin = 0.7 * r0,
                  rcut = rcut,
                  pcut = 2, pin = 2, 
                  trans=trans, 
                  maxdeg = maxdeg);

# radial basis 
trans_r = AgnesiTransform(r0)
envelope_r = ACE1.PolyEnvelope(2, r0, rcut)
Jold = transformed_jacobi(16, trans_r, rcut)
Jnew = transformed_jacobi_env(16, trans_r, envelope_r, rcut)

Bpair_old = PolyPairBasis(Jold, species)
Bpair_new = PolyPairBasis(Jnew, species)

Bold = JuLIP.MLIPs.IPSuperBasis([Bpair_old, ACE_B]);
Bnew = JuLIP.MLIPs.IPSuperBasis([Bpair_new, ACE_B]);

# reference potential (numbers from Cas' tutorial)
Vref = OneBody(:Ti => -1586.0195, :Al => -105.5954)

# regression weights (numbers from Cas' tutorial)
weights = Dict(
        "FLD_TiAl" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 ),
        "TiAl_T5000" => Dict("E" => 5.0, "F" => 1.0 , "V" => 1.0 ))

##

# solver parameters 
# solver = Dict("solver" => :rrqr, 
#               "rrqr_tol" => 1e-6, 
#               "P" => Γ
#               )
# solver = Dict("solver" => :brr, 
#               "brr_tol" => 1e-3, 
#               "P" => Γ
#               )
# solver= Dict(
#    "solver" => :ard,
#    "ard_tol" => 1e-3,
#    "ard_threshold_lambda" => 10000)


## fit both potentials - Parameters for Polytransform 
if trans isa PolyTransform
   dB = LsqDB("", Bold, train);
   # smoothness prior 
   Γ = Diagonal(vcat(ACE1.scaling.(dB.basis.BB, 3)...))

   solver = Dict("solver" => :rrqr, "rrqr_tol" => 3e-6, "P" => Γ)
   IP_old, lsqinfo_old = lsqfit(dB, solver=solver, weights=weights, Vref=Vref, error_table = true)
   IPFitting.add_fits!(IP_old, test)
   rmse_table(lsqinfo_old["errors"])
   rmse_table(test)

   dB = LsqDB("", Bnew, train);
   solver = Dict("solver" => :rrqr, "rrqr_tol" => 3e-6, "P" => Γ)
   IP_new, lsqinfo_new = lsqfit(dB, solver=solver, weights=weights, Vref=Vref, error_table = true)
   IPFitting.add_fits!(IP_new, test)
   rmse_table(lsqinfo_new["errors"])
   rmse_table(test)

elseif trans isa AgnesiTransform
   # fit both potentials - Parameters for AgnesiTransform

   dB = LsqDB("", Bold, train);
   Γ = Diagonal(vcat(ACE1.scaling.(dB.basis.BB, 3)...))
   solver = Dict("solver" => :rrqr, "rrqr_tol" => 1e-6, "P" => Γ)
   IP_old, lsqinfo_old = lsqfit(dB, solver=solver, weights=weights, Vref=Vref, error_table = true)
   IPFitting.add_fits!(IP_old, test)
   rmse_table(lsqinfo_old["errors"])
   rmse_table(test)

   dB = LsqDB("", Bnew, train);
   solver = Dict("solver" => :rrqr, "rrqr_tol" => 2e-6, "P" => Γ)
   IP_new, lsqinfo_new = lsqfit(dB, solver=solver, weights=weights, Vref=Vref, error_table = true)
   IPFitting.add_fits!(IP_new, test)
   rmse_table(lsqinfo_new["errors"])
   rmse_table(test)
else
   error("unknown transform")
end 

##

@info("Training Error - Old Basis")
rmse_table(lsqinfo_old["errors"])

@info("Training Error - New Basis")
rmse_table(lsqinfo_new["errors"])

##

@info("Test Error - Old Basis")
IPFitting.add_fits!(IP_old, test)
rmse_table(test)

@info("Test Error - New Basis")
IPFitting.add_fits!(IP_new, test)
rmse_table(test)

##

at_dimer(r, z1, z0) = Atoms(X = [ SVector(0.0,0.0,0.0), SVector(r, 0.0, 0.0)], 
                            Z = [z0, z1], pbc = false, 
                            cell = [r+1 0 0; 0 1 0; 0 0 1])

function dimer_energy(IP, r, z1, z0)
   at = at_dimer(r, z1, z0)
   at1 = Atoms(X = [SVector(0.0,0.0,0.0),], Z = [z1, ], pbc = false, cell = [1.0 0 0; 0 1.0 0; 0 0 1.0]) 
   at2 = Atoms(X = [SVector(0.0,0.0,0.0),], Z = [z0, ], pbc = false, cell = [1.0 0 0; 0 1.0 0; 0 0 1.0]) 
   return JuLIP.energy(IP, at) - JuLIP.energy(IP, at1) - JuLIP.energy(IP, at2)
end

zAl = AtomicNumber(:Al)
zTi = AtomicNumber(:Ti)
rr = range(0.1, rcut, length=200)

plt = plot(; ylims = [-4, 4])
plot!(plt, rr, dimer_energy.(Ref(IP_old), rr, zAl, zAl), lw=2, c=1, ls=:dash, label = "AlAl-old")
plot!(plt, rr, dimer_energy.(Ref(IP_new), rr, zAl, zAl), lw=2, c=1, ls=:solid, label = "AlAl-new")
plot!(plt, rr, dimer_energy.(Ref(IP_old), rr, zAl, zTi), lw=2, c=2, ls=:dash, label =  "AlTi-old")
plot!(plt, rr, dimer_energy.(Ref(IP_new), rr, zAl, zTi), lw=2, c=2, ls=:solid, label = "AlTi-new")
plot!(plt, rr, dimer_energy.(Ref(IP_old), rr, zTi, zTi), lw=2, c=3, ls=:dash, label =  "TiTi-old")
plot!(plt, rr, dimer_energy.(Ref(IP_new), rr, zTi, zTi), lw=2, c=3, ls=:solid, label = "TiTi-new")
vline!(plt, [rnn(:Ti), rnn(:Al)], c=:black, lw=2, label = "rnn")


##

# solver = Dict("solver" => :rrqr, "rrqr_tol" => 3e-6, "P" => Γ)

using JuLIP: evaluate 
using IPFitting: Dat 
zbl = JuLIP.ZBLPotential()

function dat_dimer(r, z1, z0) 
   at = at_dimer(r, z1, z0)
   E = ACE1.evaluate(envelope, r)
   # zbl(r, z1, z0) 
   # + Vref.E0[chemical_symbol(z1)] + Vref.E0[chemical_symbol(z0)]
   return Dat(at, "dimer", E = E)
end

B_ = Bnew

dB = LsqDB("", B_, train);
A, y = IPFitting.Lsq.get_lsq_system(dB; Vref=Vref, weights=weights)

r1 = 2.2
w_core = 1e-2
r_core = range(0.1, r1, length=100)
A_core = zeros(3*length(r_core), length(B_))
y_core = zeros(3*length(r_core))
for (i, (r, (z1, z0))) in enumerate(Iterators.product(r_core, 
                                        [(zAl, zAl), (zAl, zTi), (zTi, zTi)]))
   dat = dat_dimer(r, z1, z0)
   w = (r - r1) + (r-r1)^3
   A_core[i, :] = w_core * w * energy(B_, dat.at)
   y_core[i] = w_core * w * dat.D["E"][1]
end

A1 = [A; A_core]
y1 = [y; y_core] 

A2 = A1 / Γ
y2 = y1 

using LowRankApprox

F = pqrfact(A2, rtol = 3e-6)
θ_til = F \ y2 
θ = Γ \ θ_til
IP = ACE1.combine(B_, θ)

@info("Test Error")
IPFitting.add_fits!(IP, test)
rmse_table(test)


##

zAl = AtomicNumber(:Al)
zTi = AtomicNumber(:Ti)
rr = range(0.1, rcut, length=200)

plt = plot(; ylims = [-10, 10], xlims = [0.0, 5.5])
# plot!(plt, rr, dimer_energy.(Ref(IP_old), rr, zAl, zAl), lw=2, c=1, ls=:dash, label = "AlAl-old")
plot!(plt, rr, dimer_energy.(Ref(IP), rr, zAl, zAl), lw=2, c=1, ls=:solid, label = "AlAl-new")
# plot!(plt, rr, dimer_energy.(Ref(IP_old), rr, zAl, zTi), lw=2, c=2, ls=:dash, label =  "AlTi-old")
plot!(plt, rr, dimer_energy.(Ref(IP), rr, zAl, zTi), lw=2, c=2, ls=:solid, label = "AlTi-new")
# plot!(plt, rr, dimer_energy.(Ref(IP_old), rr, zTi, zTi), lw=2, c=3, ls=:dash, label =  "TiTi-old")
plot!(plt, rr, dimer_energy.(Ref(IP), rr, zTi, zTi), lw=2, c=3, ls=:solid, label = "TiTi-new")
vline!(plt, [rnn(:Ti), rnn(:Al)], c=:black, lw=2, label = "rnn")

# plot!(rr, zbl.(rr, zAl, zAl))
# plot!(rr, evaluate.(Ref(envelope), rr))

function get_rdf(data, rcut)
   rdf = Dict((zAl, zAl) => Float64[], 
              (zAl, zTi) => Float64[], 
              (zTi, zAl) => Float64[], 
              (zTi, zTi) => Float64[])
   for dat in data 
      at = dat.at 
      nlist = neighbourlist(at, rcut)
      for (i, j, rr) in NeighbourLists.pairs(nlist)
         zi = at.Z[i]
         zj = at.Z[j]
         push!(rdf[(zi, zj)], sqrt(sum(rr.^2)))
      end
   end
   return rdf 
end

rdf = get_rdf(data, 5.5)

h1 = histogram(rdf[(zAl, zAl)], nbins=100, c=1, label = "rdf(Al, Al)", xlims = [0.0, 5.5])
h2 = histogram(rdf[(zAl, zTi)], nbins=100, c=2, label = "rdf(Al, Ti)", xlims = [0.0, 5.5])
h3 = histogram(rdf[(zTi, zTi)], nbins=100, c=3, label = "rdf(Ti, Ti)", xlims = [0.0, 5.5])

plot(plt, h1, h2, h3, 
     layout = Plots.GridLayout(4,1, heights=[0.7, 0.1, 0.1, 0.1]), 
     size = (600, 800))


## 

using PyCall
ARD = pyimport("sklearn.linear_model")["ARDRegression"]
ard_threshold_lambda = 10_000 
ard_tol = 1e-4
fit_intercept=true
ard_n_iter = 1000 

clf = ARD(n_iter=ard_n_iter, threshold_lambda = ard_threshold_lambda, tol=ard_tol, fit_intercept=fit_intercept, normalize=true, compute_score=true)
clf.fit(A2, y2)
c = Γ \ clf.coef_
IP_ard = ACE1.combine(B_, c)


##

zAl = AtomicNumber(:Al)
zTi = AtomicNumber(:Ti)
rr = range(0.1, rcut, length=200)

plt = plot(; ylims = [-10, 10], xlims = [0.0, 5.5])
# plot!(plt, rr, dimer.(Ref(IP_old), rr, zAl, zAl), lw=2, c=1, ls=:dash, label = "AlAl-old")
plot!(plt, rr, dimer.(Ref(IP_ard), rr, zAl, zAl), lw=2, c=1, ls=:solid, label = "AlAl-new")
# plot!(plt, rr, dimer.(Ref(IP_old), rr, zAl, zTi), lw=2, c=2, ls=:dash, label =  "AlTi-old")
plot!(plt, rr, dimer.(Ref(IP_ard), rr, zAl, zTi), lw=2, c=2, ls=:solid, label = "AlTi-new")
# plot!(plt, rr, dimer.(Ref(IP_old), rr, zTi, zTi), lw=2, c=3, ls=:dash, label =  "TiTi-old")
plot!(plt, rr, dimer.(Ref(IP_ard), rr, zTi, zTi), lw=2, c=3, ls=:solid, label = "TiTi-new")
vline!(plt, [rnn(:Ti), rnn(:Al)], c=:black, lw=2, label = "rnn")

rdf = get_rdf(data, 5.5)
h1 = histogram(rdf[(zAl, zAl)], nbins=100, c=1, label = "rdf(Al, Al)", xlims = [0.0, 5.5])
h2 = histogram(rdf[(zAl, zTi)], nbins=100, c=2, label = "rdf(Al, Ti)", xlims = [0.0, 5.5])
h3 = histogram(rdf[(zTi, zTi)], nbins=100, c=3, label = "rdf(Ti, Ti)", xlims = [0.0, 5.5])

plot(plt, h1, h2, h3, 
     layout = Plots.GridLayout(4,1, heights=[0.7, 0.1, 0.1, 0.1]), 
     size = (600, 800))


## 

Bpair = Bpair_new
Γpair = Diagonal(ACE1.scaling(Bpair, 3))

dB = LsqDB("", Bpair, train);
A, y = IPFitting.Lsq.get_lsq_system(dB; Vref=Vref, weights=weights)

r1 = 3.0
w_core = 1.0
r_core = range(0.1, r1, length=100)
A_core = zeros(3*length(r_core), length(Bpair))
y_core = zeros(3*length(r_core))
for (i, (r, (z1, z0))) in enumerate(Iterators.product(r_core, 
                                        [(zAl, zAl), (zAl, zTi), (zTi, zTi)]))
   dat = dat_dimer(r, z1, z0)
   w = (r - r1)^2
   A_core[i, :] = w_core * w * energy(Bpair, dat.at)
   y_core[i] = w_core * w * dat.D["E"][1]
end

A1 = [A; A_core]
y1 = [y; y_core] 

A2 = A1 / Γpair
y2 = y1 

using LowRankApprox

F = pqrfact(A2, rtol = 3e-6)
θ_til = F \ y2 
θ = Γpair \ θ_til
IP = ACE1.combine(Bpair, θ)

##

plt = plot(; ylims = [-10, 10], xlims = [0.0, 5.5])
# plot!(plt, rr, dimer.(Ref(IP_old), rr, zAl, zAl), lw=2, c=1, ls=:dash, label = "AlAl-old")
plot!(plt, rr, dimer.(Ref(IP), rr, zAl, zAl), lw=2, c=1, ls=:solid, label = "AlAl-new")
# plot!(plt, rr, dimer.(Ref(IP_old), rr, zAl, zTi), lw=2, c=2, ls=:dash, label =  "AlTi-old")
plot!(plt, rr, dimer.(Ref(IP), rr, zAl, zTi), lw=2, c=2, ls=:solid, label = "AlTi-new")
# plot!(plt, rr, dimer.(Ref(IP_old), rr, zTi, zTi), lw=2, c=3, ls=:dash, label =  "TiTi-old")
plot!(plt, rr, dimer.(Ref(IP), rr, zTi, zTi), lw=2, c=3, ls=:solid, label = "TiTi-new")
vline!(plt, [rnn(:Ti), rnn(:Al)], c=:black, lw=2, label = "rnn")
