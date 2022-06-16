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
rin = 0.5 * r0 
trans = AgnesiTransform(; r0=r0, rin=rin, p = 3)

# rp = range(0.0, 2*r0; length=100)
# plot(rp, trans.(rp))

# trans = PolyTransform(; p = 2, r0 = r0)
envelope = ACE1.TwoSidedEnvelope(trans(rcut), trans(rin), 2, 2)
rbasis = transformed_jacobi_env(maxdeg, trans, envelope, rcut)

ACE_B = ace_basis(species = species,
                  N = 3,
                  r0 = r0,
                  rin = 0.7 * r0,
                  rcut = rcut,
                  pcut = 2, pin = 2, 
                  trans = trans,
                  # rbasis = rbasis, 
                  maxdeg = maxdeg);

# radial basis 
trans_r = AgnesiTransform(; r0=r0, p = 3)
envelope_r = ACE1.PolyEnvelope(2, r0, rcut)
Jnew = transformed_jacobi_env(12, trans_r, envelope_r, rcut)
Bpair_new = PolyPairBasis(Jnew, species)
Bnew = JuLIP.MLIPs.IPSuperBasis([Bpair_new, ACE_B]);

# reference potential (numbers from Cas' tutorial)
Vref = OneBody(:Ti => -1586.0195, :Al => -105.5954)

# regression weights (numbers from Cas' tutorial)
weights = Dict(
        "FLD_TiAl" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 ),
        "TiAl_T5000" => Dict("E" => 5.0, "F" => 1.0 , "V" => 1.0 ))


## fit a first potential

dB = LsqDB("", Bnew, train);
Γpair = ACE1.scaling(dB.basis.BB[1], 2)
Γace = ACE1.scaling(dB.basis.BB[2], 2)
Γ = Diagonal([10*Γpair; Γace])
# solver = Dict("solver" => :rrqr, "rrqr_tol" => 1e-7, "P" => Γ)
solver = Dict("solver" => :brr, "brr_tol" => 1e-3, "P" => Γ)
IP_new, lsqinfo_new = lsqfit(dB, solver=solver, weights=weights, Vref=Vref, error_table = true)
IPFitting.add_fits!(IP_new, test)

##

@info("Training Error - Old Basis")
rmse_table(lsqinfo_new["errors"])

@info("Test Error - New Basis")
IPFitting.add_fits!(IP_new, test)
rmse_table(test)



##

B_ = Bnew 

using JuLIP: evaluate 
using IPFitting: Dat 
zbl = JuLIP.ZBLPotential()

at_dimer(r, z1, z0) = Atoms(X = [ SVector(0.0,0.0,0.0), SVector(r, 0.0, 0.0)], 
                            Z = [z0, z1], pbc = false, 
                            cell = [r+1 0 0; 0 1 0; 0 0 1])

function dat_dimer(r, z1, z0) 
   at = at_dimer(r, z1, z0)
   E = ( ACE1.evaluate(envelope_r, r)
            + Vref.E0[chemical_symbol(z1)] + Vref.E0[chemical_symbol(z0)] )
   return Dat(at, "dimer", E = E)
end

zAl = AtomicNumber(:Al)
zTi = AtomicNumber(:Ti)

r00 = 1e-3 
e00 = evaluate(envelope_r, r00)
weights["dimer"] = Dict("E" => 100 / e00)

data_dimer = [ 
   dat_dimer(r00, zAl, zTi), 
   dat_dimer(r00, zAl, zAl), 
   dat_dimer(r00, zTi, zTi), ]

train_dimer = [data_dimer; train] 

dB = LsqDB("", B_, train_dimer);
solver = Dict("solver" => :rrqr, "rrqr_tol" => 1e-6, "P" => Γ)
IP_rep, lsqinfo_rep = lsqfit(dB, solver=solver, weights=weights, Vref=Vref, error_table = true)
IPFitting.add_fits!(IP_rep, test)

@info("Training Error - Old Basis")
rmse_table(lsqinfo_rep["errors"])

@info("Test Error - Repulsive Basis")
IPFitting.add_fits!(IP_rep, test)
rmse_table(test)

# ##

# solver = Dict("solver" => :brr, "brr_tol" => 1e-3, "P" => Γ)
# IP_rep, lsqinfo_rep = lsqfit(dB, solver=solver, weights=weights, Vref=Vref, error_table = true)
# IPFitting.add_fits!(IP_rep, test)

# @info("Training Error - Old Basis")
# rmse_table(lsqinfo_rep["errors"])

# @info("Test Error - Repulsive Basis")
# IPFitting.add_fits!(IP_rep, test)
# rmse_table(test)

##

solver = Dict("solver" => :ard, "ard_tol" => 1e-3, "ard_threshold_lambda" => 10000)
IP_rep, lsqinfo_rep = lsqfit(dB, solver=solver, weights=weights, Vref=Vref, error_table = true)
IPFitting.add_fits!(IP_rep, test)

@info("Training Error - Old Basis")
rmse_table(lsqinfo_rep["errors"])

@info("Test Error - Repulsive Basis")
IPFitting.add_fits!(IP_rep, test)
rmse_table(test)


##

IP = IP_new
IP = IP_rep

function dimer_energy(IP, r, z1, z0)
   at = at_dimer(r, z1, z0)
   at1 = Atoms(X = [SVector(0.0,0.0,0.0),], Z = [z1, ], pbc = false, cell = [1.0 0 0; 0 1.0 0; 0 0 1.0]) 
   at2 = Atoms(X = [SVector(0.0,0.0,0.0),], Z = [z0, ], pbc = false, cell = [1.0 0 0; 0 1.0 0; 0 0 1.0]) 
   return JuLIP.energy(IP, at) - JuLIP.energy(IP, at1) - JuLIP.energy(IP, at2)
end

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

rdf = get_rdf(data, 5.5)
h1 = histogram(rdf[(zAl, zAl)], nbins=100, c=1, label = "rdf(Al, Al)", xlims = [0.0, 5.5])
h2 = histogram(rdf[(zAl, zTi)], nbins=100, c=2, label = "rdf(Al, Ti)", xlims = [0.0, 5.5])
h3 = histogram(rdf[(zTi, zTi)], nbins=100, c=3, label = "rdf(Ti, Ti)", xlims = [0.0, 5.5])

plot(plt, h1, h2, h3, 
     layout = Plots.GridLayout(4,1, heights=[0.7, 0.1, 0.1, 0.1]), 
     size = (600, 800))
