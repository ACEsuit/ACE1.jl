
using ACE1pack, ACE1, IPFitting, StaticArrays, Plots
using ACE1: transformed_jacobi, transformed_jacobi_env
using LinearAlgebra: Diagonal
using JuLIP: evaluate 
using IPFitting: Dat 


data_file = joinpath(ACE1pack.artifact("TiAl_tutorial"), "TiAl_tutorial.xyz")
data = IPFitting.Data.read_xyz(data_file, energy_key="energy", force_key="force", virial_key="virial")
train = data[1:2:end]
test = data[2:2:end]

##

species = [:Ti, :Al]
r0 = 2.88
rcut = 5.5 

# construct ace basis with Agnesi transform 
#    I take rin much smaller than usual but I shift the agnesi transform 
#    so that it becomes flat towards rin and so the choice of rin is no 
#    longer that important (I think!). The large p value ensures smoothness
#    at the inner cutoff. 
rin = 0.5 * r0 
trans = AgnesiTransform(; r0=r0, rin=rin, p = 3)

# plot this to see the effect of this transform choice.
# rp = range(0.0, 2*r0; length=100)
# plot(rp, trans.(rp))

maxdeg = 6
ace_b = ace_basis(species = species,
                  N = 3,
                  r0 = r0,
                  rin = rin, 
                  rcut = rcut,
                  pcut = 2, pin = 2, 
                  trans = trans,
                  maxdeg = maxdeg);

# Pair Potential Basis 
#     again I use an agnesi transform now with even more agressive smoothing 
#     My thinking here is that I want us to generalize this to a fourth order 
#     rational polynomial that allows us to independently control the smoothness 
#     at the inner cutoff and the decay at the outer cutoff. 
#     Or  -  even better  -   construct a spline transform based on an 
#     analysis of the radial distribution function.
trans_r = AgnesiTransform(; r0=r0, p = 4)
envelope_r = ACE1.PolyEnvelope(2, r0, rcut)
Jnew = transformed_jacobi_env(12, trans_r, envelope_r, rcut)
Bpair_new = PolyPairBasis(Jnew, species)

# Now we do the usual things ... 
# combine the ace and pair basis
basis = JuLIP.MLIPs.IPSuperBasis([Bpair_new, ace_b]);
# reference potential (numbers from Cas' tutorial)
Vref = OneBody(:Ti => -1586.0195, :Al => -105.5954)
# regression weights (numbers from Cas' tutorial)
weights = Dict(
        "FLD_TiAl" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 ),
        "TiAl_T5000" => Dict("E" => 5.0, "F" => 1.0 , "V" => 1.0 ))


## fit the first potential - just the new basis, no other modifications.
#       the errors aren't optimal, with a bit of hand-tuning we can reduce them 
#       by another 20% maybe with RRQR - see at the end of the script. 

dB = LsqDB("", basis, train);
solver = Dict("solver" => :ard, "ard_tol" => 1e-3, "ard_threshold_lambda" => 10000)
IP_new, lsqinfo_new = lsqfit(dB, solver=solver, weights=weights, Vref=Vref, error_table = true)

@info("Training Error - Old Basis")
IPFitting.add_fits!(IP_new, train)
rmse_table(train)

@info("Test Error - New Basis")
IPFitting.add_fits!(IP_new, test)
rmse_table(test)


##

# Now I want to add a MINIMUM amount of dimer data. The following functions 
# define a dimer `Dat` to add to the training set. In the limit as r -> 0 
# I want the dimer energy to be exactly the envelope + the one-body 
# contributions. 

at_dimer(r, z1, z0) = Atoms(X = [ SVector(0.0,0.0,0.0), SVector(r, 0.0, 0.0)], 
                            Z = [z0, z1], pbc = false, 
                            cell = [r+1 0 0; 0 1 0; 0 0 1])

function dat_dimer(r, z1, z0) 
   at = at_dimer(r, z1, z0)
   E = ( ACE1.evaluate(envelope_r, r)
            + Vref.E0[chemical_symbol(z1)] + Vref.E0[chemical_symbol(z0)] )
   return Dat(at, "dimer", E = E)
end

# Now the fun part is that I only add dimer data at ONE SINGLE POINT. 
# I really want to do this at r = 0.0, but for that I'd need to make some 
# changes. The RIGHT way to do it would be with a reference potential and 
# I'll try that next. But for now this is a morally equivalent hack that 
# seems to work well enough. 

# Morally what this achieves it he following: in the limit r -> 0 we have 
#     E = E0(z1) + E0(z2) + Vpair(r, z1, z2) 
# We defined Vpair(r, z1, z2) = Env(r) * p(r, z1, z2). Now all we need to do 
# is require that p(0, z1, z2) = 1. Then we obtain that 
#    E(dimer) ~ Env(r)   as r -> 0.
# For for each z1, z2 pair we need to add one single data point. And this 
# gives the model maximal flexibility away from r = 0. We could of course 
# as for a difference asymptotic behaviour (zbl?) but this is less robust 
# because it is trying to force the polynomial to do something non-trivial. 
# If we want this then we need to build ZBL into the envelope which we can 
# of course do. 

zAl = AtomicNumber(:Al)
zTi = AtomicNumber(:Ti)

r00 = 1e-3 # this is morally 0.0 (as I said - hack - to be improved)
e00 = evaluate(envelope_r, r00)
weights["dimer"] = Dict("E" => 100 / e00)  
# the 100 above is just a high-ish weight to ensure that the p(0) ≈ 1 is 
# satisfied to very high precision. 

data_dimer = [ 
   dat_dimer(r00, zAl, zTi), 
   dat_dimer(r00, zAl, zAl), 
   dat_dimer(r00, zTi, zTi), ]

train_dimer = [data_dimer; train] 

##

dB = LsqDB("", basis, train_dimer);
solver = Dict("solver" => :ard, "ard_tol" => 1e-3, "ard_threshold_lambda" => 10000)
IP_rep, lsqinfo_rep = lsqfit(dB, solver=solver, weights=weights, Vref=Vref, error_table = true)
IPFitting.add_fits!(IP_rep, test)

@info("Training Error - Repulsive Basis")
IPFitting.add_fits!(IP_rep, train)
rmse_table(train)

@info("Test Error - Repulsive Basis")
IPFitting.add_fits!(IP_rep, test)
rmse_table(test)


##

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
rr = range(0.0001, rcut, length=200)

plt = plot(; ylims = [-10, 10], xlims = [0.0, 5.5])
plot!(plt, rr, dimer_energy.(Ref(IP_new), rr, zAl, zAl), lw=2, c=1, ls=:dash, label = "AlAl-old")
plot!(plt, rr, dimer_energy.(Ref(IP_rep), rr, zAl, zAl), lw=2, c=1, ls=:solid, label = "AlAl-new")
plot!(plt, rr, dimer_energy.(Ref(IP_new), rr, zAl, zTi), lw=2, c=2, ls=:dash, label =  "AlTi-old")
plot!(plt, rr, dimer_energy.(Ref(IP_rep), rr, zAl, zTi), lw=2, c=2, ls=:solid, label = "AlTi-new")
plot!(plt, rr, dimer_energy.(Ref(IP_new), rr, zTi, zTi), lw=2, c=3, ls=:dash, label =  "TiTi-old")
plot!(plt, rr, dimer_energy.(Ref(IP_rep), rr, zTi, zTi), lw=2, c=3, ls=:solid, label = "TiTi-new")
vline!(plt, [rnn(:Ti), rnn(:Al)], c=:black, lw=2, label = "rnn")

rdf = get_rdf(data, 5.5)
h1 = histogram(rdf[(zAl, zAl)], nbins=100, c=1, label = "rdf(Al, Al)", xlims = [0.0, 5.5])
h2 = histogram(rdf[(zAl, zTi)], nbins=100, c=2, label = "rdf(Al, Ti)", xlims = [0.0, 5.5])
h3 = histogram(rdf[(zTi, zTi)], nbins=100, c=3, label = "rdf(Ti, Ti)", xlims = [0.0, 5.5])

plot(plt, h1, h2, h3, 
     layout = Plots.GridLayout(4,1, heights=[0.7, 0.1, 0.1, 0.1]), 
     size = (600, 800))


## 

# alternative solver parameters - I like to make the regularisationon the 
# pair potential a bit stronger. But with ARD we don't need any of this of course. 
# Γpair = ACE1.scaling(dB.basis.BB[1], 2)
# Γace = ACE1.scaling(dB.basis.BB[2], 2)
# Γ = Diagonal([10*Γpair; Γace])
# solver = Dict("solver" => :rrqr, "rrqr_tol" => 1e-7, "P" => Γ)
# solver = Dict("solver" => :brr, "brr_tol" => 1e-3, "P" => Γ)

