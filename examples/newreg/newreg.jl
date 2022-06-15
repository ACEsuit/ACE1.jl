
using ACE1, LinearAlgebra, Plots
using ACE1: evaluate

r0, rcut = 2.7, 5.5 
trans = AgnesiTransform(r0)
envelope = ACE1.PolyEnvelope(2, r0, rcut)
Jold = transformed_jacobi(10, trans, rcut)
Jnew = transformed_jacobi_env(10, trans, envelope, rcut)

Γ3 = Diagonal((1:length(Jold)).^3)
Γ0 = Diagonal((1:length(Jold)).^0)

##

"""
This defines a rdf with a peak at r0 and another peak at 2r0 
"""
function r_sample(r0, rcut, σ = 6.0, rin = 0.85*r0)
   r = -1.0 
   while r < rin || r > rcut 
      f = rand([1, 2, 2, 2, 2])
      r = f * r0 + f * randn() / σ
   end
   return r 
end

# we can see here what this distribution looks like. It is supposed 
# to mimic a realistic RDF. 
R = [r_sample(r0, rcut) for i = 1:10_000]
Plots.histogram(R, nbins=50, yscale = :log10)


## 
# a "proper" training set - we will fit the potential to averages 
# which corresponds more or less to total energies
Rs = [ [ r_sample(r0, rcut) for i = 1:rand(20:40) ] for _ = 1:15 ]
rdf = vcat(Rs...)


mean(J, R::AbstractVector{<: Number}) = 
         sum(evaluate(J, r) for r in R) / length(R)

"""
A baby lsq fit of a pair potential to mean data. 
"""
function lsq_fit(f, Rs, J, Γ)
   A = zeros(length(Rs), length(J))
   y = zeros(length(Rs))
   for i = 1:length(Rs)
      R = Rs[i]
      A[i, :] = mean(J, R)
      y[i] = mean(f, R)
   end
   A = [A; Γ]
   y = [y; zeros(length(J))]
   θ = qr(A) \ y 
   return r -> dot(θ, evaluate(J, r))
end

##

Rs = [ [ r_sample(r0, rcut) for i = 1:rand(20:40) ] for _ = 1:100 ]
rdf = vcat(Rs...)
xp = range(0.1, rcut, length=300)

g_target(r) = exp(-10*(r/r0-1)) - 2* exp(-5*(r/r0-1))
f_target(r) = g_target(r) - g_target(rcut)
ACE1.evaluate(::typeof(f_target), args...) = f_target(args...)

plot(xp, f_target.(xp),  lw=2, label = "target", legend = :topleft, 
      ylims=[-1.2, 300.0], )

V_old = lsq_fit(f_target, Rs, Jold, 0.0002*Γ3)

plot!(xp, V_old.(xp), lw=2, label = "old")

V_new = lsq_fit(f_target, Rs, Jnew, 0.0002*Γ3)
plt = plot!(xp, V_new.(xp), lw=2, label = "new")

plt1 = plot!(deepcopy(plt); ylims=[-1.0, 0.2])

plot(plt, plt1, histogram(rdf, nbins=30, xlims = [0.1, rcut], label = "rdf", legend = :topleft), 
     layout = grid(3, 1, heights=[0.5, 0.4, 0.1]), 
     size = (800, 800))



##

using ACE1pack
tial_set = ACE1pack.artifact("Ti