

using ACE1
using Random, Printf, Test, LinearAlgebra, JuLIP, JuLIP.Testing
using JuLIP: evaluate, evaluate_d, evaluate_ed
using JuLIP.MLIPs: combine
using ACE1.Testing: println_slim


##

@info("======== Testing ACE1._cleanup ==============")

degrees = [ 0, 12, 10, 8, 8]
maxdeg = 20
rcut = 5.0

@info("Check original and cleaned-up basis are the same ")
for species in (:X, :Si, [:C, :H], [:C, :O, :H]), N = 2:length(degrees)
   @info("species = $species, N = $N")
   local Rs, Zs, z0, B, dB, basis, D, P1, Nat, h, V 
   D = SparsePSHDegree()
   trans = PolyTransform(1, 1.0)
   Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
   P1 = ACE1.BasicPSH1pBasis(Pr; species = species)
   basis1 = ACE1.RPIBasis(P1, N, D, degrees[N])
   cc = randn(length(basis1)) ./ (1:length(basis1)).^2
   pot1 = combine(basis1, cc)

   basis2 = ACE1._cleanup(basis1)

   Nat = 15
   for ntest = 1:20
      Rs, Zs, z0 = ACE1.rand_nhd(Nat, Pr, species)
      print_tf(@test(evaluate(basis1, Rs, Zs, z0) ≈ evaluate(basis2, Rs, Zs, z0)))
   end

   # pot2 = ACE1._cleanup(pot1)
   # for ntest = 1:20
   #    Rs, Zs, z0 = ACE1.rand_nhd(Nat, Pr, species)
   #    print_tf(@test(evaluate(pot1, Rs, Zs, z0) ≈ evaluate(pot2, Rs, Zs, z0)))
   # end

   println()
end

##

@warn("_cleanup for potential can sometimes fail?")
# species = [:Si,]
# D = SparsePSHDegree()
# trans = PolyTransform(1, 1.0)
# Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
# P1 = ACE1.BasicPSH1pBasis(Pr; species = species)
# basis1 = ACE1.RPIBasis(P1, 3, D, 16)

# cc = randn(length(basis1)) ./ (1:length(basis1)).^2
# pot1 = combine(basis1, cc)

# Nat = 15
# Rs, Zs, z0 = ACE1.rand_nhd(Nat, Pr, species)
# v1 = evaluate(pot1, Rs, Zs, z0)

# pot2 = ACE1._cleanup(pot1)
# v2 = evaluate(pot2, Rs, Zs, z0)

# v1 ≈ v2