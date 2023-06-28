

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
   Nat = 15
   D = SparsePSHDegree()
   trans = PolyTransform(1, 1.0)
   Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
   P1 = ACE1.BasicPSH1pBasis(Pr; species = species)
   basis1 = ACE1.RPIBasis(P1, N, D, degrees[N])
   basis2 = ACE1._cleanup(basis1)

   for ntest = 1:30
      Rs, Zs, z0 = ACE1.rand_nhd(Nat, Pr, species)
      print_tf(@test(evaluate(basis1, Rs, Zs, z0) ≈ evaluate(basis2, Rs, Zs, z0)))
   end
   println()
end

