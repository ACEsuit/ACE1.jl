
using Test 
using ACE1.Testing: print_tf, println_slim
using ACE1, LinearAlgebra, Interpolations, BenchmarkTools, ForwardDiff
using ACE1.Transforms: multitransform
using JuLIP: evaluate, evaluate!, evaluate_d, evaluate_d!

##


cutoffs = Dict(
   (:Fe, :C) => (1.5, 5.0), 
   (:C, :Al) => (0.7, 6.0), 
   (:Fe, :Al) => (2.2, 4.5), 
   (:Fe, :Fe) => (2.0, 5.0), 
   (:Al, :Al) => (2.0, 5.0), 
   (:C, :C) => (1.5, 5.2), 
   (:Al, :Fe) => (1.5, 5.0)  )

transforms = Dict(
      (:Fe, :C) => PolyTransform(2, (rnn(:Fe)+rnn(:C)) / 2), 
      (:C, :Al) => PolyTransform(2, (rnn(:Al)+rnn(:C)) / 2), 
      (:Fe, :Al) => PolyTransform(2, (rnn(:Al)+rnn(:Fe)) / 2), 
      (:Fe, :Fe) => PolyTransform(2, rnn(:Fe)), 
      (:Al, :Al) => PolyTransform(2, rnn(:Al)), 
      (:C, :C) => PolyTransform(2, rnn(:C)), 
      (:Al, :Fe) => PolyTransform(2, (rnn(:Al)+rnn(:Fe)) / 2 + 1)  
    )
 

mtrans = multitransform(transforms, cutoffs=cutoffs)

maxdeg = 8
N = 3
Pr = transformed_jacobi(maxdeg, mtrans; pcut = 2)

Sr = ACE1.Splines.RadialSplines(Pr; nnodes=4_000)

##

@info("test fio")
println_slim(@test all( JuLIP.Testing.test_fio(Sr) ))

##

zFe = AtomicNumber(:Fe)
zC = AtomicNumber(:C)
zAl = AtomicNumber(:Al)

@info("Test radial basis accuracy")
npass = 0 
d_npass = 0 
for ntest = 1:100
   local r, z, z0, p, s, dp, ds 
   global npass, d_npass
   r = 2.5 + 3 * rand() 
   z = rand([zFe, zC, zAl])
   z0 = rand([zFe, zC, zAl])

   p = evaluate(Pr, r, z, z0)
   s = evaluate(Sr, r, z, z0)
   dp = evaluate_d(Pr, r, z, z0)
   ds = evaluate_d(Sr, r, z, z0)

   npass += norm(p - s, Inf) < 1e-9
   d_npass += norm(dp - ds, Inf) < 1e-5
end
print("  npass = $npass / 100: "); println_slim(@test npass > 95)
print("d_npass = $d_npass / 100: "); println_slim(@test d_npass > 95)

##

# r = 2.5 + 3 * rand() 
# z = rand([zFe, zC, zAl])
# z0 = rand([zFe, zC, zAl])

# p = evaluate(Pr, r, z, z0)
# dp = evaluate_d(Pr, r, z, z0)
# s = evaluate(Sr, r, z, z0)
# ds = evaluate_d(Sr, r, z, z0)

# @btime evaluate!($p, nothing, $Pr, $r, $z, $z0)
# @btime evaluate_d!($p, $dp, nothing, $Pr, $r, $z, $z0)
# @btime evaluate!($s, nothing, $Sr, $r, $z, $z0)
# @btime evaluate_d!($s, $ds, nothing, $Sr, $r, $z, $z0)


##

@info("Test basis evaluation")
N = 3
D = SparsePSHDegree()
B1p_P = BasicPSH1pBasis(Pr; species = [:Fe, :Al, :C], D = D)
B1p_S = BasicPSH1pBasis(Sr; species = [:Fe, :Al, :C], D = D)

rpibasis_P = RPIBasis(B1p_P, N, D, maxdeg)
rpibasis_S = RPIBasis(B1p_S, N, D, maxdeg)

npass = 0 
d_npass = 0

for ntest = 1:100 
   local Nat, Rs, Zs, z0 
   global npass, d_npass
   Nat = 12 
   Rs = [ (2 + rand() * 3) * ACE1.Random.rand_sphere() for _ = 1:Nat ]
   Zs = rand([zFe, zC, zAl], Nat)
   z0 = rand([zFe, zC, zAl])

   B_P = evaluate(rpibasis_P, Rs, Zs, z0)
   B_S = evaluate(rpibasis_S, Rs, Zs, z0)

   dB_P = evaluate_d(rpibasis_P, Rs, Zs, z0)
   dB_S = evaluate_d(rpibasis_S, Rs, Zs, z0)

   npass += (norm(B_P - B_S, Inf) / Nat < 1e-7)
   d_npass += (norm(dB_P - dB_S, Inf) / Nat < 1e-4)
end

print("  npass = $npass / 100: "); println_slim(@test npass > 95)
print("d_npass = $d_npass / 100: "); println_slim(@test d_npass > 95)

