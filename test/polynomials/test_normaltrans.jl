

using ACE1, ForwardDiff, Test, ACEbase 
using ACE1.Transforms: normalized_agnesi_transform, transform_d
using ACE1.Testing: println_slim, print_tf

## 

@info("Testing normalized Agnesi Transform")
for (p, q) in [ (2, 4), (1, 4), (1, 3), (1,2), (2,2), (2,3), (3,3), (4,4)]
   local r0, trans 
   r0 = 1+rand()
   rcut = (2+rand()) * r0 
   trans = normalized_agnesi_transform(r0, rcut; p = p, q = q)
   ACE1.Testing.test_transform(trans, [0.0, rcut])
   print_tf(@test all(ACEbase.Testing.test_fio(trans)))
   print_tf(@test (     trans(0.0) ≈ 0.0 
                     && trans(rcut) ≈ 1.0
                     && transform_d(trans, rcut) ≈ 0.0))
end
println()



