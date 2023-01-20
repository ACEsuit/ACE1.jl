


# --------------------------------------------------------------------------
# ACE1.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


#---


using ACE1, JuLIP, Test
using ACE1: combine

#---

@info("Testing :Any Potential to evaluate energy, forces, virial")
basis = ACE1.Utils.rpi_basis(species = :Any, N = 3, r0 = 2.7, rcut = 5.0, maxdeg = 8)
c = ACE1.Random.randcoeffs(basis)
V = combine(basis, c)

#---

at1 = rattle!(bulk(:Fe, cubic=true) * (2,2,2), 0.1)
at2 = deepcopy(at1); at2.Z[:] .= AtomicNumber(:Al)
at3 = deepcopy(at1); at2.Z[1:3:end] .= AtomicNumber(:Al)

for f in (energy, forces, virial)
   println_slim(@test f(V, at1) ≈ f(V, at2) ≈ f(V, at3))
end

#---

@info(" ... check that a non-Any potential will fail this test")
basis1 = ACE1.Utils.rpi_basis(species = :Fe, N = 3, r0 = 2.7, rcut = 5.0, maxdeg = 8)
V1 = combine(basis1, c)   # same coefficients

println_slim(@test energy(V1, at1) ≈ energy(V, at1))
for at in (at2, at3)
   println_slim(@test (
      try
         energy(V1, at2)
         false
      catch
         true
      end
   ))
end


#---

