
module Descriptors 

using JuLIP: neighbourlist, cutoff, JVecF 
using JuLIP.Potentials: neigsz, z2i, i2z 
using ACE1: evaluate, evaluate_d

export descriptors, descriptors_d 

"""
Returns a matrix `X` of size length(basis) x length(at) where each column 
`X[:, i]` contains the site-energy ACE basis for atom `i`, now interpreted
as a vector of invariant features for the atomic environment of atom `i`.
"""
function descriptors(basis, at)
   len = maximum( length.(basis.Bz0inds) )
   X = zeros(len, length(at))
   nlist = neighbourlist(at, cutoff(basis))
   for i in 1:length(at)
      Js, Rs, Zs = neigsz(nlist, at, i)
      z0 = at.Z[i]
      Iz0 = basis.Bz0inds[z2i(basis, z0)]
      # ----
      b = evaluate(basis, Rs, Zs, z0)
      bi = b[Iz0]
      X[1:length(bi), i] = bi
   end
   return X
end

"""
Jacobian of `descriptors` w.r.t. atomic positions. This returns a 
 `nfeatures` x `natoms` x `natoms` tensor `dX` where 
 `dX[i,j,k]` contains a 3-vector which is the derivative of 
 `X[i,j]` w.r.t. the position `at.X[k]`.
"""
function descriptors_d(basis, at)
   len = maximum( length.(basis.Bz0inds) )
   dX = zeros(JVecF, len, length(at), length(at))
   nlist = neighbourlist(at, cutoff(basis))
   for i in 1:length(at)
      Js, Rs, Zs = neigsz(nlist, at, i)
      z0 = at.Z[i]
      Iz0 = basis.Bz0inds[z2i(basis, z0)]
      leni = length(Iz0)
      # ----
      db = evaluate_d(basis, Rs, Zs, z0)
      dbi = db[Iz0, :]
      for a = 1:length(Js) 
         j = Js[a]
         dX[1:leni, i, j] += dbi[:, a]
         dX[1:leni, i, i] -= dbi[:, a]
      end
   end
   return dX
end

end