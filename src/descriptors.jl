
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
      b = evaluate(basis, Rs, Zs, z0)
      bi = b[basis.Bz0inds[z2i(basis, z0)]]
      X[1:length(bi), i] = bi
   end
   return X
end

function descriptors_d(basis, at)
   len = maximum( length.(basis.Bz0inds) )
   dX = zeros(JVecF, len, length(at), length(at))
   nlist = neighbourlist(at, cutoff(basis))
   for i in 1:length(at)
      Js, Rs, Zs = neigsz(nlist, at, i)
      z0 = at.Z[i]
      db = evaluate_d(basis, Rs, Zs, z0)
      dbi = db[basis.Bz0inds[z2i(basis, z0)], :]
      leni = size(dbi, 1)
      for (ij, j) in enumerate(Js)
         dX[1:leni, i, j] = dbi[:, ij]
         dX[1:leni, i, i] -= dbi[:, ij]
      end
   end
   return dX
end

end