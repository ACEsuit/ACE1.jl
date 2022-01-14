


using JuLIP, ACE1, LinearAlgebra

#---

module ParamTest

using ACE1, JuLIP, LinearAlgebra
import ACE1: alloc_temp
import ACE1.PairPotentials: PolyPairBasis
import ACE1.OrthPolys: TransformedPolys
import JuLIP: energy, evaluate!

set_params(trans::PolyTransform, params) =
          PolyTransform(params[1], params[2])

set_params(J::TransformedPolys, params) =
          TransformedPolys(J.J,
                           set_params(J.trans, params),
                           J.rl, J.ru)

set_params(pB::PolyPairBasis, params) =
         PolyPairBasis( set_params(pB.J, params),
                        pB.zlist, pB.bidx0 )

function energy(params::AbstractVector{TP}, pB::PolyPairBasis, at::Atoms{T}) where {T, TP}
   E = zeros(TP, length(pB))
   pB_p = set_params(pB, params)
   # return energy(pB_p, at)   #  <---- goal for later ???!!!
   tmp = alloc_temp(pB_p)
   tmpJ = zeros(TP, length(pB_p))     # HACK
   for (i, j, R) in pairs(at, cutoff(pB_p))
      r = norm(R)
      evaluate!(tmpJ, tmp.tmp_J, pB_p.J, r)
      idx0 = ACE1.PairPotentials._Bidx0(pB_p, at.Z[i], at.Z[j])
      for n = 1:length(pB_p.J)
         E[idx0 + n] += 0.5 * tmpJ[n]
      end
   end
   return E
end

end

#---

pB = ACE1.Utils.pair_basis(; species=:W, trans = PolyTransform(2.0, 2.7))

at = bulk(:W, cubic=true) * 3
B1 = energy(pB, at)

params = [2.1, 2.6]
B2 = energy(params, pB, at)

using ForwardDiff
DB = ForwardDiff.jacobian( p -> energy(p, pB, at), params)

dp = rand(2) .= 0.5
for p = 2:10
   h = 0.1^p
   Bh = energy(params + h * dp, pB, at)
   err = norm((Bh - B2) / h - DB * dp, Inf)
   @show err
end
