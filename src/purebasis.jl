
using Combinatorics
import JuLIP: evaluate

struct PureBasis{TB}
   basis::TB
   Ibas::Vector{Int}
end

Base.length(b::PureBasis) = length(b.Ibas)

(b::PureBasis)(args...) = evaluate(b, args...)

PureBasis(ship::SHIPBasis) = PureBasis(ship, collect(1:length(ship)))

function PureBasis(ship::SHIPBasis, N::Integer)
   @assert length(ship.zlist) == 1
   Ibas = SHIPs.Utils.findall_basis_N(ship, N)
   return PureBasis(ship, Ibas)
end

function evaluate(b::PureBasis, Rs::AbstractVector)
   N = length(Rs)
   ship = b.basis
   @assert N <= 5
   @assert length(ship.zlist) == 1
   Zs = zeros(Int16, N)
   tmp = alloc_temp(ship, Rs)

   # one-particle basis
   phi = precompute_phi(tmp, Rs, Zs, ship)

   # permutation symmetrised tensor products
   Phi = precompute_prod_phi(tmp, phi, ship)

   # get B via the same rotation-symmetrisation as in the density trick case
   _my_mul!(tmp.Bc[1], ship.A2B[1], Phi)
   return real.(tmp.Bc[1])[b.Ibas]
end


function precompute_phi(tmp, Rs, Zs, ship)
   alist = ship.alists[1]
   phi = zeros(Complex{Float64}, length(alist), length(Rs))
   for (iR, (R, Z)) in enumerate(zip(Rs, Zs))
      # evaluate the r-basis and the R̂-basis for the current neighbour at R
      evaluate!(tmp.J, tmp.tmpJ, ship.J, norm(R))
      evaluate!(tmp.Y, tmp.tmpY, ship.SH, R)
      # write the products into phi
      iz = z2i(ship, Z)
      for i = alist.firstz[iz]:(alist.firstz[iz+1]-1)
         zklm = alist[i]
         phi[i, iR] = tmp.J[zklm.k+1] * tmp.Y[index_y(zklm.l, zklm.m)]
      end
   end
   return phi
end


function precompute_prod_phi(tmp, phi, ship)
   aalist = ship.aalists[1]
   N = size(phi, 2)
   factN = factorial(N)
   Phi = fill!(tmp.AA[1], 0)
   for i = 1:length(aalist)
      if aalist.len[i] != N; continue; end
      for σ in permutations(1:N)
         phi_temp = one(eltype(Phi))
         for α = 1:aalist.len[i]
            iphi = aalist.i2Aidx[i, α]
            phi_temp *= phi[iphi, σ[α]]
         end
         Phi[i] += phi_temp / factN
      end
   end
   return Phi
end
