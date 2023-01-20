
# --------------------------------------------------------------------------
# ACE1.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



# ----------------------------------------------------

using StaticArrays: SVector 
export PolyPairPot
import ACE1: write_committee, read_committee

struct PolyPairPot{T,TJ,NZ, NCO} <: PairPotential
   coeffs::Vector{T}
   basis::PolyPairBasis{TJ, NZ}
   committee::Union{Nothing, Vector{SVector{NCO, T}}}
end

@pot PolyPairPot

PolyPairPot(pB::PolyPairBasis, coeffs::Vector, committee=nothing) = 
            PolyPairPot(coeffs, pB, committee)

JuLIP.MLIPs.combine(pB::PolyPairBasis, coeffs::AbstractVector) =
            PolyPairPot(identity.(collect(coeffs)), pB, nothing)

function PolyPairPot(coeffs_t::Vector{T}, basis::PolyPairBasis{TJ, NZ}, 
                     committee::Nothing = nothing) where {TJ, NZ, T}
   return PolyPairPot{T, TJ, NZ, 0}(coeffs_t, basis, nothing)                      
end
               

JuLIP.cutoff(V::PolyPairPot) = cutoff(V.basis)

z2i(V::PolyPairPot, z::AtomicNumber) = z2i(V.basis, z)
i2z(V::PolyPairPot, iz::Integer) = i2z(V.basis, iz)
numz(V::PolyPairPot) = numz(V.basis)

==(V1::PolyPairPot, V2::PolyPairPot) =
            ( (V1.basis == V2.basis) && (V1.coeffs == V2.coeffs) )

write_dict(V::PolyPairPot{T}) where {T} = Dict(
      "__id__" => "ACE1_PolyPairPot",
      "T" => write_dict(T),
      "coeffs" => V.coeffs,
      "basis" => write_dict(V.basis), 
      "committee" => write_committee(V.committee)
      )

read_dict(::Val{:ACE1_PolyPairPot}, D::Dict, T = read_dict(D["T"])) =
      PolyPairPot(read_dict(D["basis"]), T.(D["coeffs"]), 
                  read_committee(D["committee"]))

import ACE1
ACE1.ncommittee(V::PolyPairPot{T, TJ, NZ, NCO}) where {T,TJ,NZ, NCO} = 
                  isnothing(V.committee) ? 0 : NCO 

# ----- allocation and evaluation codes       


alloc_temp(V::PolyPairPot{T}, N::Integer) where {T} =
      ( R = zeros(JVec{T}, N),
        Z = zeros(AtomicNumber, N),
        alloc_temp(V.basis)... )

alloc_temp_d(V::PolyPairPot{T}, N::Integer) where {T} =
      ( dV = zeros(JVec{T}, N),
         R = zeros(JVec{T}, N),
         Z = zeros(AtomicNumber, N),
         alloc_temp_d(V.basis)... )


function _dot_zij(V, B, z, z0)
   i0 = _Bidx0(V.basis, z, z0)  # cf. pair_basis.jl
   return sum( V.coeffs[i0 + n] * B[n]  for n = 1:length(V.basis, z, z0) )
end

function evaluate!(tmp, V::PolyPairPot, r::Number, z, z0) 
      Iz = z2i(V, z)
      Iz0 = z2i(V, z0)
      evaluate!(tmp.J[Iz, Iz0], tmp.tmp_J[Iz, Iz0], V.basis.J[Iz, Iz0], r, z, z0)
      return _dot_zij(V, tmp.J[Iz, Iz0], z, z0)
end

function evaluate_d!(tmp, V::PolyPairPot, r::Number, z, z0) 
      Iz = z2i(V, z)
      Iz0 = z2i(V, z0)
      evaluate_d!(tmp.J[Iz, Iz0], tmp.dJ[Iz, Iz0], tmp.tmpd_J[Iz, Iz0], 
                  V.basis.J[Iz, Iz0], r, z, z0)
      return _dot_zij(V, tmp.dJ[Iz, Iz0], z, z0)
end

function evaluate!(tmp, V::PolyPairPot, r::Number)
   @assert numz(V) == 1
   z = i2z(V, 1)
   return evaluate!(tmp, V, r, z, z)
end

function evaluate_d!(tmp, V::PolyPairPot, r::Number)
   @assert numz(V) == 1
   z = i2z(V, 1)
   return evaluate_d!(tmp, V, r, z, z)
end

evaluate(V::PolyPairPot, r::Number, args...) = evaluate!(alloc_temp(V, 1), V, r, args...)
evaluate_d(V::PolyPairPot, r::Number, args...) = evaluate_d!(alloc_temp_d(V, 1), V, r, args...)

