
# --------------------------------------------------------------------------
# ACE1.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



module OrthPolys

using SparseArrays
using LinearAlgebra: dot

import JuLIP: evaluate!, evaluate_d!, JVec, cutoff, fltype
import JuLIP.FIO: read_dict, write_dict
import JuLIP.MLIPs: alloc_B, alloc_dB, IPBasis

import ACE1
using ACE1.Transforms: DistanceTransform, transform, transform_d,
                        inv_transform, MultiTransform

import Base: ==

export transformed_jacobi, transformed_jacobi_env, PolyEnvelope, TwoSidedEnvelope

# this is a hack to prevent a weird compiler error that I don't understand
# at all yet
___f___(D::Dict) = (@show D)

function _fcut_(pl, tl, pr, tr, t)
   if (pl > 0 && t < tl) || (pr > 0 && t > tr)
      return zero(t)
   end
   return (t - tl)^pl * (t - tr)^pr
end

function _fcut_d_(pl, tl, pr, tr, t)
   if (pl > 0 && t < tl) || (pr > 0 && t > tr)
      return zero(t)
   end
   df = 0.0
   if pl > 0; df += pl * (t - tl)^(pl-1) * (t-tr )^pr ; end
   if pr  > 0; df += pr  * (t -  tr)^(pr -1) * (t-tl)^pl; end
   return df
end


struct OrthPolyBasis{T} <: ACE1.ScalarBasis{T}
   # ----------------- the parameters for the cutoff function
   pl::Int        # cutoff power left
   tl::T          # cutoff left (transformed variable)
   pr::Int        # cutoff power right
   tr::T          # cutoff right (transformed variable)
   # ----------------- the recursion coefficients
   A::Vector{T}
   B::Vector{T}
   C::Vector{T}
   # ----------------- used only for construction ...
   #                   but useful to have since it defines the notion or orth.
   tdf::Vector{T}
   ww::Vector{T}
end

fltype(P::OrthPolyBasis{T}) where {T} = T
Base.length(P::OrthPolyBasis) = length(P.A)

==(J1::OrthPolyBasis, J2::OrthPolyBasis) =
      all( getfield(J1, sym) == getfield(J2, sym)
           for sym in (:pr, :tr, :pl, :tl, :A, :B, :C) )

write_dict(J::OrthPolyBasis{T}) where {T} = Dict(
      "__id__" => "ACE1_OrthPolyBasis",
      "T" => write_dict(T),
      "pr" => J.pr,
      "tr" => J.tr,
      "pl" => J.pl,
      "tl" => J.tl,
      "A" => J.A,
      "B" => J.B,
      "C" => J.C
   )

OrthPolyBasis(D::Dict, T=read_dict(D["T"])) =
   OrthPolyBasis(
      D["pl"], D["tl"], D["pr"], D["tr"],
      Vector{T}(D["A"]), Vector{T}(D["B"]), Vector{T}(D["C"]),
      T[], T[]
   )

read_dict(::Val{:ACE1_OrthPolyBasis}, D::Dict) = OrthPolyBasis(D)

# rand applied to a J will return a random transformed distance drawn from
# the measure w.r.t. which the polynomials were constructed.
# TODO: allow non-constant weights!
function ACE1.rand_radial(J::OrthPolyBasis)
   @assert maximum(abs, diff(J.ww)) == 0
   return rand(J.tdf)
end

function OrthPolyBasis(N::Integer,
                       pcut::Integer,
                       tcut::T,
                       pin::Integer,
                       tin::T,
                       tdf::AbstractVector{T},
                       ww::AbstractVector{T} = ones(T, length(tdf))
                       ) where {T <: AbstractFloat}
   @assert pcut >= 0  && pin >= 0
   @assert N > 2

   if tcut < tin
      tl, tr = tcut, tin
      pl, pr = pcut, pin
   else
      tl, tr = tin, tcut
      pl, pr = pin, pcut
   end

   if minimum(tdf) < tl || maximum(tdf) > tr
      @warn("OrthoPolyBasis: t range outside [tl, tr]")
   end

   A = zeros(T, N)
   B = zeros(T, N)
   C = zeros(T, N)

   # normalise the weights s.t. <1, 1> = 1
   ww = ww ./ sum(ww)
   # define inner products
   dotw = (f1, f2) -> dot(f1, ww .* f2)

   # start the iteration
   _J1 = _fcut_.(pl, tl, pr, tr, tdf)
   a = sqrt( dotw(_J1, _J1) )
   A[1] = 1/a
   J1 = A[1] * _J1

   # a J2 = (t - b) J1
   b = dotw(tdf .* J1, J1)
   _J2 = (tdf .- b) .* J1
   a = sqrt( dotw(_J2, _J2) )
   A[2] = 1/a
   B[2] = -b / a
   J2 = (A[2] * tdf .+ B[2]) .* J1

   # keep the last two for the 3-term recursion
   Jprev = J2
   Jpprev = J1

   for n = 3:N
      # a Jn = (t - b) J_{n-1} - c J_{n-2}
      b = dotw(tdf .* Jprev, Jprev)
      c = dotw(tdf .* Jprev, Jpprev)
      _J = (tdf .- b) .* Jprev -c * Jpprev
      a = sqrt( dotw(_J, _J) )
      A[n] = 1/a
      B[n] = - b / a
      C[n] = - c / a
      Jprev, Jpprev = _J / a, Jprev
   end

   return OrthPolyBasis(pl, tl, pr, tr, A, B, C, collect(tdf), collect(ww))
end

alloc_B( J::OrthPolyBasis{T}) where {T} = zeros(T, length(J))
alloc_dB(J::OrthPolyBasis{T}) where {T} = zeros(T, length(J))
alloc_B( J::OrthPolyBasis{T}, ::Integer) where {T} = zeros(T, length(J))
alloc_dB(J::OrthPolyBasis{T}, ::Integer) where {T} = zeros(T, length(J))

# TODO: revisit this to allow type genericity!!!
alloc_B( J::OrthPolyBasis, ::TX) where {TX <: Number} = zeros(TX, length(J))
alloc_dB(J::OrthPolyBasis, ::TX) where {TX <: Number} = zeros(TX, length(J))
alloc_B( J::OrthPolyBasis,
         ::Union{JVec{TX}, AbstractVector{JVec{TX}}}) where {TX} =
   zeros(TX, length(J))
alloc_dB( J::OrthPolyBasis,
         ::Union{JVec{TX}, AbstractVector{JVec{TX}}}) where {TX} =
   zeros(TX, length(J))

function evaluate!(P, tmp, J::OrthPolyBasis, t; maxn=length(J))
   @assert length(P) >= maxn
   P[1] = J.A[1] * _fcut_(J.pl, J.tl, J.pr, J.tr, t)
   if maxn == 1; return P; end
   P[2] = (J.A[2] * t + J.B[2]) * P[1]
   if maxn == 2; return P; end
   @inbounds for n = 3:maxn
      P[n] = (J.A[n] * t + J.B[n]) * P[n-1] + J.C[n] * P[n-2]
   end
   return P
end

function evaluate_ed!(P, dP, tmp, J::OrthPolyBasis, t; maxn=length(J))
   @assert maxn <= min(length(P), length(dP))

   P[1] = J.A[1] * _fcut_(J.pl, J.tl, J.pr, J.tr, t)
   dP[1] = J.A[1] * _fcut_d_(J.pl, J.tl, J.pr, J.tr, t)
   if maxn == 1; return dP; end

   α = J.A[2] * t + J.B[2]
   P[2] = α * P[1]
   dP[2] = α * dP[1] + J.A[2] * P[1]
   if maxn == 2; return dP; end

   @inbounds for n = 3:maxn
      α = J.A[n] * t + J.B[n]
      P[n] = α * P[n-1] + J.C[n] * P[n-2]
      dP[n] = α * dP[n-1] + J.C[n] * dP[n-2] + J.A[n] * P[n-1]
   end
   return P, dP
end

evaluate_d!(P, dP, tmp, J::OrthPolyBasis, t; maxn=length(J)) = 
         evaluate_ed!(P, dP, tmp, J, t; maxn=maxn)[2]

"""
`discrete_jacobi(N; pcut=0, tcut=1.0, pin=0, tin=-1.0, Nquad = 1000)`

A utility function to generate a jacobi-type basis
"""
function discrete_jacobi(N; pcut=0, tcut=1.0, pin=0, tin=-1.0, Nquad = 1000)
   tl, tr = minmax(tin, tcut)
   dt = (tr - tl) / Nquad
   tdf = range(tl + dt/2, tr - dt/2, length=Nquad)
   return OrthPolyBasis(N, pcut, tcut, pin, tin, tdf)
end


# ----------------------------------------------------------------
#   Transformed Polynomials Basis
# ----------------------------------------------------------------

struct TransformedPolys{T, TT, TJ, TENV} <: ACE1.ScalarBasis{T}
   J::TJ          # the actual basis
   trans::TT      # coordinate transform
   rl::T          # lower bound r
   ru::T          # upper bound r = rcut
   envelope::TENV
end

==(J1::TransformedPolys, J2::TransformedPolys) = (
   (J1.J == J2.J) &&
   (J1.trans == J2.trans) &&
   (J1.rl == J2.rl) &&
   (J1.ru == J2.ru) && 
   (J1.envelope == J2.envelope) )

function TransformedPolys(J, trans, rl, ru, env = OneEnvelope())
   # get the combined type if rl, ru are different 
   T = promote_type(typeof(rl), typeof(ru))
   # integer is not allowed and we then default to float64 
   if !(T <: AbstractFloat)
      T = Float64 
   end 
   rl_ = convert(T, rl) 
   ru_ = convert(T, ru)
   return TransformedPolys(J, trans, rl_, ru_, env)
end

write_dict(J::TransformedPolys) = Dict(
      "__id__" => "ACE1_TransformedPolys",
      "J" => write_dict(J.J),
      "rl" => J.rl,
      "ru" => J.ru,
      "trans" => write_dict(J.trans), 
      "envelope" => write_dict(J.envelope),
   )

TransformedPolys(D::Dict) =
   TransformedPolys(
      read_dict(D["J"]),
      read_dict(D["trans"]),
      D["rl"],
      D["ru"],
      read_dict(D["envelope"]), 
   )

read_dict(::Val{:ACE1_TransformedPolys}, D::Dict) = TransformedPolys(D)

Base.length(J::TransformedPolys) = length(J.J)
fltype(P::TransformedPolys{T}) where {T} = T

function ACE1.rand_radial(J::TransformedPolys, z1, z2)
   t = ACE1.rand_radial(J.J)
   return inv_transform(J.trans, t, z1, z2)
end

function ACE1.rand_radial(J::TransformedPolys)
   t = ACE1.rand_radial(J.J)
   return inv_transform(J.trans, t)
end

cutoff(J::TransformedPolys) = J.ru

alloc_B( J::TransformedPolys, args...) = alloc_B(J.J, args...)
alloc_dB(J::TransformedPolys) = alloc_dB(J.J)
alloc_dB(J::TransformedPolys, N::Integer) = alloc_dB(J.J)

# in evaluate! and evaluate_d!: args... can be nothing or z, z0 

function evaluate!(P, tmp, J::TransformedPolys, r, args...; maxn=length(J))
   # transform coordinates
   t = transform(J.trans, r, args...)
   # evaluate the actual polynomials
   evaluate!(P, nothing, J.J, t; maxn=maxn)
   e = evaluate(J.envelope, r)
   @. P *= e
   return P
end


function evaluate_d!(P, dP, tmp, J::TransformedPolys, r, args...; maxn=length(J))
   # transform coordinates
   t = transform(J.trans, r, args...)::Float64
   dt = transform_d(J.trans, r, args...)::Float64
   # evaluate the actual Jacobi polynomials + derivatives w.r.t. x
   evaluate_ed!(P, dP, nothing, J.J, t, maxn=maxn)
   e = evaluate(J.envelope, J.trans, r)
   de = evaluate_d(J.envelope, J.trans, r)
   @. dP = de * P + e * dP * dt 
   # dP[:] .= de * P + e * dP * dt
   return dP
end


"""
`transformed_jacobi(maxdeg, trans, rcut, rin = 0.0; kwargs...)` : construct
a `TransformPolys` basis with an inner polynomial basis of `OrthPolys` type.

* `maxdeg` : maximum degree
* `trans` : distance transform; normally `PolyTransform(...)`
* `rin, rcut` : inner and outer cutoff

**Keyword arguments:**

* `pcut = 2` : cutoff parameter
* `pin = 0` : inner cutoff parameter
* `Nquad = 1000` : number of quadrature points
"""
function transformed_jacobi(maxdeg::Integer,
                            trans::DistanceTransform,
                            rcut::Real, rin::Real = 0.0;
                            kwargs...)
   J =  discrete_jacobi(maxdeg; tcut = transform(trans, rcut),
                                tin = transform(trans, rin),
                                pcut = 2,
                                kwargs...)
   return TransformedPolys(J, trans, rin, rcut)
end



function transformed_jacobi(maxdeg::Integer,
                           trans::MultiTransform; 
                           pcut = 2, 
                           kwargs...)
   # this construction can only work if the transforms are then 
   # sent to a single unified domain. 
   @assert eltype(trans.transforms) <: ACE1.Transforms.AffineT
   @assert all( (t.y1 == -1) && (t.y2 == 1) 
                 for t in trans.transforms )
   # obtain the maximum outer cutoff and minimum inner cutoff 
   rin, rcut = ACE1.Transforms.cutoff_extrema(trans)
   # now construct the orthogonal polynomials with the [-1,1] domain. 
   J =  discrete_jacobi(maxdeg; tcut = 1.0, tin = -1.0, 
                                pcut = pcut, kwargs...)
   return TransformedPolys(J, trans, rin, rcut)
end



# -------------------- Envelopes 

import ACE1: evaluate, evaluate_d
import ForwardDiff

abstract type AbstractEnvelope end 

abstract type AbstractEnvelopeX <: AbstractEnvelope end 

abstract type AbstractEnvelopeR <: AbstractEnvelope end 

evaluate(env::AbstractEnvelopeR, trans, r::Real) = evaluate(env, r) 

evaluate_d(env::AbstractEnvelopeR, trans, r::Real) = evaluate_d(env, r) 

evaluate(env::AbstractEnvelopeX, trans, r::Real) = 
      evaluate(env, transform(trans, r)) 

evaluate_d(env::AbstractEnvelopeX, trans, r::Real) = 
      evaluate_d(env, transform(trans, r)) * transform_d(trans, r)

evaluate_d(env::AbstractEnvelope, r::Real) = 
      ForwardDiff.derivative( r1 -> evaluate(env, r1), r )

struct OneEnvelope <: AbstractEnvelopeR
end

evaluate(env::OneEnvelope, r::T) where {T <: Real} = one(T)

write_dict(env::OneEnvelope) = Dict(
      "__id__" => "ACE1_OneEnvelope",
   )

read_dict(::Val{:ACE1_OneEnvelope}, D::Dict) = OneEnvelope()


struct PolyEnvelope{T} <: AbstractEnvelopeR
   p::Int
   r0::T 
   rcut::T
end

function evaluate(env::PolyEnvelope, r::T) where {T <: Real}
   p, r0, rcut = env.p, env.r0, env.rcut
   if r > rcut; return 0.0; end
   s = r/r0; scut = rcut/r0 
   return s^(-p) - scut^(-p) + p * scut^(-p-1) * (s - scut)
end

write_dict(env::PolyEnvelope{T}) where {T} = Dict(
      "__id__" => "ACE1_PolyEnvelope",
      "T" => write_dict(T), 
      "p" => env.p,
      "r0" => env.r0,
      "rcut" => env.rcut, )

function read_dict(::Val{:ACE1_PolyEnvelope}, D::Dict) 
   T = read_dict(D["T"])
   return PolyEnvelope(Int(D["p"]), T(D["r0"]), T(D["rcut"]), )
end


struct TwoSidedEnvelope{T} <: AbstractEnvelopeX
   xl::T 
   xr::T 
   pl::Int 
   pr::Int
end

function evaluate(env::TwoSidedEnvelope, x)
   if x <= min(env.xl, env.xr) || x >= max(env.xl, env.xr) 
      return 0.0
   end
   return  (x - env.xl)^(env.pl) * (x - env.xr)^(env.pr)
end

write_dict(env::TwoSidedEnvelope{T}) where {T} = Dict(
      "__id__" => "ACE1_TwoSidedEnvelope",
      "T" => write_dict(T),
      "xl" => env.xl,
      "xr" => env.xr,
      "pl" => env.pl,
      "pr" => env.pr, )

function read_dict(::Val{:ACE1_TwoSidedEnvelope}, D::Dict) 
   T = read_dict(D["T"])
   return TwoSidedEnvelope(T(D["xl"]), T(D["xr"]), Int(D["pl"]), Int(D["pr"]))
end

# -------------------- utility function to construct radial basis with envelope 


"""
`transformed_jacobi_env(maxdeg, trans, envelope, rcut, rin)`

This creates a radial basis where the cutoff mechanism is provided by the 
envelope in the r domain. The orthogonality relation of the basis does not 
include the envelope which can therefore be interpreted as a prior. 
"""
function transformed_jacobi_env(maxdeg::Integer,
                                trans::DistanceTransform,
                                envelope::AbstractEnvelope, 
                                rcut::Real, rin::Real = 0.0;
                                kwargs...)
   J =  discrete_jacobi(maxdeg; 
            tcut = transform(trans, rcut),
            tin = transform(trans, rin),
            pcut = 0,
            kwargs...)
   return TransformedPolys(J, trans, rin, rcut, envelope)
end


end
