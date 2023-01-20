
module Splines 


import Interpolations 
using Interpolations: cubic_spline_interpolation

using JuLIP.Potentials: z2i, SZList
import ACE1: alloc_B, alloc_dB, alloc_temp, alloc_temp_d, evaluate!, evaluate_d!

# _rr = range(0.0, stop=5.0, length=10)
# _ff = sin.(_rr)
# const SPLINE{T} = Interpolations.Extrapolation{T, 1, ScaledInterpolation{T, 1, Interpolations.BSplineInterpolation{T, 1, OffsetArrays.OffsetVector{T, Vector{T}}, BSpline{Cubic{Line{OnGrid}}}, Tuple{Base.OneTo{Int64}}}, BSpline{Cubic{Line{OnGrid}}}, Tuple{StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int64}}}, BSpline{Cubic{Line{OnGrid}}}, Throw{Nothing}}

struct RadialSplines{T, NZ, SPLINE}
   rcut::Matrix{T}
   splines::Matrix{SPLINE}
   zlist::SZList{NZ}
end

Base.length(basis::RadialSplines) = size(basis.splines, 1)

alloc_B(basis::RadialSplines{T}, args...) where {T} = zeros(T, length(basis))

alloc_dB(basis::RadialSplines{T}, args...) where {T} = zeros(T, length(basis))

alloc_temp(basis::RadialSplines, args...) = nothing 

alloc_temp_d(basis::RadialSplines, args...) = nothing 


function evaluate!(B, tmp, basis::RadialSplines, r, z, z0)
   iz = z2i(basis.zlist, z)
   iz0 = z2i(basis.zlist, z0)
   fill!(B, 0)

   # suppose we are outside the cutoff, then skip this
   if basis.rcut[iz, iz0] >= r
      return B 
   end

   len = size(basis.splines, 1)
   @assert length(B) >= len 

   for n = 1:len 
      spl = basis.splines[n, iz, iz0]
      B[n] = spl(r)
   end

   return B
end

function evaluate_d!(B, dB, tmpd, basis::RadialSplines, r, z, z0)
   iz = z2i(basis.zlist, z)
   iz0 = z2i(basis.zlist, z0)
   fill!(B, 0)
   fill!(dB, 0)

   # suppose we are outside the cutoff, then skip this
   if basis.rcut[iz, iz0] >= r
      return dB 
   end

   len = size(basis.splines, 1)
   @assert length(B) >= len 

   dr = ForwardDiff.Dual(r, 1)

   for n = 1:len 
      spl = basis.splines[n, iz, iz0]
      d_spl = spl(dr)
      B[n] = d_spl.value 
      dB[n] = d_spl.partials[1]
   end

   return dB
end


end