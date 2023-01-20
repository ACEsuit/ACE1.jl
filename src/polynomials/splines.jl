
module Splines 


import Interpolations, ACE1, ForwardDiff

using Interpolations: cubic_spline_interpolation

using JuLIP.Potentials: z2i, i2z, SZList, cutoff
import ACE1: alloc_B, alloc_dB, alloc_temp, alloc_temp_d, 
             evaluate!, evaluate_d!, evaluate 

# _rr = range(0.0, stop=5.0, length=10)
# _ff = sin.(_rr)
# const SPLINE{T} = Interpolations.Extrapolation{T, 1, ScaledInterpolation{T, 1, Interpolations.BSplineInterpolation{T, 1, OffsetArrays.OffsetVector{T, Vector{T}}, BSpline{Cubic{Line{OnGrid}}}, Tuple{Base.OneTo{Int64}}}, BSpline{Cubic{Line{OnGrid}}}, Tuple{StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int64}}}, BSpline{Cubic{Line{OnGrid}}}, Throw{Nothing}}

struct RadialSplines{T, NZ, SPLINE} <: ACE1.ScalarBasis{T}
   rcut::T
   splines::Array{SPLINE, 3}
   zlist::SZList{NZ}
end


Base.length(basis::RadialSplines) = size(basis.splines, 1)

# --------------------- Setup codes

# assuming that J has a multi-transform, but if not then the 
# zlist can be passed in. 

function RadialSplines(J; nnodes = 1_000, zlist = J.trans.zlist)
   rcut = cutoff(J) 
   dx = rcut/nnodes 
   rcut += 4*dx 
   rr = range(0.0, stop=rcut, length=nnodes)
   NZ = length(zlist)
   NB = length(J)
   splines = Array{Any}(undef, NB, NZ, NZ)
   for iz = 1:NZ, iz0 = 1:NZ
      z = i2z(zlist, iz)
      z0 = i2z(zlist, iz0)
      J_zz0 = zeros(NB, nnodes)
      for (ir, r) in enumerate(rr) 
         J_zz0[:, ir] = evaluate(J, r, z, z0)
      end
      for ib = 1:NB 
         splines[ib, iz, iz0] = cubic_spline_interpolation(rr, J_zz0[ib, :])
      end
   end

   splines_ = identity.(splines)
   return RadialSplines(rcut, splines_, zlist)
end

# --------------------- Evaluation codes 


alloc_B(basis::RadialSplines{T}, args...) where {T} = zeros(T, length(basis))

alloc_dB(basis::RadialSplines{T}, ::Number) where {T} = zeros(T, length(basis))
alloc_dB(basis::RadialSplines{T}, ::Int64) where {T} = zeros(T, length(basis))

alloc_temp(basis::RadialSplines, args...) = nothing 

alloc_temp_d(basis::RadialSplines, args...) = nothing 


function evaluate!(B, tmp, basis::RadialSplines, r, z, z0)
   iz = z2i(basis.zlist, z)
   iz0 = z2i(basis.zlist, z0)
   fill!(B, 0)

   # suppose we are outside the cutoff, then skip this
   if basis.rcut <= r
      print(".")
      return B 
   end

   len = size(basis.splines, 1)
   @assert length(B) >= len 

   # inbounds here would redice cost by ca 10%, not worth it, this is 
   # not a bottleneck
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
   if basis.rcut <= r
      return dB 
   end

   len = size(basis.splines, 1)
   # @assert length(B) >= len 

   dr = ForwardDiff.Dual(r, 1)

   # inbounds here would redice cost by ca 10%, not worth it, this is 
   # not a bottleneck
   for n = 1:len 
      spl = basis.splines[n, iz, iz0]
      d_spl = spl(dr)
      B[n] = d_spl.value 
      dB[n] = d_spl.partials[1]
   end

   return dB
end


end