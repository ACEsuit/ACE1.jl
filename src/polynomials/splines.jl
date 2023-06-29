
module Splines 

import JuLIP: z2i, AtomicNumber 
import Interpolations, ACE1, ForwardDiff

using Interpolations: cubic_spline_interpolation, Flat, OnGrid

using JuLIP.Potentials: z2i, i2z, SZList
import JuLIP.Potentials: cutoff 
import ACE1: alloc_B, alloc_dB, alloc_temp, alloc_temp_d, 
             evaluate!, evaluate_d!, evaluate, 
             write_dict, read_dict 

# _rr = range(0.0, stop=5.0, length=10)
# _ff = sin.(_rr)
# const SPLINE{T} = Interpolations.Extrapolation{T, 1, ScaledInterpolation{T, 1, Interpolations.BSplineInterpolation{T, 1, OffsetArrays.OffsetVector{T, Vector{T}}, BSpline{Cubic{Line{OnGrid}}}, Tuple{Base.OneTo{Int64}}}, BSpline{Cubic{Line{OnGrid}}}, Tuple{StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int64}}}, BSpline{Cubic{Line{OnGrid}}}, Throw{Nothing}}

struct RadialSplines{T, NZ, SPLINE} <: ACE1.ScalarBasis{T}
   splines::Array{SPLINE, 3}
   zlist::SZList{NZ}
   ranges::Array{Tuple{T, T, Int}, 3}
end


Base.length(basis::RadialSplines) = size(basis.splines, 1)

cutoff(basis::RadialSplines) = maximum(rg[2] for rg in basis.ranges)

cutoff(basis::RadialSplines, z::AtomicNumber, z0::AtomicNumber) = 
         maximum( rg[2] for rg in basis.ranges[:, z2i(basis.zlist, z), z2i(basis.zlist, z0)] )

# --------------------- Setup codes

# assuming that J has a multi-transform, but if not then the 
# zlist can be passed in. 

function RadialSplines(J; nnodes = 1_000, zlist = J.trans.zlist, 
                          mode = :bspline)
   if mode == :hermite 
      return RadialSplines_hermite(J, nnodes=nnodes, zlist=zlist) 
   elseif mode == :bspline 
      return RadialSplines_bspline(J, nnodes=nnodes, zlist=zlist) 
   else 
      error("Unknown mode: $mode")
   end
end 

function RadialSplines_hermite(J; nnodes = 1_000, zlist = J.trans.zlist)
   @warn("Hermite splines uses an un-documented feature of Interpolations.jl. This means that it can break at any time.")
   NZ = length(zlist)
   NB = length(J)
   rcut = cutoff(J)
   rg_params = (0.0, rcut, nnodes)
   rr = range(0.0, stop=rcut, length=nnodes)
   splines = Array{Any}(undef, NB, NZ, NZ)

   for iz = 1:NZ, iz0 = 1:NZ
      z = i2z(zlist, iz)
      z0 = i2z(zlist, iz0)
   
      J_zz0 = zeros(NB, nnodes)
      dJ_zz0 = zeros(NB, nnodes)
      for (ir, r) in enumerate(rr) 
         J_zz0[:, ir] = evaluate(J, r, z, z0)
         dJ_zz0[:, ir] = ACE1.evaluate_d(J, r, z, z0)
      end
      for ib = 1:NB 
         splines[ib, iz, iz0] = Interpolations.CubicHermite(rr, J_zz0[ib, :], dJ_zz0[ib, :])
      end
   end

   splines_ = identity.(splines)
   range_params = fill(rg_params, size(splines))
   return RadialSplines(splines_, zlist, range_params)
end

function cutoff(J::ACE1.OrthPolys.TransformedPolys, z::AtomicNumber, z0::AtomicNumber)
   if !(J.trans isa ACE1.Transforms.MultiTransform)
      return cutoff(J) 
   end 

   if !( (J.J.tl ≈ -1) && (J.J.tr ≈ 1) && 
         all(trans isa ACE1.Transforms.AffineT 
                   for trans in J.trans.transforms) )
      error("""To use cubic B splines I must be able to infer the species-dependent rcuts.
               I can't figure this out for the given basis. Try to use `mode = :hermite` instead.
            """)
   end

   rcut = try
      rr = ACE1.Transforms.inv_transform(J.trans, 1.0, z, z0)
      rl = ACE1.Transforms.inv_transform(J.trans, -1.0, z, z0)
      max(rr, rl)
   catch 
      error("""To use cubic B splines I need the inverse transform implemented. 
      Try to use a different transform, or try to use hermite interpolation.""")
   end

   return rcut 
end


function RadialSplines_bspline(J; nnodes = 1_000, zlist = J.trans.zlist)
   NZ = length(zlist)
   NB = length(J)
   splines = Array{Any}(undef, NB, NZ, NZ)
   range_params = Array{Any}(undef, NB, NZ, NZ)

   for iz = 1:NZ, iz0 = 1:NZ
      z = i2z(zlist, iz)
      z0 = i2z(zlist, iz0)

      rcut = cutoff(J, z, z0)
      range_params[:, iz, iz0] .= Ref((0.0, rcut, nnodes))
      rr = range(0.0, stop=rcut, length=nnodes)
   
      J_zz0 = zeros(NB, nnodes)
      for (ir, r) in enumerate(rr) 
         J_zz0[:, ir] = evaluate(J, r, z, z0)
      end
      for ib = 1:NB 
         splines[ib, iz, iz0] = cubic_spline_interpolation(rr, J_zz0[ib, :]; 
                                                         bc = Flat(OnGrid()))
      end
   end

   splines_ = identity.(splines)
   range_params_ = identity.(range_params)
   return RadialSplines(splines_, zlist, range_params_)
end

# --------------------- FIO 

function export_splines(basis::RadialSplines)
   _get_rg(rgp) = range(rgp[1], stop=rgp[2], length=rgp[3])
   _get_vals(spl, rg) = spl.(rg)
   ranges = _get_rg.(basis.ranges) 
   nodalvals = _get_vals.(basis.splines, ranges)
   # quick check that the splines we are exporting are not Hermite splines
   @assert eltype(basis.splines) <: Interpolations.Extrapolation
   return ranges, nodalvals, basis.zlist.list 
end


import Base: == 
==(P1::RadialSplines, P2::RadialSplines) = (
            (P1.ranges == P2.ranges) && (P1.zlist == P2.zlist) &&
            all([s1.itp.itp.coefs ≈ s2.itp.itp.coefs for (s1, s2) in zip(P1.splines, P2.splines)])
      )


function write_dict(basis::RadialSplines{T})  where {T} 
   ranges, nodalvals, _ = export_splines(basis)
   sz_spl = size(basis.splines)
   return Dict("__id__" => "ACE1_RadialSplines",
                "T" => write_dict(T), 
                "zlist" => write_dict(basis.zlist),
                "size" => sz_spl,
                "range_params" => basis.ranges[:], 
                "nodal_values" => nodalvals[:],
               )
end

function read_dict(::Val{:ACE1_RadialSplines}, D::Dict)
   T = read_dict(D["T"])
   zlist = read_dict(D["zlist"])
   range_params = D["range_params"]
   ranges = [ range(T(p[1]), stop=T(p[2]), length=Int(p[3])) 
               for p in range_params ]
   spl_vals = [ T.(v) for v in D["nodal_values"] ]
   
   splines = [ cubic_spline_interpolation(rg, vals; bc = Flat(OnGrid())) 
               for (rg, vals) in zip(ranges, spl_vals) ]
   splines_ = collect( reshape(splines, D["size"]...) )
   range_params = collect(reshape([ tuple(T(p[1]), T(p[2]), Int(p[3])) for p in range_params ], D["size"]...))
   return RadialSplines(splines_, zlist, range_params)
end

# --------------------- Evaluation codes 


alloc_B(basis::RadialSplines{T}, args...) where {T} = zeros(T, length(basis))

alloc_dB(basis::RadialSplines{T}, args...) where {T} = zeros(T, length(basis))
alloc_dB(basis::RadialSplines{T}, ::Number) where {T} = zeros(T, length(basis))
alloc_dB(basis::RadialSplines{T}, ::Int64) where {T} = zeros(T, length(basis))

alloc_temp(basis::RadialSplines, args...) = nothing 

alloc_temp_d(basis::RadialSplines, args...) = nothing 


function evaluate!(B, tmp, basis::RadialSplines, r, z, z0)
   iz = z2i(basis.zlist, z)
   iz0 = z2i(basis.zlist, z0)
   fill!(B, 0)

   # suppose we are outside the cutoff, then skip this
   if cutoff(basis) <= r
      print(".")
      return B 
   end

   len = size(basis.splines, 1)
   @assert length(B) >= len 

   # inbounds here would redice cost by ca 10%, not worth it, this is 
   # not a bottleneck
   for n = 1:len 
      spl = basis.splines[n, iz, iz0]
      if basis.ranges[n, iz, iz0][1] <= r <= basis.ranges[n, iz, iz0][2]
         B[n] = spl(r)
      else
         B[n] = 0
      end
   end

   return B
end

function evaluate_d!(B, dB, tmpd, basis::RadialSplines, r, z, z0)
   iz = z2i(basis.zlist, z)
   iz0 = z2i(basis.zlist, z0)
   fill!(B, 0)
   fill!(dB, 0)

   # suppose we are outside the cutoff, then skip this
   if cutoff(basis) <= r
      return dB 
   end

   len = size(basis.splines, 1)
   # @assert length(B) >= len 

   # dr = ForwardDiff.Dual(r, 1)

   # inbounds here would redice cost by ca 10%, not worth it, this is 
   # not a bottleneck
   for n = 1:len 
      spl = basis.splines[n, iz, iz0]
      if basis.ranges[n, iz, iz0][1] <= r <= basis.ranges[n, iz, iz0][2]
         # d_spl = spl(dr)
         # B[n] = d_spl.value 
         # dB[n] = d_spl.partials[1]
         B[n] = spl(r)
         dB[n] = Interpolations.gradient(spl, r)[1]
      else
         B[n] = 0
         dB[n] = 0
      end
   end

   return dB
end


end