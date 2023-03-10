
# --------------------------------------------------------------------------
# ACE1.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------



module Transforms

import Base:   ==
import JuLIP:  cutoff, AtomicNumber
import JuLIP.FIO: read_dict, write_dict
import ACE1: _allfieldsequal

abstract type DistanceTransform end

export PolyTransform, IdTransform, MorseTransform, AgnesiTransform, 
         agnesi_transform 

# fall-backs from species dependent to independent transforms 
transform(t::DistanceTransform, r::Number, z::AtomicNumber, z0::AtomicNumber) = transform(t, r)
transform_d(t::DistanceTransform, r::Number, z::AtomicNumber, z0::AtomicNumber) = transform_d(t, r)
inv_transform(t::DistanceTransform, x::Number, z::AtomicNumber, z0::AtomicNumber) = inv_transform(t, x)



poly_trans(p, r0, a, r) = 
         @fastmath(((a+r0)/(a+r))^p)

poly_trans_d(p, r0, a, r) = 
         @fastmath((-p/(a+r0)) * ((a+r0)/(a+r))^(p+1))

poly_trans_inv(p, r0, a, x) = 
         ( (a+r0)/(x^(1/p)) - a )


@doc raw"""
Implements the distance transform
```math
   x(r) = \Big(\frac{\varepsilon + r_0}{\varepsilon + r}\Big)^p
```

Constructor:
```
PolyTransform(; p = 2, r0 = 2.5, a = 1.0)
PolyTransform(p, r0, a)
```
"""
struct PolyTransform{TP, T} <: DistanceTransform
   p::TP
   r0::T
   a::T 
end

PolyTransform(p, r0) = PolyTransform(p, r0, one(eltype(r0)))
PolyTransform(; p = 2, r0 = 2.5, a = one(eltype(r0))) = PolyTransform(p, r0, a)

write_dict(T::PolyTransform) =
   Dict("__id__" => "ACE1_PolyTransform", 
             "p" => T.p, 
            "r0" => T.r0, 
             "a" => T.a)

PolyTransform(D::Dict) = PolyTransform(D["p"], D["r0"], D["a"])

read_dict(::Val{:ACE1_PolyTransform}, D::Dict) = PolyTransform(D)

transform(t::PolyTransform, r::Number) = poly_trans(t.p, t.r0, t.a, r)

transform_d(t::PolyTransform, r::Number) = poly_trans_d(t.p, t.r0, t.a, r)

inv_transform(t::PolyTransform, x::Number) = poly_trans_inv(t.p, t.r0, t.a, x)

(t::PolyTransform)(x) = transform(t, x)

"""
`IdTransform`: Implements the distance transform `z -> z`;
Primarily used for the z-coordinate for the EnvPairPots

Constructor: `IdTransform()`
"""
struct IdTransform <: DistanceTransform
end

write_dict(T::IdTransform) =  Dict("__id__" => "ACE1_IdTransform")
IdTransform(D::Dict) = IdTransform()
read_dict(::Val{:ACE1_IdTransform}, D::Dict) = IdTransform(D)
transform(t::IdTransform, z::Number) = z
transform_d(t::IdTransform, r::Number) = one(r)
inv_transform(t::IdTransform, x::Number) = x


   


@doc raw"""
Implements the distance transform
```math
   x(r) = \exp( - \lambda (r/r_0) )
```

Constructor:
```
MorseTransform(lambda, r0)
```
"""
struct MorseTransform{T} <: DistanceTransform
   lambda::T
   r0::T
end

write_dict(T::MorseTransform) =
   Dict("__id__" => "ACE1_MorseTransform", "lambda" => T.p, "r0" => T.r0)
MorseTransform(D::Dict) = MorseTransform(D["lambda"], D["r0"])
read_dict(::Val{:ACE1_MorseTransform}, D::Dict) = MorseTransform(D)
transform(t::MorseTransform, r::Number) = exp(- t.lambda * (r/t.r0))
transform_d(t::MorseTransform, r::Number) = (-t.lambda/t.r0) * exp(- t.lambda * (r/t.r0))
inv_transform(t::MorseTransform, x::Number) = - t.r0/t.lambda * log(x)
(t::MorseTransform)(x) = transform(t, x)


@doc raw"""
Implements the distance transform
```math
   x(r) = \frac{1}{1 + a (r/r_0)^p}
```
with default $a = (p-1)/(p+1)$. That default is chosen such that
$|x'(r)|$ is maximised at $r = r_0$. Default for $p$ is $p = 2$. Any value
$p > 1$ is permitted, but $p >= 2$ is recommended. 

Constructor:
```
AgnesiTransform(r0, [p, [, a]])
```
"""
struct AgnesiTransform{T, TP} <: DistanceTransform
   r0::T
   p::TP
   a::T
   rin::T
end

AgnesiTransform(r0, p=2, a=(p-1)/(p+1)) = (@assert p > 1;
                  AgnesiTransform(r0, p, a, zero(eltype(r0))))

AgnesiTransform(; r0 = nothing, 
                  p = 2, 
                  a = (p-1)/(p+1), 
                  rin = zero(eltype(r0)), ) = 
         AgnesiTransform(r0, p, a, rin)

write_dict(T::AgnesiTransform) =
      Dict("__id__" => "ACE1_AgnesiTransform", 
           "r0" => T.r0, "p" => T.p, "a" => T.a, "rin" => T.rin)

AgnesiTransform(D::Dict) = AgnesiTransform(; r0=D["r0"], p=D["p"], 
                                             a=D["a"], rin=D["rin"])

read_dict(::Val{:ACE1_AgnesiTransform}, D::Dict) = AgnesiTransform(D)

function transform(t::AgnesiTransform{T}, r::Number) where {T} 
   if r <= t.rin 
      return one(T) 
   end 
   s = (r-t.rin)/(t.r0-t.rin)
   return 1 / (1 + t.a * s^t.p)
end

transform_d(t::AgnesiTransform, r::Number) = 
     ForwardDiff.derivative(r -> transform(t, r), r)

# (s1 = (r/t.r0); s2 = s1^(t.p-1);
# @fastmath t.c * s2 / (1+t.a * s2*s1)^2)

function inv_transform(t::AgnesiTransform, x::Number) 
   s = ( (1/x-1)/t.a )^(1/t.p)   # s = (r - rin) / (r0 - rin)
   return t.rin + s * (t.r0 - t.rin)
end

(t::AgnesiTransform)(r) = transform(t, r)

# --------- generalized agnesi transform 

@doc raw"""
`function agnesi_transform:` constructs a generalized agnesi transform. 
```
trans = agnesi_transform(r0, p, q)
```
with `q >= p`. This generates an `AnalyticTransform` object that implements 
```math
   x(r) = \frac{1}{1 + a (r/r_0)^q / (1 + (r/r0)^(q-p))}
```
with default `a` chosen such that $|x'(r)|$ is maximised at $r = r_0$. But `a` may also be specified directly as a keyword argument. 

The transform satisfies 
```math 
   x(r) \sim \frac{1}{1 + a (r/r_0)^p} \quad \text{as} \quad r \to 0 
   \quad \text{and} 
   \quad 
   x(r) \sim \frac{1}{1 + a (r/r_0)^p}  \quad \text{as} r \to \infty.
```

As default parameters we recommend `p = 2, q = 4` and the defaults for `a`.

Note that the inverse transform is only implemented for the special cases 
$p = q$ and $q = 2p$. The inverse is not needed for fitting or simulation, but 
only for some tests.
"""
function agnesi_transform(r0, p, q;    
               a = (-2 * q + p * (-2 + 4 * q)) / (p + p^2 + q + q^2) )
   @assert p > 0
   @assert q > 0
   @assert q >= p      
   @assert a > 0 
   return Agnesi2Transform(r0, p, q, a, zero(r0))
end


struct Agnesi2Transform{T, TP} <: DistanceTransform
   r0::T
   p::TP
   q::TP 
   a::T
   rin::T
end

(t::Agnesi2Transform)(r) = transform(t, r)

write_dict(T::Agnesi2Transform) =
      Dict("__id__" => "ACE1_Agnesi2Transform", 
           "r0" => T.r0, "p" => T.p, "q" => T.q, "a" => T.a, "rin" => T.rin)

Agnesi2Transform(D::Dict) = Agnesi2Transform(D["r0"], D["p"], D["q"], 
                                             D["a"], D["rin"])

read_dict(::Val{:ACE1_Agnesi2Transform}, D::Dict) = Agnesi2Transform(D)

function transform(t::Agnesi2Transform{T}, r::Number) where {T} 
   if r <= t.rin 
      return one(T) 
   end 
   a, r0, q, p, rin = t.a, t.r0, t.q, t.p, t.rin
   s = (r-t.rin)/(t.r0-t.rin)
   return 1 / (1 + a * s^q / (1 + s^(q-p)))
end

transform_d(t::Agnesi2Transform, r::Number) = 
     ForwardDiff.derivative(r -> transform(t, r), r)

function inv_transform(t::Agnesi2Transform, x::Number)
   a, r0, q, p, rin = t.a, t.r0, t.q, t.p, t.rin
   y = (1/x-1)/a
   if y < 0 
      return rin 
   end
   if p == q
      s = (2 * y)^(1/p)
   elseif q == 2*p
      s = (y/2 + sqrt(y^2/4 + y))^(1/p)
   else
      error("inverse transform not implemented for this case")
   end
   return t.rin + s * (t.r0 - t.rin)
end




# --------- AnalyticTransform 

import JuLIP
import JuLIP.Potentials: ScalarFun

"""
`AnalyticTransform`: implements a distance transform that can be specified
by an analytic expression. 

Constructor: 
```julia 
AnalyticTransform(forwardmap, inversemap)
```
For `forwardmap` and `inversemap` must both be of type `AnalyticFunction`.
(cf. `JuLIP.@analytic`).

Example: 
```julia
using ACE1: AnalyticTransform
trans = AnalyticTransform( "r -> exp( - 2 * r )", 
                           "x -> -0.5 * log(x)" )
```
"""
struct AnalyticTransform{T} <: DistanceTransform
   f::ScalarFun{T}
   df::ScalarFun{T}
   finv::ScalarFun{T}
   str_f::String
   str_finv::String
end

==(T1::AnalyticTransform, T2::AnalyticTransform) = T1.str_f == T2.str_f

Base.show(io::IO, trans::AnalyticTransform) = 
      print(io, "AnalyticTransform($(trans.str_f))")

function AnalyticTransform(str_f::String, str_finv::String; T=Float64)
   ex_f = Meta.parse(str_f)
   ex_df = JuLIP.Potentials.fdiff( ex_f, 1 )
   if str_finv == "auto"
      @warn("automatic inverse not implemented, inverse will return NaN")
      ex_finv = :(r -> NaN)
   else
      ex_finv = Meta.parse(str_finv)
   end
   f = ScalarFun{T}(Meta.eval(ex_f))
   df = ScalarFun{T}(Meta.eval(ex_df))
   finv = ScalarFun{T}(Meta.eval(ex_finv))
   return AnalyticTransform(f, df, finv, str_f, str_finv)
end


write_dict(T::AnalyticTransform) =
         Dict("__id__" => "ACE1_AnalyticTransform", 
              "f" => T.str_f, "finv" => T.str_finv)
AnalyticTransform(D::Dict) = AnalyticTransform(D["f"], D["finv"])
read_dict(::Val{:ACE1_AnalyticTransform}, D::Dict) = AnalyticTransform(D)

transform(t::AnalyticTransform, r::Number) = t.f(r)
transform_d(t::AnalyticTransform, r::Number) = t.df(r) 
inv_transform(t::AnalyticTransform, x::Number) = t.finv(x)
(t::AnalyticTransform)(r) = transform(t, r)

import ForwardDiff
transform(t::AnalyticTransform, r::ForwardDiff.Dual) = t.f.obj.x(r)

# --------- Utility function - affine transform 

"""
`AffineT` : wraps another transform and then applies an affine transformation. 

Constructor: 
```julia
AffineT(transform, x1, x2, y1, y2)
```
then `x = t(r)` and then `x -> y` with `xi -> yi`.
"""
struct AffineT{T, TT} <: DistanceTransform
   t::TT   # the inner transform  : t(r) = x
   x1::T   # intervals to be transformed x1 -> y1 etc...
   x2::T 
   y1::T 
   y2::T
end

transform(t::AffineT, r) = t.y1 + (transform(t.t, r) - t.x1) * (t.y2-t.y1)/(t.x2-t.x1)
transform_d(t::AffineT, r) = ((t.y2-t.y1)/(t.x2-t.x1)) * transform_d(t.t, r)
inv_transform(t::AffineT, y) = inv_transform(t.t, t.x1 + (y - t.y1) * (t.x2-t.x1)/(t.y2-t.y1))

==(t1::AffineT, t2::AffineT) = _allfieldsequal(t1, t2) 

write_dict(T::AffineT) = 
      Dict("__id__" => "ACE1_AffineT", 
           "t" => write_dict(T.t), 
           "xy" => [T.x1, T.x2, T.y1, T.y2] )

read_dict(::Val{:ACE1_AffineT}, D::Dict) = AffineT(D) 

AffineT(D::Dict) = AffineT(read_dict(D["t"]), 
                         D["xy"]... )

# --------- Multi-transform: species-dependent transform 

import JuLIP: chemical_symbol
import JuLIP.Potentials: ZList, SZList, i2z, z2i
using StaticArrays: SMatrix
struct MultiTransform{NZ, TT} <: DistanceTransform
   zlist::SZList{NZ}
   transforms::SMatrix{NZ, NZ, TT}
end 


cutoff_extrema(T::MultiTransform) = 
   minimum( inv_transform(t, -1.0) for t in T.transforms ), 
   maximum( inv_transform(t,  1.0) for t in T.transforms )

# FIO 

==(T1::MultiTransform, T2::MultiTransform) =  _allfieldsequal(T1, T2)

write_dict(T::MultiTransform) =
      Dict("__id__" => "ACE1_MultiTransform", 
           "zlist" => write_dict(T.zlist), 
           "transforms" => write_dict.(T.transforms[:]))

read_dict(::Val{:ACE1_MultiTransform}, D::Dict) = MultiTransform(D)

function MultiTransform(D::Dict) 
   zlist = read_dict(D["zlist"])
   NZ = length(zlist) 
   transforms = SMatrix{NZ, NZ}( read_dict.(D["transforms"])... )
   return MultiTransform(zlist, transforms)   
end

#  Constructor 


function multitransform(D::Dict; rin=nothing, rcut=nothing, cutoffs = nothing)

   if rin != nothing && rcut != nothing && cutoffs == nothing 
      cutoffs = Dict( [key => (rin, rcut) for key in keys(D)]... )
   end 

   species = Symbol[] 
   for key in keys(D) 
      append!(species, [key...])
   end
   species = unique(species) 
   zlist = ZList(species, static=true)
   NZ = length(zlist) 
   transforms = Matrix{Any}(undef, NZ, NZ) 
   for i = 1:NZ, j = 1:NZ 
      Si = chemical_symbol(i2z(zlist, i))
      Sj = chemical_symbol(i2z(zlist, j))
      if haskey(D, (Si, Sj))
         key = (Si, Sj)
      else 
         key = (Sj, Si)
      end
      # get the transform from the dict 
      t = D[key]
      # apply another affine transform so all transforms have the same range 
      if cutoffs != nothing
         rin, rcut = cutoffs[key]
         x1 = transform(t, rin)
         x2 = transform(t, rcut)
         transforms[i, j] = AffineT(t, x1, x2, -1.0, 1.0)
      else 
         transforms[i, j] = t
      end
   end
   transforms = identity.(transforms)  # infer the type 
   return MultiTransform(zlist, SMatrix{NZ, NZ}(transforms...))
end

transform(t::MultiTransform, r::Number, z::AtomicNumber, z0::AtomicNumber) = 
      transform(t.transforms[z2i(t.zlist, z), z2i(t.zlist, z0)], r)

transform_d(t::MultiTransform, r::Number, z::AtomicNumber, z0::AtomicNumber) =
      transform_d(t.transforms[z2i(t.zlist, z), z2i(t.zlist, z0)], r)

inv_transform(t::MultiTransform, y::Number, z::AtomicNumber, z0::AtomicNumber) =
      inv_transform(t.transforms[z2i(t.zlist, z), z2i(t.zlist, z0)], y)

# # NOTE: This is a bit of a hack, I'm checking whether 
# #       the transforms have been transformed to the domain [-1, 1]
# #       then I'm checking whether r is either rin or rcut and only 
# #       then do I return the value
# function transform(t::MultiTransform{NZ, TT}, r::Number) where {NZ, TT}
#    @assert (TT <: AffineT) "transform(::MultiTransfrom, r) is only defined if rin, rcut are specified during construction"
#    x = sum( transform(_t, r)  for _t in t.transforms ) / NZ^2 
#    @assert (abs(abs(x) - 1) <= 1e-7) "transform(::MultiTransfrom, r) is only defined for r = rin, rcut"
#    return x  
# end

end 
