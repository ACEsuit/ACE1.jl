

# tested a wide range of methods. Brent seems robust + fastest 
using Roots: find_zero, ITP, Brent 

struct Normalized{T, TT} <: DistanceTransform
   trans::TT
   y0::T 
   y1::T
   rcut::T
end

"""
takes an agnesi transform and normalizes it so that it maps 
[0, rcut] to [0, 1] with zero-slope at rcut. 

This is an experimental transformation.
"""
function normalized_agnesi_transform(r0, rcut; p = 2, q = 4)
   # 1 - (ag(r) - ag(1.0)) * (1.0 - r^4) / (ag(0.0) - ag(1.0))
   ag = agnesi_transform(r0, p, q)
   return Normalized(ag, ag(0.0), ag(rcut), rcut)
end

(t::Normalized)(r) = transform(t, r)

function transform(t::Normalized{T}, r::Number) where {T}
   y = transform(t.trans, r) 
   return 1 - (y - t.y1) / (t.y0 - t.y1) * (1 - (r/t.rcut)^4)
end

transform_d(t::Normalized, r::Number) = 
     ForwardDiff.derivative(r -> transform(t, r), r)

function inv_transform(t::Normalized{T}, x::Number) where {T} 
   if x <= 0 
      return zero(T)
   elseif x >= 1 
      return t.rcut 
   end

   g = r -> transform(t, r) - x
   r = find_zero(g, (0.0, t.rcut), Brent())
   @assert 0 <= r <= t.rcut
   @assert abs(g(r)) < 1e-14
   return r 
end


write_dict(T::Normalized) =
      Dict("__id__" => "ACE1_Normalized",
            "trans" => write_dict(T.trans), 
            "y0" => T.y0, "y1" => T.y1, "rcut" => T.rcut ) 
            
read_dict(::Val{:ACE1_Normalized}, D::Dict) = 
      Normalized(read_dict(D["trans"]), D["y0"], D["y1"], D["rcut"])

