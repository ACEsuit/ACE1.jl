
# --------------------------------------------------------------------------
# ACE1.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


module Utils

import ACE1
import ACE1.RPI: BasicPSH1pBasis, SparsePSHDegree, RPIBasis, get_maxn, 
                 SparsePSHDegreeM
import ACE1: PolyTransform, transformed_jacobi
import ACE1.PairPotentials: PolyPairBasis

# - simple ways to construct a radial basis
# - construct a descriptor
# - simple wrappers to generate RPI basis functions (ACE + relatives)

export rpi_basis, descriptor, pair_basis, ace_basis, splinify 


"""
`_auto_degrees` : provide a simplified interface to generate degree parameters 
for the ACE1 basis. 
"""
function _auto_degrees(N::Integer, maxdeg::Number, wL::Number, D = nothing)
   if D == nothing
      D = SparsePSHDegree(; wL = wL)
   end
   return D, maxdeg 
end

function _auto_degrees(N::Integer, maxdeg::AbstractVector, wL::Union{Number, AbstractVector}, D = nothing)
   if D == nothing 
      if wL isa Number
         wL = wL .* ones(N)
      end
      Dn = Dict("default" => 1.0) 
      Dl = Dict([n => wL[n] for n in 1:N]...)
      Dd = Dict([n => maxdeg[n] for n in 1:N]...)
      D = SparsePSHDegreeM(Dn, Dl, Dd)
      maxdeg = 1
   end

   return D, maxdeg
end


function rpi_basis(; species = :X, N = 3,
      # transform parameters
      r0 = 2.5,
      trans = PolyTransform(2, r0),
      # degree parameters
      wL = 1.5, 
      maxdeg = 8,
      D = nothing,
      # radial basis parameters
      rcut = 5.0,
      rin = 0.5 * r0,
      pcut = 2,
      pin = 2,
      constants = false,
      rbasis = nothing, 
      # one-particle basis type 
      Basis1p = BasicPSH1pBasis, 
      warn = true)

   D, maxdeg = _auto_degrees(N, maxdeg, wL, D)     

   if rbasis == nothing    
      if (pcut < 2) && warn 
         @warn("`pcut` should normally be ≥ 2.")
      end
      if (pin < 2) && (pin != 0) && warn 
         @warn("`pin` should normally be ≥ 2 or 0.")
      end

      rbasis = transformed_jacobi(get_maxn(D, maxdeg, species), trans, rcut, rin;
                                  pcut=pcut, pin=pin)
   end

   basis1p = Basis1p(rbasis; species = species, D = D)
   return RPIBasis(basis1p, N, D, maxdeg, constants)
end

descriptor = rpi_basis
ace_basis = rpi_basis

function pair_basis(; species = :X,
      # transform parameters
      r0 = 2.5,
      trans = PolyTransform(2, r0),
      # degree parameters
      maxdeg = 8,
      # radial basis parameters
      rcut = 5.0,
      rin = 0.5 * r0,
      pcut = 2,
      pin = 0,
      rbasis = transformed_jacobi(maxdeg, trans, rcut, rin; pcut=pcut, pin=pin))

   return PolyPairBasis(rbasis, species)
end


# --------------------- Splinify an RPI basis 


function splinify(basis::RPIBasis)
   basis1p = deepcopy(basis.pibasis.basis1p)
   J_spl = ACE1.Splines.RadialSplines(basis1p.J; zlist = basis1p.zlist)
   basis1p_spl = ACE1.RPI.BasicPSH1pBasis(J_spl, deepcopy(basis1p.SH), 
                  deepcopy(basis1p.zlist), deepcopy(basis1p.spec), deepcopy(basis1p.Aindices) )
   pibasis_spl = ACE1.PIBasis(basis1p_spl, deepcopy(basis1p.zlist), 
               deepcopy(basis.pibasis.inner), deepcopy(basis.pibasis.evaluator))
   basis_spl = RPIBasis(pibasis_spl, deepcopy(basis.A2Bmaps), deepcopy(basis.Bz0inds) )   
   return basis_spl
end


end
