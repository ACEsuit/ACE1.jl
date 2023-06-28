
# experimental code for compressing a PIPotential 

function _cleanup(object::Any)
   # @info("Cannot cleanup/compress $(typeof(object)); leaving it unchanged.")
   return object
end

function _cleanup(basis::JuLIP.MLIPs.IPSuperBasis)
   BB_new = _cleanup.(basis.BB)
   return JuLIP.MLIPs.IPSuperBasis(BB_new)
end

function _cleanup(basis::PolyPairBasis)
   # the pair basis is normally not an issue, just return as is. 
   # but at some point this should also be implemented. 
   return basis 
end 

function _cleanup(basis::RPIBasis; tol=1e-15)
   NZ = length(basis.A2Bmaps)
   _findIz(C)  = sort( findall( sum(abs, C, dims=1)[:] .<= tol ) )
   _findInz(C) = sort( findall( sum(abs, C, dims=1)[:] .>  tol ) )
   Iz  = ntuple(i ->  _findIz(basis.A2Bmaps[i]), NZ)
   Inz = ntuple(i -> _findInz(basis.A2Bmaps[i]), NZ)

   A2Bmaps_new = ntuple(i -> basis.A2Bmaps[i][:, Inz[i]], NZ)
   pibasis_new = _cleanup(basis.pibasis, Inz, Iz)

   return RPIBasis(pibasis_new, A2Bmaps_new, basis.Bz0inds)
end

function _cleanup(pibasis::PIBasis, IInz, IIz)
   NZ = length(pibasis.zlist) 
   @assert length(IInz) == NZ 
   @assert length(IIz) == NZ 
   @assert all( sort(union(IIz[i], IInz[i])) == 1:length(pibasis.inner[i]) 
                for i = 1:NZ )

   # START THE CLEANUP 
   # STAGE 1 - remove the inner basis functions that we no longer want                   
   innernew = []
   idx0 = 0

   for iz0 = 1:NZ
      Iz = IIz[iz0]  # zeros
      Inz = IInz[iz0]   # non-zeros
      # inverse mapping from old to new (local) indices
      invInz = Dict{Int, Int}()
      for (i, val) in enumerate(Inz)
         invInz[val] = i
      end
      # extract the inner basis to start rebuilding it
      inner = pibasis.inner[iz0]     # old
      iAA2iA = inner.iAA2iA[Inz, :]    # new
      # construct the new b2iAA mapping
      b2iAA = Dict{ACE1.PIBasisFcn, Int}()   # new
      for (b, iAA) in inner.b2iAA
         if haskey(invInz, iAA)
            b2iAA[b] = invInz[iAA]
         end
      end
      # create the new inner basis
      push!(innernew,
            ACE1.InnerPIBasis( inner.orders[Inz],
                              iAA2iA,
                              b2iAA,
                              inner.b2iA,
                              (idx0+1):(idx0+length(Inz)),
                              inner.z0,
                              ACE1.DAG.CorrEvalGraph{Int, Int}() ) )
      idx0 += length(Inz)                              
   end
   pibasis_new = PIBasis( pibasis.basis1p, pibasis.zlist,
                      tuple( innernew... ), pibasis.evaluator )

   # STAGE 2: find all the required A basis functions and see 
   #          if we can remove an of the old ones, then rebuild basis1p 

   # make a list of all used iA indices 
   new_iA = Int[] 
   for inner in pibasis_new.inner
      append!(new_iA, unique(sort(inner.iAA2iA[:])))
   end
   # get rid of 0 which is artificial 
   if new_iA[1] == 0
      new_iA = new_iA[2:end]
   end

   # compute the maximum l and n indices to compress the Rn and Ylm 
   # basis sets. 
   basis1p = pibasis_new.basis1p
   maxl = maximum(b.l for b in basis1p.spec[new_iA])
   maxn = maximum(b.n for b in basis1p.spec[new_iA])
   SH_new = ACE1.SHBasis(maxl)
   J_new = deepcopy(basis1p.J)
   deleteat!(J_new.J.A, maxn+2:length(J_new))
   deleteat!(J_new.J.B, maxn+2:length(J_new))
   deleteat!(J_new.J.C, maxn+2:length(J_new))

   # now we can rebuild the basis1p
   spec_new = basis1p.spec[new_iA]
   Ainds_new = fill( 1:length(new_iA) , size(basis1p.Aindices) )
   basis1p_new = ACE1.RPI.BasicPSH1pBasis(J_new, SH_new, basis1p.zlist, spec_new, Ainds_new)
   pibasis_new.basis1p = basis1p_new
   
   # STAGE 3: rebuild the inner basis, pointing to the correct new 
   #          iA indices in the new A basis 
   
   # first we need the mapping from old to new iA indices 
   inv_iA = Dict{Int, Int}()
   for (i, iA) in enumerate(new_iA)
      inv_iA[iA] = i
   end
   inv_iA[0] = 0 
   # and the new mapping from b -> iA 
   inv_1pspec = Dict{eltype(basis1p_new.spec), Int}()
   for (i, b) in enumerate(basis1p_new.spec)
      inv_1pspec[b] = i
   end
   
   # rewrite the pibasis_new.inner arrays to point to the correct 
   # A basis information 
   for iz0 = 1:NZ
   inner = pibasis_new.inner[iz0]
      for i = 1:length(inner.iAA2iA)
         inner.iAA2iA[i] = inv_iA[inner.iAA2iA[i]]      
      end
      for (key, val) in inner.b2iA
         if haskey(inv_1pspec, key)
            inner.b2iA[key] = inv_iA[val]
         else
            delete!(inner.b2iA, key)
         end
      end
   end 

   ## STAGE 4: rebuild the evaluation graphs 
   for iz0 = 1:NZ
      inner = pibasis_new.inner[iz0]
      ACE1.generate_dag!(inner)
   end

   return pibasis_new 
end


function _cleanup(V::PIPotential; tol = 1e-15)
   NZ = length(V.coeffs)

   # first determine the list of zero and non-zero coefficients 
   # for each species 
   IIz = ntuple( iz0 -> sort(findall(abs.(V.coeffs[iz0]) .<= tol)), NZ)
   IInz = ntuple(iz0 -> sort(findall(abs.(V.coeffs[iz0]) .>  tol)), NZ)

   coeffs_new = ntuple(iz0 -> V.coeffs[iz0][IInz[iz0]], NZ)
   pibasis_new = _cleanup(V.pibasis, IInz, IIz)
   
   # rebuild the potential with the new pibasis
   return PIPotential( pibasis_new, coeffs_new )
end


function _old()
   idx0 = 0
   innernew = []
   coeffs = []
   for iz0 = 1:length(V.coeffs)
      # zeros
      Iz = findall(abs.(V.coeffs[iz0]) .<= tol)
      # non-zeros
      Inz = findall(abs.(V.coeffs[iz0]) .> 0)
      # inverse mapping from old to new (local) indices
      invInz = Dict{Int, Int}()
      for (i, val) in enumerate(Inz)
         invInz[val] = i
      end
      # extract the inner basis to start rebuilding it
      inner = V.pibasis.inner[iz0]     # old
      iAA2iA = inner.iAA2iA[Inz, :]    # new
      # construct the new b2iAA mapping
      b2iAA = Dict{ACE1.PIBasisFcn, Int}()   # new
      for (b, iAA) in inner.b2iAA
         if haskey(invInz, iAA)
            b2iAA[b] = invInz[iAA]
         end
      end
      # create the new inner basis
      push!(innernew,
            ACE1.InnerPIBasis( inner.orders[Inz],
                              iAA2iA,
                              b2iAA,
                              inner.b2iA,
                              (idx0+1):(idx0+length(Inz)),
                              inner.z0,
                              ACE1.DAG.CorrEvalGraph{Int, Int}() ) )
      # and the new coefficients
      push!(coeffs, V.coeffs[iz0][Inz])
   end
   pibasis = PIBasis( V.pibasis.basis1p, V.pibasis.zlist,
                      tuple( innernew... ), V.pibasis.evaluator )

   # STAGE 2: find all the required A basis functions and see 
   #          if we can remove an of the old ones, then rebuild basis1p 

   # make a list of all used iA indices 
   new_iA = Int[] 
   for inner in pibasis.inner
      append!(new_iA, unique(sort(inner.iAA2iA[:])))
   end
   # get rid of 0 which is artificial 
   if new_iA[1] == 0
      new_iA = new_iA[2:end]
   end

   # compute the maximum l and n indices to compress the Rn and Ylm 
   # basis sets. 
   basis1p = pibasis.basis1p
   maxl = maximum(b.l for b in basis1p.spec[new_iA])
   maxn = maximum(b.n for b in basis1p.spec[new_iA])
   SH_new = ACE1.SHBasis(maxl)
   J_new = deepcopy(basis1p.J)
   deleteat!(J_new.J.A, maxn+2:length(J_new))
   deleteat!(J_new.J.B, maxn+2:length(J_new))
   deleteat!(J_new.J.C, maxn+2:length(J_new))

   # now we can rebuild the basis1p
   spec_new = basis1p.spec[new_iA]
   Ainds_new = fill( 1:length(new_iA) , size(basis1p.Aindices) )
   basis1p_new = ACE1.RPI.BasicPSH1pBasis(J_new, SH_new, basis1p.zlist, spec_new, Ainds_new)
   pibasis.basis1p = basis1p_new
   
   # STAGE 3: rebuild the inner basis, pointing to the correct new 
   #          iA indices in the new A basis 
   
   # first we need the mapping from old to new iA indices 
   inv_iA = Dict{Int, Int}()
   for (i, iA) in enumerate(new_iA)
      inv_iA[iA] = i
   end
   inv_iA[0] = 0 
   # and the new mapping from b -> iA 
   inv_1pspec = Dict{eltype(basis1p_new.spec), Int}()
   for (i, b) in enumerate(basis1p_new.spec)
      inv_1pspec[b] = i
   end
   
   # rewrite the pibasis.inner arrays to point to the correct 
   # A basis information 
   for iz0 = 1:length(V.coeffs)
   inner = pibasis.inner[iz0]
      for i = 1:length(inner.iAA2iA)
         inner.iAA2iA[i] = inv_iA[inner.iAA2iA[i]]      
      end
      for (key, val) in inner.b2iA
         if haskey(inv_1pspec, key)
            inner.b2iA[key] = inv_iA[val]
         else
            delete!(inner.b2iA, key)
         end
      end
   end 

   ## STAGE 4: rebuild the evaluation graphs 
   for iz0 = 1:length(V.coeffs) 
      inner = pibasis.inner[iz0]
      ACE1.generate_dag!(inner)
   end
   
   # this should complete the pibasis compression 

   # STAGE 5: rebuild the potential with the new pibasis
   V_new = PIPotential( pibasis, tuple(coeffs...) )

   return V_new
end 