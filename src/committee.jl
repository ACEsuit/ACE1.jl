
using ACE1.RPI: get_picoeffs
using ACE1.DAG: CorrEvalGraph

function assert_has_co(V::PIPotential) 
   if ncommittee(V) == 0
      error("No committee found in potential")
   end
   return nothing 
end


"""
Construct a potential with committee:
* `basis` is an rpi basis (i.e. ACE basis)
* `c` is the coefficient vector for the actual prediction (this could be but 
   need not be the mean of the committee coefficients)
* `co_c` is a matrix where each column represents the coefficients of one 
   committee member    

Note, that the committee potential currently works only with the 
`StandardEvaluator`. The recursive committee evaluator still needs to be 
implemented. 
"""
function committee_potential(basis::RPIBasis, 
                             c::AbstractVector, 
                             co_c::AbstractMatrix)
   NZ = numz(basis.pibasis)
   NCO = size(co_c, 2)
   T = eltype(co_c)

   c_pi = get_picoeffs(basis, c)
   # convert committee coefficients first to a tuple of matrices ... 
   co_c_pi_pre = ntuple(iz0 -> basis.A2Bmaps[iz0]' * co_c[basis.Bz0inds[iz0], :],
                        NZ)
   # ... and now to a vector of SVectors 
   co_c_pi = ntuple(iz0 -> reinterpret(SVector{NCO, T}, co_c_pi_pre[iz0]')[:],
                    NZ)

   evaluator = StandardEvaluator()
   dags = ntuple(iz0 -> CorrEvalGraph{T, Int}(), NZ)
   
   return PIPotential(basis.pibasis, c_pi, dags, evaluator, co_c_pi)                              
end



co_evaluate(V::PIPotential, Rs, Zs, z0) = 
   co_evaluate!(alloc_temp(V, length(Rs)), V, Rs, Zs, z0)

co_evaluate!(tmp, V::PIPotential, Rs, Zs, z0) = 
      co_evaluate!(tmp, V, V.evaluator, Rs, Zs, z0)

# compute one site energy
function co_evaluate!(tmp, V::PIPotential, ::StandardEvaluator,
                      Rs::AbstractVector{JVec{T}},
                      Zs::AbstractVector{<:AtomicNumber},
                      z0::AtomicNumber) where {T}
   assert_has_co(V)
   iz0 = z2i(V, z0)
   A = evaluate!(tmp.tmp_pibasis.A, tmp.tmp_pibasis.tmp_basis1p,
                 V.pibasis.basis1p, Rs, Zs, z0)
   inner = V.pibasis.inner[iz0]
   c = V.coeffs[iz0]
   co_c = V.committee[iz0]

   Es = zero(T)
   NCO = ncommittee(V)
   co_Es = zero(SVector{NCO, T})

   for iAA = 1:length(inner)
      aa = one(Complex{T})
      for α = 1:inner.orders[iAA]
         aa *= A[inner.iAA2iA[iAA, α]]
      end
      Es += real(c[iAA] * aa)
      co_Es += real(co_c[iAA] * aa)
   end
   return Es, co_Es
end


