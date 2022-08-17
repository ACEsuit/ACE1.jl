
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

"""
return the coefficients for committee member i with center atom species z0.
"""
function get_committee_coeffs(V::PIPotential, z0::AtomicNumber, i::Integer)
   iz0 = z2i(V, z0)
   return [ x[i] for x in V.committee[iz0] ]
end



# ------------------------------------------------------------
#   Site energy


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



# ------------------------------------------------------------
#   Site energy gradient 

function co_evaluate_d(V::PIPotential, Rs, Zs, z0) 
   T = fltype(V) 
   NCO = ncommittee(V)
   dV = zeros(SVector{3, T}, length(Rs))
   co_dV = SVector(ntuple(_ -> copy(dV), NCO)...)
   tmpd = alloc_temp_d(V, length(Rs))
   return co_evaluate_d!(dV, co_dV, tmpd, V, Rs, Zs, z0)
end

co_evaluate_d!(dV, co_dV, tmpd, V::PIPotential, Rs, Zs, z0) =
      co_evaluate_d!(dV, co_dV, tmpd, V, V.evaluator, Rs, Zs, z0)


function co_evaluate_d!(dEs, co_dEs, tmpd, 
         V::PIPotential, ::StandardEvaluator,
         Rs::AbstractVector{JVec{T}},
         Zs::AbstractVector{<:AtomicNumber},
         z0::AtomicNumber) where {T}
   assert_has_co(V)
   iz0 = z2i(V, z0)
   NCO = ncommittee(V)

   basis1p = V.pibasis.basis1p
   tmpd_1p = tmpd.tmpd_pibasis.tmpd_basis1p
   Araw = tmpd.tmpd_pibasis.A
   c = V.coeffs[iz0]
   co_c = V.committee[iz0]

   # stage 1: precompute all the A values
   A = evaluate!(Araw, tmpd_1p, basis1p, Rs, Zs, z0)

   # stage 2: compute the coefficients for the ∇A_{klm} = ∇ϕ_{klm}
   dAco = tmpd.dAco
   # WARNING : allocation happens here, should profile whether this matters!
   co_dAco = zeros(SVector{NCO, eltype(dAco)}, length(dAco))
   inner = V.pibasis.inner[iz0]
   dAAt = tmpd.dAAt
   fill!(dAco, 0)
   for iAA = 1:length(inner)
      ord = inner.orders[iAA]
      Afwd = one(eltype(A))
      dAAt[1] = 1
      for α = 1:ord-1
         Afwd *= A[inner.iAA2iA[iAA, α]]
         dAAt[α+1] = Afwd
      end
      Abwd = one(eltype(A))
      for α = ord:-1:2
         Abwd *= A[inner.iAA2iA[iAA, α]]
         dAAt[α-1] *= Abwd
      end

      _c = c[iAA]
      _co_c = co_c[iAA]
      for α = 1:ord
         iAα = inner.iAA2iA[iAA, α]
         dAco[iAα] += _c * dAAt[α]
         co_dAco[iAα] += _co_c * dAAt[α]
      end
   end

   # stage 3: get the gradients
   fill!(dEs, zero(JVec{T}))
   for ico = 1:NCO 
      fill!(co_dEs[ico], zero(JVec{T}))
   end
   
   dAraw = tmpd.tmpd_pibasis.dA
   for (iR, (R, Z)) in enumerate(zip(Rs, Zs))
      evaluate_d!(Araw, dAraw, tmpd_1p, basis1p, R, Z, z0)
      iz = z2i(basis1p, Z)
      zinds = basis1p.Aindices[iz, iz0]
      for iA = 1:length(basis1p, iz, iz0)
         dEs[iR] += real(dAco[zinds[iA]] * dAraw[zinds[iA]])
         for ico = 1:NCO
            co_dEs[ico][iR] += real(co_dAco[zinds[iA]][ico] * dAraw[zinds[iA]])
         end
      end
   end

   return dEs, co_dEs
end



# ------------------------------------------------------------
#   Total Energy
#   this is basically a copy-paste of the JuLIP implementation
#   surely we can do this more generally ... 

using Base.Threads: nthreads, threadid, @threads
using JuLIP: neighbourlist, maxneigs
using JuLIP.Potentials: neigsz! 

function co_energy(V::PIPotential, at::AbstractAtoms)
   assert_has_co(V)
   nt = nthreads()
   NCO = ncommittee(V)
   T = fltype(V)
   tmp = [ alloc_temp(V, at) for _ in 1:nt ]
   E = [ zero(T) for _ in 1:nt ]
   co_E = [ zero(SVector{NCO, fltype(V)}) for _=1:nt ]
   return co_energy!(E, co_E, tmp, V, at)
end

function co_energy!(E, co_E, tmp, V, at)
   assert_has_co(V)
   NCO = ncommittee(V)
   nt = nthreads() 
   @assert nt == length(tmp) == length(E) == length(co_E)
   @assert all(length(co_E[i]) == NCO for i in 1:nt)
   nlist = neighbourlist(at, cutoff(V))
   @threads for i = 1:length(at) 
      tid = threadid() 
      z0 = at.Z[i] 
      j, Rs, Zs = neigsz!(tmp[tid], nlist, at, i)
      Es, co_Es = co_evaluate!(tmp[tid], V, Rs, Zs, z0)
      E[tid] += Es 
      co_E[tid] += co_Es 
   end
   return sum(E), sum(co_E)
end


# ------------------------------------------------------------
#   Total forces

function co_forces(V::PIPotential, at::AbstractAtoms)
   assert_has_co(V)
   nt = nthreads()
   NCO = ncommittee(V)
   T = fltype(V)
   tmp_d = [ alloc_temp_d(V, at) for _ in 1:nt ]
   F0 = zeros(JVec{T}, length(at))
   F = [ copy(F0) for _ in 1:nt ]
   co_F = [ SVector(ntuple(_ -> copy(F0), NCO)...) for _ in 1:nt ] 
   return co_forces!(F, co_F, tmp_d, V, at)
end

function co_forces!(F, co_F, tmp_d, V, at)
   assert_has_co(V)
   NCO = ncommittee(V)
   T = fltype(V)
   nt = nthreads() 
   @assert nt == length(tmp_d) == length(F) == length(co_F)
   @assert all(length(co_F[i]) == NCO for i in 1:nt)
   nlist = neighbourlist(at, cutoff(V))

   # allocate storage for the site energy gradients 
   dV0 = zeros(SVector{3, T}, maxneigs(nlist))
   dV = [ copy(dV0) for _ in 1:nt ]
   co_dV = [ SVector(ntuple(_ -> copy(dV0), NCO)...) for _ in 1:nt ]

   @threads for i = 1:length(at) 
      tid = threadid() 
      z0 = at.Z[i] 
      j, Rs, Zs = neigsz!(tmp_d[tid], nlist, at, i)
      co_evaluate_d!(dV[tid], co_dV[tid], tmp_d[tid], 
                                   V, Rs, Zs, z0)
      frc = F[tid] 
      co_frc = co_F[tid]                                   
      for a = 1:length(j)
         F[tid][j[a]] -= dV[tid][a]
         F[tid][i]    += dV[tid][a]
         for ico = 1:NCO
            co_F[tid][ico][j[a]] -= co_dV[tid][ico][a]
            co_F[tid][ico][i]    += co_dV[tid][ico][a]
         end
      end
   end
   return sum(F), sum(co_F)
end
