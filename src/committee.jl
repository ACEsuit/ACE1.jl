
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
   if !(size(co_c, 1) == length(c) == length(basis))
      error("""cofficient arrays don't match the basis size; 
               need `length(c) == size(co_c,1) == length(basis)`""")
   end
                        
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
   return PIPotential(basis.pibasis, c_pi, co_c_pi)
end


function PIPotential(pibasis::PIBasis, c::Tuple, co_c::Tuple) 
   NZ = numz(pibasis)
   T = eltype(c[1])
   evaluator = StandardEvaluator()
   dags = ntuple(iz0 -> CorrEvalGraph{T, Int}(), NZ)   
   return PIPotential(pibasis, c, dags, evaluator, co_c)                              
end


"""
return the coefficients for committee member i with center atom species z0.
"""
function get_committee_coeffs(V::PIPotential, z0::AtomicNumber, i::Integer)
   iz0 = z2i(V, z0)
   return [ x[i] for x in V.committee[iz0] ]
end


write_committee(::Nothing) = "nothing"

write_committee(committee::Tuple) = committee 

read_committee(str::String) = (@assert str == "nothing"; nothing)

function read_committee(committee::Union{AbstractVector, Tuple}) 
   NZ = length(committee)
   NCO = length(first(first(committee)))
   T = typeof(first(first(first(committee))))

   _read_committee(committee, TV, ::Val{NZ}) where {NZ}  = 
          ntuple(iz0 -> TV.(committee[iz0]), NZ)

   return _read_committee(committee, SVector{NCO, T}, Val(NZ))
end

write_committee(committee::Vector{<: StaticVector}) = 
            write_dict(collect(mat(committee)'))

function read_committee(committee::Dict) 
   co = read_dict(committee)
   @assert co isa AbstractMatrix 
   NCO = size(co, 2)
   T = eltype(co)
   return [ SVector{NCO, T}(co[i, :]) for i in 1:size(co, 1) ]
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
         V::PIPotential{T, NZ, TPI, TEV, NCO1}, ::StandardEvaluator,
         Rs::AbstractVector{JVec{T}},
         Zs::AbstractVector{<:AtomicNumber},
         z0::AtomicNumber) where {T, NZ, TPI, TEV, NCO1}
   assert_has_co(V)
   iz0 = z2i(V, z0)
   NCO = ncommittee(V)

   basis1p = V.pibasis.basis1p
   tmpd_1p = tmpd.tmpd_pibasis.tmpd_basis1p
   Araw = tmpd.tmpd_pibasis.A
   c = V.coeffs[iz0]
   co_c = V.committee[iz0]
   @assert eltype(c) == eltype(co_c[1])

   # stage 1: precompute all the A values
   A = evaluate!(Araw, tmpd_1p, basis1p, Rs, Zs, z0)

   # stage 2: compute the coefficients for the ∇A_{klm} = ∇ϕ_{klm}
   dAco = tmpd.dAco
   T_DACO = eltype(dAco)
   @assert T_DACO == promote_type(eltype(A), eltype(c))
   # WARNING : allocation happens here, should profile whether this matters!
   co_dAco = zeros(SVector{NCO, T_DACO}, length(dAco))
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
   
   _co_dEs = zeros(eltype(co_dEs[1]), NCO, length(Rs))

   dAraw = tmpd.tmpd_pibasis.dA
   for (iR, (R, Z)) in enumerate(zip(Rs, Zs))
      evaluate_d!(Araw, dAraw, tmpd_1p, basis1p, R, Z, z0)
      iz = z2i(basis1p, Z)
      zinds = basis1p.Aindices[iz, iz0]
      @inbounds for iA = 1:length(basis1p, iz, iz0)
         dEs[iR] += real(dAco[zinds[iA]] * dAraw[zinds[iA]])
         _dAraw = dAraw[zinds[iA]]
         _co_dAco = co_dAco[zinds[iA]]
         @simd ivdep for ico = 1:NCO
            # co_dEs[ico][iR] += real(_co_dAco[ico] * _dAraw)
            _co_dEs[ico, iR] += real(_co_dAco[ico] * _dAraw)
         end
      end
   end

   for ico = 1:NCO
      @inbounds for iR = 1:length(Rs)
         co_dEs[ico][iR] = _co_dEs[ico, iR]
      end
   end

   return dEs, co_dEs
end



# ------------------------------------------------------------
#   Total Energy
#   this is basically a copy-paste of the JuLIP implementation
#   surely we can do this more generally ... 

using Base.Threads: nthreads, threadid, @threads
using JuLIP: neighbourlist, maxneigs, AbstractCalculator
using JuLIP.Potentials: neigsz!, site_virial

function co_energy_alloc(V::AbstractCalculator, at::AbstractAtoms)
   assert_has_co(V)
   nt = nthreads()
   NCO = ncommittee(V)
   T = fltype(V)
   tmp = [ alloc_temp(V, at) for _ in 1:nt ]
   E = [ zero(T) for _ in 1:nt ]
   co_E = [ zero(SVector{NCO, T}) for _=1:nt ]
   return E, co_E, tmp
end

function co_energy(V::AbstractCalculator, at::AbstractAtoms)
   E, co_E, tmp = co_energy_alloc(V, at)
   return co_energy!(E, co_E, tmp, V, at)
end

function co_energy!(E, co_E, tmp, V::PIPotential, at)
   assert_has_co(V)
   NCO = ncommittee(V)
   nt = nthreads() 
   @assert nt == length(tmp) == length(E) == length(co_E)
   @assert all(length(co_E[i]) == NCO for i in 1:nt)
   nlist = neighbourlist(at, cutoff(V))
   @threads :static for i = 1:length(at) 
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

function co_forces_alloc(V::AbstractCalculator, at::AbstractAtoms)
   assert_has_co(V)
   nt = nthreads()
   NCO = ncommittee(V)
   T = fltype(V)
   tmp_d = [ alloc_temp_d(V, at) for _ in 1:nt ]
   F0 = zeros(JVec{T}, length(at))
   F = [ copy(F0) for _ in 1:nt ]
   co_F = [ SVector(ntuple(_ -> copy(F0), NCO)...) for _ in 1:nt ]
   return F, co_F, tmp_d 
end

function co_forces(V::AbstractCalculator, at::AbstractAtoms)
   F, co_F, tmp_d = co_forces_alloc(V, at)
   return co_forces!(F, co_F, tmp_d, V, at)
end

function co_forces!(F, co_F, tmp_d, V::PIPotential, at)
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

   @threads :static for i = 1:length(at) 
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


# ------------------------------------------------------------
#   Virial 

function co_virial_alloc(V::AbstractCalculator, at::AbstractAtoms)
   assert_has_co(V)
   nt = nthreads()
   NCO = ncommittee(V)
   T = fltype(V)
   tmp_d = [ alloc_temp_d(V, at) for _ in 1:nt ]
   v0 = zero(JMat{T})
   vir = [ copy(v0) for _ in 1:nt ]
   co_vir = [ MVector(ntuple(_ -> copy(v0), NCO)...) for _ in 1:nt ] 
   return vir, co_vir, tmp_d
end

function co_virial(V::AbstractCalculator, at::AbstractAtoms)
   vir, co_vir, tmp_d = co_virial_alloc(V, at)
   return co_virial!(vir, co_vir, tmp_d, V, at)
end

function co_virial!(vir, co_vir, tmp_d, V::PIPotential, at)
   assert_has_co(V)
   NCO = ncommittee(V)
   T = fltype(V)
   nt = nthreads() 
   @assert nt == length(tmp_d) == length(vir) == length(co_vir)
   @assert all(length(co_vir[i]) == NCO for i in 1:nt)
   nlist = neighbourlist(at, cutoff(V))

   # allocate storage for the site energy gradients 
   dV0 = zeros(SVector{3, T}, maxneigs(nlist))
   dV = [ copy(dV0) for _ in 1:nt ]
   co_dV = [ SVector(ntuple(_ -> copy(dV0), NCO)...) for _ in 1:nt ]

   @threads :static for i = 1:length(at) 
      tid = threadid() 
      z0 = at.Z[i] 
      j, Rs, Zs = neigsz!(tmp_d[tid], nlist, at, i)
      co_evaluate_d!(dV[tid], co_dV[tid], tmp_d[tid], 
                                   V, Rs, Zs, z0)
      vir[tid] += site_virial(dV[tid], Rs)
      for ico = 1:NCO
         co_vir[tid][ico] += site_virial(co_dV[tid][ico], Rs)
      end
   end 
   return sum(vir), SVector(sum(co_vir))
end



# ------------------------------------------------------------
#    Pair Potential Committee 

using ACE1.PairPotentials: PolyPairBasis, PolyPairPot


function committee_potential(basis::PolyPairBasis, 
                             c::AbstractVector, 
                             co_c::AbstractMatrix)

   if !(length(c) == size(co_c, 1) == length(basis))
      error("""cofficient arrays don't match the basis size; 
               need `length(c) == size(co_c,1) == length(basis)`""")
   end

   NCO = size(co_c, 2)
   T = eltype(co_c)

   co_c2 = [ SVector{NCO, T}(co_c[i, :]...) for i = 1:length(c) ]

   return PolyPairPot(c, basis, co_c2)
end


function assert_has_co(V::PolyPairPot) 
   if ncommittee(V) == 0
      error("No committee found in potential")
   end
   return nothing 
end


"""
return the coefficients for committee member i with center atom species z0.
"""
function get_committee_coeffs(V::PolyPairPot, ico::Integer)
   return [ x[ico] for x in V.committee ]
end



function _dot_zij_co(V, B, z, z0)
   i0 = ACE1.PairPotentials._Bidx0(V.basis, z, z0)  # cf. pair_basis.jl
   return sum( V.committee[i0 + n] * B[n]  for n = 1:length(V.basis, z, z0) )
end

function co_evaluate!(tmp, V::PolyPairPot, r::Number, z, z0) 
   Iz = z2i(V, z)
   Iz0 = z2i(V, z0)
   evaluate!(tmp.J[Iz, Iz0], tmp.tmp_J[Iz, Iz0], V.basis.J[Iz, Iz0], r, z, z0)  
   v = ACE1.PairPotentials._dot_zij(V, tmp.J[Iz, Iz0], z, z0)
   v_co = _dot_zij_co(V, tmp.J[Iz, Iz0], z, z0)
   return v, v_co
end


function co_evaluate_d!(tmp, V::PolyPairPot, r::Number, z, z0) 
   Iz = z2i(V, z)
   Iz0 = z2i(V, z0)
   evaluate_d!(tmp.J[Iz, Iz0], tmp.dJ[Iz, Iz0], tmp.tmpd_J[Iz, Iz0], 
               V.basis.J[Iz, Iz0], r, z, z0)
   dv = ACE1.PairPotentials._dot_zij(V, tmp.dJ[Iz, Iz0], z, z0)
   co_dv = _dot_zij_co(V, tmp.dJ[Iz, Iz0], z, z0)
   return dv, co_dv
end



function co_energy!(E, co_E, tmp, V::PolyPairPot, at)
   assert_has_co(V)
   NCO = ncommittee(V)
   nt = nthreads() 
   @assert nt == length(tmp) == length(E) == length(co_E) 
   @assert all(length(co_E[i]) == NCO for i in 1:nt)
   nlist = neighbourlist(at, cutoff(V))
   @threads :static for i = 1:length(at) 
      tid = threadid() 
      z0 = at.Z[i] 
      Js, Rs, Zs = neigsz!(tmp[tid], nlist, at, i)
      for (j, rr, z) in zip(Js, Rs, Zs)
         r = norm(rr)
         v, v_co = co_evaluate!(tmp[tid], V, r, z, z0)
         E[tid] += v/2
         co_E[tid] += v_co/2
      end
   end
   return sum(E), sum(co_E)
end


function co_forces!(F, co_F, tmp_d, V::PolyPairPot, at)
   assert_has_co(V)
   NCO = ncommittee(V)
   T = fltype(V)
   nt = nthreads() 
   @assert nt == length(tmp_d) == length(F) == length(co_F)
   @assert all(length(co_F[i]) == NCO for i in 1:nt)
   nlist = neighbourlist(at, cutoff(V))

   for i = 1:length(at) 
      tid = threadid() 
      z0 = at.Z[i] 
      Js, Rs, Zs = neigsz!(tmp_d[tid], nlist, at, i)
      for (j, rr, z) in zip(Js, Rs, Zs)
         r = norm(rr)
         r̂ = rr/r
         dv, co_dv = co_evaluate_d!(tmp_d[tid], V, r, z, z0)

         F[tid][j] -= 0.5 * dv * r̂
         F[tid][i] += 0.5 * dv * r̂
         for ico = 1:NCO
            co_F[tid][ico][j] -= 0.5 * co_dv[ico] * r̂
            co_F[tid][ico][i] += 0.5 * co_dv[ico] * r̂
         end
      end
   end
   return sum(F), sum(co_F)
end


function co_virial!(vir, co_vir, tmp_d, V::PolyPairPot, at)
   assert_has_co(V)
   NCO = ncommittee(V)
   T = fltype(V)
   nt = nthreads() 
   @assert nt == length(tmp_d) == length(vir) == length(co_vir)
   @assert all(length(co_vir[i]) == NCO for i in 1:nt)
   nlist = neighbourlist(at, cutoff(V))

   dV0 = zeros(SVector{3, T}, maxneigs(nlist))
   dV = [ copy(dV0) for _ in 1:nt ]
   co_dV = [ MVector(ntuple(_ -> copy(dV0), NCO)...) for _ in 1:nt ]

   @threads :static for i = 1:length(at) 
      tid = threadid() 
      z0 = at.Z[i] 
      Js, Rs, Zs = neigsz!(tmp_d[tid], nlist, at, i)
   
      for (a, (j, rr, z)) in enumerate( zip(Js, Rs, Zs) )
         r = norm(rr)
         r̂ = rr/r
         dv, co_dv = co_evaluate_d!(tmp_d[tid], V, r, z, z0)
         dV[tid][a] = 0.5 * dv * r̂
         for ico = 1:NCO
            co_dV[tid][ico][a] = 0.5 * co_dv[ico] * r̂
         end
      end

      vir[tid] += site_virial(dV[tid], Rs)
      for ico = 1:NCO
         co_vir[tid][ico] += site_virial(co_dV[tid][ico], Rs)
      end
   end 
   return sum(vir), SVector(sum(co_vir))
end


# -----------------  IPCOllection

using JuLIP.MLIPs: SumIP

# energy(sumip::SumIP, at::AbstractAtoms; kwargs...) =
#          sum(energy(calc, at; kwargs...) for calc in sumip.components)
# forces(sumip::SumIP, at::AbstractAtoms; kwargs...) =
#          sum(forces(calc, at; kwargs...) for calc in sumip.components)
# virial(sumip::SumIP, at::AbstractAtoms; kwargs...) =
#          sum(virial(calc, at; kwargs...) for calc in sumip.components)

function co_energy(V::SumIP, at::AbstractAtoms; kwargs...)
   ee = [ co_energy(calc, at; kwargs...) for calc in V.components ]
   return sum( e[1] for e in ee ), sum( e[2] for e in ee )
end

function co_forces(V::SumIP, at::AbstractAtoms; kwargs...)
   ff = [ co_forces(calc, at; kwargs...) for calc in V.components ]
   return sum( f[1] for f in ff ), sum( f[2] for f in ff )
end

function co_virial(V::SumIP, at::AbstractAtoms; kwargs...)
   vv = [ co_virial(calc, at; kwargs...) for calc in V.components ]
   return sum( v[1] for v in vv ), sum( v[2] for v in vv )
end


# ------------------------------------------------------------
#    One Body Committee 

using JuLIP: OneBody

struct FlexConstTensor{T} 
   v::T 
end

import Base: +

+(a::FlexConstTensor, b::AbstractArray) = b .+ Ref(a.v)
+(b::AbstractArray, a::FlexConstTensor) = b .+ Ref(a.c)

function co_energy(V::OneBody, at::AbstractAtoms; kwargs...)
   E = energy(V, at) 
   return E, FlexConstTensor(E)
end

function co_forces(V::OneBody, at::AbstractAtoms; kwargs...)
   F = forces(V, at)
   return F, FlexConstTensor(F)
end

function co_virial(V::OneBody, at::AbstractAtoms; kwargs...)
   vir = virial(V, at)
   return vir, FlexConstTensor(vir)
end


# ------------------------------------------------------------
#    Committee for superbasis 

import JuLIP.MLIPs: IPSuperBasis

function committee_potential(basis::IPSuperBasis, 
                             c::AbstractVector, 
                             co_c::AbstractMatrix)
   idx = 0
   co_pots = []
   for b in basis.BB
      len = length(b)
      Ib = idx .+ (1:len)
      co_pot = committee_potential(b, c[Ib], co_c[Ib,:])
      push!(co_pots, co_pot)
      idx += len
   end
   return JuLIP.MLIPs.SumIP(co_pots)
end
