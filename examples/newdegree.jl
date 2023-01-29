
# --------------------------------------------------------------------------
# ACE1.jl: Julia implementation of the Atomic Cluster Expansion
# Copyright (c) 2019 Christoph Ortner <christophortner0@gmail.com>
# Licensed under ASL - see ASL.md for terms and conditions.
# --------------------------------------------------------------------------


# A demonstration how to use the new SparsePSHDegreeM degree type
# see also ?ACE1.RPI.SparsePSHDegreeM

using ACE1, JuLIP
using ACE1: z2i, i2z, order
##

zTi = AtomicNumber(:Ti)
zAl = AtomicNumber(:Al)

# the weights for the n
Dn = Dict( "default" => 1.0,
            (zTi, zAl) => 2.0,    # weak interaction between species
            (zAl, zTi) => 2.0,    # weak interaction between species
            (zAl, zAl) => 1.2,    # slightly smaller basis for Al than for Ti
            (2, zTi, zAl) => 0.8  # but for 3-body we override this ...
         )

# the weights for the l
Dl = Dict( "default" => 1.5, )    # let's do nothing special here...
                                  # but same options are available for for n

# the degrees
Dd = Dict( "default" => 10,
           1 => 20,     # N = 1
           2 => 20,     # ...
           3 => 15,
           (2, zTi) => 25   # an extra push for the 3-body Ti basis
        )                   # (probably a dumb idea, just for illustration!)

Deg = ACE1.RPI.SparsePSHDegreeM(Dn, Dl, Dd)

##

# generate basis
# - note that degree is already incorporated into Deg
#   but we can still enlarge it e.g. by using maxdeg = 1.2, 1.5, 2.0, ...
basis = ACE1.Utils.rpi_basis(species = [:Ti, :Al],
                              N = 5,
                              r0 = rnn(:Ti),
                              D = Deg,
                              maxdeg = 1)

##
# analyse the basis a bit to see the effect this had

iAl = z2i(basis, AtomicNumber(:Al))
iTi = z2i(basis, AtomicNumber(:Ti))

println("The Ti site has more basis functions than the Al site")
@show length(basis.pibasis.inner[iAl])
@show length(basis.pibasis.inner[iTi])

specTi = collect(keys(basis.pibasis.inner[iTi].b2iAA))
specTi2 = specTi[ order.(specTi) .== 2 ]
specTi3 = specTi[ order.(specTi) .== 3 ]

println("The Ti-2N interaction has more basis functions than Ti-3N interaction")
@show length(specTi2)
@show length(specTi3)

specTi2_Ti = specTi2[ [ all(b.z == zTi for b in B.oneps) for B in specTi2 ] ]
specTi2_Al = specTi2[ [ all(b.z == zAl for b in B.oneps) for B in specTi2 ] ]

println("The Ti-Ti interaction has more basis functions than Ti-Al interaction")
@show length(specTi2_Ti)
@show length(specTi2_Al)

##
#
# specAl = collect(keys(basis.pibasis.inner[iAl].b2iAA))
# specTi = collect(keys(basis.pibasis.inner[iTi].b2iAA))
# spec = [specAl; specTi]
# I_ord2 = findall( order.(spec) .== 1 )
#
# ACE1.degree.(Ref(Deg), spec[I_ord2])
#
# spec2 = spec[I_ord2]
# ns_spec2 = [ b.oneps[1].n for b in spec2 ]

##

@info("Check the simplified interface")
@info("Test 1")
Dn = Dict( "default" => 1.0, )
Dl = Dict( "default" => 1.5, ) 
Dd = Dict( 1 => 20, 2 => 20, 3 => 15, 4 => 10)
Deg = ACE1.RPI.SparsePSHDegreeM(Dn, Dl, Dd)
basis1 = ACE1.Utils.rpi_basis(species = :X,
                              N = 4,
                              r0 = 1.0,
                              D = Deg,
                              maxdeg = 1)

basis2 = ACE1.Utils.rpi_basis(species = :X,
                              N = 4,
                              r0 = 1.0,
                              wL = 1.5, 
                              maxdeg = [20, 20, 15, 10])

@assert basis1 == basis2 

##

@info("Test 2")

N = 4
Dn = Dict( "default" => 1.0, )
Dl = Dict( 1 => 1.0, 2 => 1.5, 3 => 1.75, 4 => 2.0, ) 
Dd = Dict( 1 => 24, 2 => 22, 3 => 16, 4 => 11)
Deg = ACE1.RPI.SparsePSHDegreeM(Dn, Dl, Dd)
basis1 = ACE1.Utils.rpi_basis(species = [:Ti, :Al],
                              N = N,
                              r0 = 1.0,
                              D = Deg,
                              maxdeg = 1)

basis2 = ACE1.Utils.rpi_basis(species = [:Ti, :Al],
                              N = N,
                              r0 = 1.0,
                              wL = [Dl[n] for n = 1:N], 
                              maxdeg = [Dd[n] for n = 1:N])

@assert basis1 == basis2 

##

