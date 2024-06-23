using LinearAlgebra
using Random, Distributions
using SparseArrays
using Polynomials
using StatsBase
using GR: delaunay
using Arpack
using StatsBase

using BenchmarkTools


using Plots, ColorSchemes, LaTeXStrings
pgfplotsx()
theme(:dao)
Plots.scalefontsizes(1.5)
cols=[ # vibrant
      colorant"#e73", # orange
      colorant"#07b", # blue
      colorant"#3be", # cyan
      colorant"#e37", # magenta
      colorant"#c31", # red
      colorant"#098", # teal
      colorant"#bbb", # grey
];

#----------------------------------------------------------------------------#
#------------------------Functions space-------------------------------------#
#----------------------------------------------------------------------------#

include("pointCloudGenerator.jl")
include("generateDelaunay.jl")



function getExactMeasure( B2, L1, W2 )
      L1inv = pinv( Matrix( L1 ) )
      r = W2 * W2 * diag( B2' * L1inv * B2 )
      return r / sum( r )
end



N = 20
m0 = N + 4
H, Hd, Hu, σ1, L1, Ld, Lu, σ, m1,  α, B1, B2, L0, H0, edges2, trians = generateDelaunayMatrices( N, ν = 0.75 );
m2 = size( trians, 1 )


W1 = spdiagm( 0.5 * rand( m1 ) .+ 0.5 )

L0 = B1 * W1 * W1 * B1' 
@btime pinv( Matrix( L0 ) );
#   70.041 μs (21 allocations: 45.55 KiB)

@btime getExactMeasure( B1, L0, W1 );
#   98.667 μs (29 allocations: 426.33 KiB)

Ld = W1 * B1' * B1 * W1
λ = eigs(Ld, which=:LM, nev=1)[1][1]
H = 1 / ( λ + 1e-6 ) * Ld 



function exactLDoS( k, σ, Q ; h = 1e-3 )
      bins = vec( 0:h:1)
      return [ sum( Q[k, ( σ .< bins[ i+ 1 ] ) .&& ( σ .>= ( (i == 1 ) ? -1e-6 : bins[i] ) )] .^ 2 ) for i in 1:size( bins, 1 ) - 1 ]
end

function getDmkExact( H, M, m1 )
      dmk = zeros( m1, size(1:2:M, 1) )
      T0 = I
      T1 = H
      for i in 1:2:M
            dmk[:, Int( ( i + 1 ) / 2 )] = diag( T1 )
            T0 = 2 * H * T1 - T0
            T1 = 2 * H * T0 - T1
      end
      return dmk
end

function fromDmk2LDoS( dmk, M; h = 1e-3 )
      bins = vec( 0:h:1)
      intTm = zeros( size(1:2:M, 1), size(bins, 1) - 1  )
      for m in 1:2:M
            intTm[ Int( (m+1) /2 ), : ] = ( sin.( m * acos.( bins[1:end-1] ) ) - sin.( m * acos.( bins[2:end] ) ) ) / m
      end
      return dmk * intTm
end





σ, Q = eigen( Matrix( H ) )
μk = exactLDoS( 1, σ, Q; h = 1/20 )
ldos = reduce( hcat, [ exactLDoS( k, σ, Q; h =1/20 ) for k in 1:m1 ] )

M = 21
dmk = getDmkExact( H, M, m1 )
dLDoS = fromDmk2LDoS( dmk, M; h = 1/20 )
err = norm( (ldos - dLDoS')[2:end, :], Inf ) / norm( ldos[2:end, :], Inf )

measure1 = sum( ldos[2:end, :] ; dims=1)'
measure1 = measure1 / sum( measure1 )
measure2 = sum( (dLDoS')[2:end, :] ; dims=1)'
measure2 = measure2 / sum( measure2 )

test = getExactMeasure( B1, L0, W1 )
"""
    getOddMomentsExact( σ1; moments_num = 1000 )

TBW
"""
function getOddMomentsExact( σ; M = 1000 )
      return [ diag( cos.( m * acos.( σ ) ) ) for m in 1:2:M-1 ]
end

σ = eigvals( Matrix( H ) )
getOddMomentsExact( σ )