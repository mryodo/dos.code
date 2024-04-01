using LinearAlgebra
using Random, Distributions
using SparseArrays
using Polynomials
using StatsBase
using GR: delaunay
using Arpack

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


123

N = 8
H, Hd, Hu, σ1, L1, Ld, Lu, σ, m1, α, B1, B2 = generateDelaunayMatrices( N );
W2 = spdiagm( 0.5 * rand( size( B2, 2)) .+ 0.5 )

L1 = B2 * W2 * W2 * B2'

r = diag( B2' * pinv( Matrix( L1 ) ) * B2 )

U, S, V = svd( Matrix( B2 * W2 ) )

x = rand(m1, 1)
norm( U * pinv( diagm( S ) )^2 * U' * x - pinv( Matrix( L1 ) )* x )

r2 = diag( B2' * U * pinv( diagm( S ) ) * U' * B2 )

S2 = deepcopy( S )
S2[ abs.(S2) .> 1e-10 ] = 1 ./ S2[ abs.(S2) .> 1e-10 ]

r3 = diag( inv(Matrix(W2)) * V * diagm(S) * pinv( diagm( S ) ) * diagm(S) * V' * inv(Matrix(W2)) )

r4 = diag( inv(Matrix(W2)) * V * diagm(S) * V' * inv(Matrix(W2)) )

r5 = diag( inv(Matrix(W2)) * W2 * B2' * B2 * W2 * inv(Matrix(W2)) )


[r2 r3 r4 ] 