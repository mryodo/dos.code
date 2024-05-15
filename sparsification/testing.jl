using LinearAlgebra
using Random, Distributions
using SparseArrays
using Polynomials
using StatsBase
using GR: delaunay
using Arpack

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

N = 40
n = N + 4
H, Hd, Hu, σ1, L1, Ld, Lu, σ, m1,  α, B1, B2, L0, H0, edges2, trians = generateDelaunayMatrices( N, ν = 0.75 );
W2 = spdiagm( 0.5 * rand( size( B2, 2)) .+ 0.5 )

function sparseSignMatrix( n, m, d)
      X = zeros(n, m )
      for i in 1:m
            pos = sample( 1:n, d; replace = false )
            X[pos, i] .= sample([-1, 1])
      end
      return sparse(X)
end

Ω = sparseSignMatrix( size(B2, 2), size(B2, 1) - n + 1, 3 )
print(m1)
@btime qr(B2 * Ω );
@btime B1' * pinv(Matrix( B1 * B1' )) * B1;


R = B2 * Ω;
for i in axes(R, 2)
      R[:, i] = R[:, i] / norm(R[:, i])
end

sum( abs.( R' * R - I ) .> 0.1) / ( size(R, 1)^2)









L1 = B2 * W2 * W2 * B2'

@btime getExactMeasure( B2, L1, W2 ) ;



countTriansperNode = zeros(n )
for i in axes(trians, 1)
      countTriansperNode[ trians[i, 1] ] += 1
      countTriansperNode[ trians[i, 2] ] += 1
      countTriansperNode[ trians[i, 3] ] += 1
end

setTrian = Set( [ trians[ i, : ] for i in axes( trians, 1 ) ] )
tethras = zeros( Int, binomial(n, 4), 4)

ind = 1
for ii in 1:size(trians, 1)
      i, j, k = trians[ii, :]
      for l in k+1:n 
            if ~( l in trians[ii, :] )
                  if ([ i, j, l] in setTrian) && ([ i, k, l] in setTrian) && ([ j, k, l] in setTrian)
                        tethras[ind, :] = [ i j k l ]
                        ind += 1
                  end
            end
      end
end

tethras = tethras[ 1:sum( sum( tethras; dims = 2 ) .> 1) , :]


function B3fromTethras(trigs, tethras)
      
      m2 = size(trigs, 1);
      if length(tethras)==1
          return spzeros(m2, 1)
      end
      m3 = size(tethras, 1);
      B3 = spzeros(m2, m3);
      
      for i in 1:m3
            B3[findfirst( all( [ tethras[i, 1], tethras[i, 2], tethras[i, 3]]' .== trigs, dims=2 )[:, 1] ), i]=-1;
            B3[findfirst( all( [ tethras[i, 1], tethras[i, 2], tethras[i, 4]]' .== trigs, dims=2 )[:, 1] ), i]=1;
            B3[findfirst( all( [ tethras[i, 1], tethras[i, 3], tethras[i, 4]]' .== trigs, dims=2 )[:, 1] ), i]=-1;
            B3[findfirst( all( [ tethras[i, 2], tethras[i, 3], tethras[i, 4]]' .== trigs, dims=2 )[:, 1] ), i]=1;
      end 
      return B3
end


B3 = B3fromTethras( trians, tethras )













@btime [4, 5, 6] in setTrian;


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