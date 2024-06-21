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
n = N + 4
H, Hd, Hu, σ1, L1, Ld, Lu, σ, m1,  α, B1, B2, L0, H0, edges2, trians = generateDelaunayMatrices( N, ν = 0.75 );
W2 = spdiagm( 0.5 * rand( size( B2, 2)) .+ 0.5 )
m2 = size( trians, 1 )


L1 = B2 * W2 * W2 * B2' 
p = getExactMeasure( B2, L1, W2 )


m0 = n

@btime pinv( Matrix( L0 ) );
tmp = B2 * randn( m2, Int(round((m1-m0)/3)));
@btime qr( tmp );



1
















trians2 = trians[1:2, :]
Π =  1 ./ ( m2 * p ) 

seq = [1; 2; 1; 3; 4; 5; 7]

function getMultMatrix(seq, m2)
      dict = countmap( seq )
      mult = zeros(Int, m2 )
      for key in keys(dict)
            mult[ key ] = dict[ key ]
      end
      return mult
end

function compareSpectral( A, B )
      return maximum( [ eigmax( Matrix(A - B) ), eigmax( Matrix(B - A) ) ] )
end


function sparsificationProgress( p, L1, B2, W2, m1, m2; rep = 12, ν = 30, step = 30 )
      dists = zeros( rep, size(4:step:ν*round( m1*log(m1) ), 1)) 
      for jj in 1:rep
            seq = sample( 1:m2, Weights(p), 3)
            dist = []
      
            for i in 4:step:ν*round( m1*log(m1) )
                  seq = [seq; sample( 1:m2, Weights(p), 10)]
                  mult = getMultMatrix( seq, m2 )
                  Π =  1 ./ ( size( seq, 1) * p ) 
                  dist = [dist; compareSpectral( L1, B2 * W2 * W2 * diagm( Π .* mult ) * B2' ) ]
            end
            dists[jj, :] = dist
      end
      return dists, mean(dists; dims=1)'
end

function perturbMeasure( p; ϵ = 1e-4 )
      δ = ϵ * randn( size( p, 1 ) )
      p1 = p + δ
      p1[ p1 .< 0 ] .= 1e-10
      return p1 / sum( p1 )
end

_, meanDist = sparsificationProgress( p, L1, B2, W2, m1, m2; rep = 12, ν = 30, step = 30 )
p1 = perturbMeasure( p; ϵ = 1e-2 * 1/m2 )
_, meanDist1 = sparsificationProgress( p1, L1, B2, W2, m1, m2; rep = 12, ν = 30, step = 30 )
p2 = perturbMeasure( p; ϵ = 1e-1 * 1/m2 )
_, meanDist2 = sparsificationProgress( p2, L1, B2, W2, m1, m2; rep = 12, ν = 30, step = 30 )
p3 = perturbMeasure( p; ϵ = 2*1e-1 * 1/m2 )
_, meanDist3 = sparsificationProgress( p3, L1, B2, W2, m1, m2; rep = 12, ν = 30, step = 30 )
p4 = perturbMeasure( p; ϵ = 4*1e-1 * 1/m2 )
_, meanDist4 = sparsificationProgress( p4, L1, B2, W2, m1, m2; rep = 12, ν = 30, step = 30 )

begin
      plot()
      plot!( meanDist, lw = 3, color = cols[1], labels="unperturbed", alpha=0.5 )
      plot!( meanDist1, lw = 3, color = cols[2], labels=L"\varepsilon=\frac{1}{100 m_2}", alpha=0.5 )
      plot!( meanDist2, lw = 3, color = cols[3], labels=L"\varepsilon=\frac{1}{10 m_2}", alpha=0.5 )
      plot!( meanDist3, lw = 3, color = cols[4], labels=L"\varepsilon=\frac{1}{5 m_2}", alpha=0.5 )
      plot!( meanDist4, lw = 3, color = cols[6], labels=L"\varepsilon=\frac{2}{5 m_2}", alpha=0.5 )
      plot!( yscale = :log10, legend = :topright )
end

savefig("perturbed_sparse.tex")
savefig("perturbed_sparse.pdf")






rep = 25

dists = zeros( rep, size(4:10:30*round( m1*log(m1) ), 1)) 

for jj in 1:rep
      seq = sample( 1:m2, Weights(p), 3)
      dist = []

      for i in 4:10:30*round( m1*log(m1) )
            seq = [seq; sample( 1:m2, Weights(p), 10)]
            mult = getMultMatrix( seq, m2 )
            Π =  1 ./ ( size( seq, 1) * p ) 
            dist = [dist; compareSpectral( L1, B2 * W2 * W2 * diagm( Π .* mult ) * B2' ) ]
      end
      dists[jj, :] = dist
end

begin
      plot()
      plot!( dists', lw=4, color = cols[2], alpha=0.1, labels="" )
      plot!( mean(dists; dims=1)', lw=4, color = cols[1], labels="" )
      plot!( yscale = :log10 )
end


##########





































##################################################