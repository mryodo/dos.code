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

function δ( ind; N = 20)
      res = zeros( N, 1)
      res[ ind ] = 1
      return res
end

#----------------------------------------------------------------------------#
#------------------------Functions space-------------------------------------#
#----------------------------------------------------------------------------#

include("../sample/pointCloudGenerator.jl")
include("../sample/generateDelaunay.jl")

"""
    getLDoSInt( Λ, Q, L1up, L1down; Δx = 0.01, ϵ = 0.01)

TBW
"""
function getLDoSInt( Λ, Q, L1up, L1down; Δx = 0.01, ϵ = 0.01, λ)
      Λ1 = 2 / (λ+ϵ) * Λ .- 1
      @assert Λ1[2]-Λ1[1] < Δx "Oh, shit's fucked, ain't it?"
      xs = -1-1e-6:Δx:1+1e-6

      E = zeros( size(xs, 1) - 1, size(Λ, 1) )
      for i in axes(E, 2)
            for j in 1:size(xs, 1)-1
                  E[j, i] = sum(Q[i, ( Λ1 .>= xs[j] ) .& ( Λ1 .<= xs[j+1] )] .^2)
            end
      end

      components = zeros( size(Λ, 1), 3 )



      for i in axes( components, 1 )
            for j in axes( components, 1 )
                  uptest = norm( L1up * Q[:, j] )
                  downtest = norm( L1down * Q[:, j] )
                  if abs( Λ[j] ) < 1e-6
                        components[i, 1] += Q[ i,j ]^2
                  end
                  if abs( uptest ) < 1e-6 && abs( downtest ) > 1e-6
                        components[i, 2] += Q[ i,j ]^2
                  end
                  if abs( uptest ) > 1e-6 && abs( downtest ) < 1e-6
                        components[i, 3] += Q[ i,j ]^2
                  end
            end
      end

      return E, components
end

function getLDoSIntDist( E1, E2; Δx = 0.01, ϵ = 0.01)
    k1 = size(E1, 2)
    k2 = size(E2, 2)

    distances = zeros( k1, k2 )

      for i in axes( distances, 1 )
            for j in axes( distances, 2 )
                  distances[i, j] = sqrt(sum( ( E1[:, i] - E2[:, j] ) .^ 2 ) * Δx)
            end
      end
      return distances
end



N = 20
points = getPoints( N )

m0, edges2, trians = getEdgesTriansFromPoints( N, points; ϵ = 10 )
B1full = B1fromEdges( m0, edges2 )
B2full = B2fromTrig( edges2, trians)
distances = [ norm( points[:, edges2[i, 1]] - points[:, edges2[i, 2]] ) for i in axes(edges2, 1) ] 





ϵ = 2.5
mask = distances .<= ϵ
ind = findall( mask )
B1 = B1full[:, ind]
B2 = B2full[ind, :]
B2 = B2[:, sum(abs.(B2); dims = 1)' .== 3]

L1 = B1' * B1 + B2 * B2'
Ld = B1' * B1
Lu = B2 * B2'
L0 = B1 * B1'
λ = max( eigs(Lu, which=:LM, nev=1)[1], eigs(Ld, which=:LM, nev=1)[1] )[1]

H = 2 / ( λ + 1e-3 ) * L1 - ( 1 - 1e-6 ) * I
Hd = 2 / ( λ + 1e-3 ) * Ld - ( 1 - 1e-6 ) * I
Hu = 2 / ( λ + 1e-3 ) * Lu - ( 1 - 1e-6 ) * I
H0 = 2 / ( λ + 1e-3 ) * L0 - ( 1 - 1e-6 ) * I
σ =  eigvals( Matrix( L1 ) )
σ1 =  eigvals( Matrix( H ) )

#---------------------------------------------------------------------------#

function getSpectralGap(Λ)
      if abs(Λ[1]) > 1e-8
            return false, 0.02
      end
      return true, minimum([0.02, 0.9*minimum( Λ[ abs.(Λ) .> 1e-8 ] )])
end

function LDoS( Λnew, Qnew, λ ; Δx = 0.02, ϵ = 0.01 )
      Λ1 = 2 / (λ+ϵ) * Λnew .- 1
      #@assert Λ1[2]-Λ1[1] > Δx "Oh, shit's fucked, ain't it?"
      xs = -1-1e-6:Δx:1+1e-6

      E = zeros( size(xs, 1) - 1, size(Λ, 1) )
      for i in axes(E, 2)
            for j in 1:size(xs, 1)-1
                  E[j, i] = sum(Qnew[i, ( Λ1 .>= xs[j] ) .& ( Λ1 .<= xs[j+1] )] .^2)
            end
      end
      return E
end

#----------------------------- we redesign LDoS-----------------------------#
Λ, Q = eigen(Matrix(L1))
Λu, Qu = eigen(Matrix(Lu))
Qu_plus = Qu[ : , abs.(Λu) .> 1e-8 ]
Λd, Qd = eigen(Matrix(Ld))
Qd_plus = Qd[ : , abs.(Λd) .> 1e-8 ]
Qh = Q[ : , abs.(Λ) .< 1e-8 ]
Qnew = [ Qh Qd_plus Qu_plus ]

Λnew = [ norm( (L1 * Qnew)[:, i]) for i in axes(L1, 2)] 

norm( sort( Λ ) - sort( Λnew ) )


λ = Λ[end]
flag, Δx = getSpectralGap( 2 * Λ / λ )
E = LDoS( Λnew, Qnew, λ; Δx = Δx )
Eu = LDoS( Λu, Qu, λ; Δx = Δx )
Ed = LDoS( Λd, Qd, λ; Δx = Δx )

E_decomp = Eu + Ed
E_decomp[1, : ] .-= 1

norm.(eachrow(Eu[2:end, :]'), 1)
norm.(eachrow(Ed[2:end, :]'), 1)

norm(E - E_decomp)



#----------------------------- we redesign LDoS-----------------------------#




Λ, Q = eigen(Matrix(L1))
E, components = getLDoSInt( Λ, Q, Lu, Ld; Δx = 0.1 , λ = Λ[end])

maskcurl = abs.( [ norm( ( Lu * Q )[:, i] ) for i in axes(Lu, 1) ]  ) .> 1e-6
maskgrad = abs.( [ norm( ( Ld * Q )[:, i] ) for i in axes(Ld, 1) ]  ) .> 1e-6
( maskcurl + maskgrad ) .> 1




Λup, Qup = eigen(Matrix(Lu))
Λdown, Qdown = eigen(Matrix(Ld))
Eup, components = getLDoSInt( Λup, Q, Lu, Ld; Δx = 0.1 , λ = Λ[end])
Edown, components = getLDoSInt( Λdown, Q, Lu, Ld; Δx = 0.1, λ = Λ[end] )

E_decomp = Eup + Edown

sum(E_decomp[1:end, :]; dims = 1)



#---------------------------------------------------------------------------#

