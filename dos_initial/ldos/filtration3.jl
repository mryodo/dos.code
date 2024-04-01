using LinearAlgebra
using Random, Distributions
using SparseArrays
using Polynomials
using StatsBase
using GR: delaunay
using Arpack

using Plots, ColorSchemes, LaTeXStrings, StatsPlots
pgfplotsx()
theme(:dao)
Plots.scalefontsizes(2.25)
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

include("../sample/pointCloudGenerator.jl")
include("../sample/generateDelaunay.jl")

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

      E = zeros( size(xs, 1) - 1, size(Λnew, 1) )
      for i in axes(E, 2)
            for j in 1:size(xs, 1)-1
                  E[j, i] = sum(Qnew[i, ( Λ1 .>= xs[j] ) .& ( Λ1 .<= xs[j+1] )] .^2)
            end
      end
      return E
end

function getDecomposedLDOSNorms( ϵ, distances, B1full, B2full )
      mask = distances .<= ϵ
      ind = findall( mask )
      B1 = B1full[:, ind]
      B2 = B2full[ind, :]
      B2 = B2[:, sum(abs.(B2); dims = 1)' .== 3]
      
      L1 = B1' * B1 + B2 * B2'
      Ld = B1' * B1
      Lu = B2 * B2'
      #λ = max( eigs(Lu, which=:LM, nev=1)[1], eigs(Ld, which=:LM, nev=1)[1] )[1]

      Λ, Q = eigen(Matrix(L1))
      λ = Λ[end]
      Λu, Qu = eigen(Matrix(Lu))
      Qu_plus = Qu[ : , abs.(Λu) .> 1e-8 ]
      Λd, Qd = eigen(Matrix(Ld))
      Qd_plus = Qd[ : , abs.(Λd) .> 1e-8 ]
      Qh = Q[ : , abs.(Λ) .< 1e-8 ]
      Qnew = [ Qh Qd_plus Qu_plus ]

      Λnew = [ norm( (L1 * Qnew)[:, i]) for i in axes(L1, 2)] 
      @assert norm( sort( Λ ) - sort( Λnew ) ) < 1e-8 "reordering failed! Go back to basics!"
      flag, Δx = getSpectralGap( 2 * Λ / λ )
      #print(flag)
      E = LDoS( Λnew, Qnew, λ; Δx = Δx )
      Eu = LDoS( Λu, Qu, λ; Δx = Δx )
      Ed = LDoS( Λd, Qd, λ; Δx = Δx )

      E_decomp = Eu + Ed
      E_decomp[1, : ] .-= 1

      return norm.(eachrow(Eu[2:end, :]'), 1), norm.(eachrow(Ed[2:end, :]'), 1), norm(E - E_decomp) < 1e-8, ind
end

function checkTrian( ϵ, trian, points )
      return ( norm( [ points[1, trian[1]] - points[1, trian[2]]; points[2, trian[1]] - points[2, trian[2]] ] ) < ϵ ) && ( norm( [ points[1, trian[1]] - points[1, trian[3]]; points[2, trian[1]] - points[2, trian[3]] ] ) < ϵ ) && ( norm( [ points[1, trian[3]] - points[1, trian[2]]; points[2, trian[3]] - points[2, trian[2]] ] ) < ϵ )
end

#---------------------------------------------------------------------------#

#N = 10
#points = getPoints( N; dist = 5 )
N = 6
#points = [
#      0  1.0 2.5 3.2 1.0 2.3 -0.5 0.6 2.4 3.0 0 1.2;
#      0 -0.5 -0.3 0.5 -1.0 -1.2 1.5 2.0 2.2 2.3 2.5 3.5
#]
points = getPoints3( N; dist = 5 )

m0, edges2, trians = getEdgesTriansFromPoints( 3*N, points; ϵ = 10 )
B1full = B1fromEdges( m0, edges2 )
B2full = B2fromTrig( edges2, trians)
distances = [ norm( points[:, edges2[i, 1]] - points[:, edges2[i, 2]] ) for i in axes(edges2, 1) ] 

perm = sortperm( distances )

#=
ϵ = 3.5
curlNorms, gradNorms, flag, ind = getDecomposedLDOSNorms( ϵ, distances, B1full, B2full ) 
placement = [ findall( perm[1:size(ind, 1)] .== ind[i] )[1] for i in 1:size(ind,1) ]
=#
#filters = 0.5:0.05:7.5
filters = sort(distances) .+ 1e-8
decompH = zeros( size(distances, 1), size( filters, 1 ) )
decompD = zeros( size(distances, 1), size( filters, 1 ) )
decompU = zeros( size(distances, 1), size( filters, 1 ) )


for i in eachindex(filters)
      ϵ = filters[i]
      
      curlNorms, gradNorms, flag, ind = getDecomposedLDOSNorms( ϵ, distances, B1full, B2full ) 
      placement = [ findall( perm[1:size(ind, 1)] .== ind[i] )[1] for i in eachindex(ind) ]
      decompD[ 1 : size(ind, 1), i ] = gradNorms[ placement ]
      decompU[ 1 : size(ind, 1), i ] = curlNorms[ placement ]
      decompH[ 1 : size(ind, 1), i ] = - gradNorms[ placement ] - curlNorms[ placement ] .+ 1
end

print(maximum(decompH))


begin
      plot( layout = grid(9, 2, widths = [0.66, 0.34]), right_margin = -5Plots.mm)

      #indices = [ 1, 2, 3, 4, 5, 6, 7, 8]
      #indices = 1:45
      #indices = [ 1 7 12 19 20 21 22 30 44]
      indices = [ 1 7 12 15 16 20 25 35 60]
      #indices = [ 1 7 12 13 16 35 36 37 38]
      
      for j in eachindex(indices)
            i=indices[j]
            
            #if j != 9
                  groupedbar!(
                        [ decompD[:, i] decompU[:, i] decompH[:, i] ] , bar_position = :stack, color = [cols[1] cols[2] cols[4]],
                        #labels = "",
                        labels = ["gradient" "curl" "harmonic"],
                        legend = :topright,
                        sp=2*(j-1)+1
                  )
            #=else
                  groupedbar!(
                        [ decompD[:, i] decompU[:, i] decompH[:, i] ] , bar_position = :stack, color = [cols[1] cols[2] cols[4]],
                        labels = ["gradient" "curl" "harmonic"],
                        legend = :outerbottom, legend_columns=3,
                        sp=2*(j-1)+1
                  )=#
            #end
            title!( L"%$(i)\mathrm{th \; edge \; appears}",
                  sp=2*(j-1)+1
            )
            xlims!( 0.5, size(distances, 1), sp=2*(j-1)+1)
            ylims!( 0, 1, sp=2*(j-1)+1)

            scatter!( points[1, 1:N], points[2, 1:N],
                  color = cols[1], marker = 6, labels = "", sp = 2*j )
            scatter!( points[1, N+1:2*N], points[2, N+1:2*N],
                  color = cols[2], marker = 6, labels = "", sp = 2*j )
            scatter!( points[1, 2*N+1:3*N], points[2, 2*N+1:3*N],
                  color = cols[6], marker = 6, labels = "", sp = 2*j )

            for ii in axes(edges2, 1)
                  if distances[ii] < filters[i]
                        plot!(
                              [ points[ 1, edges2[ii, 1] ], points[ 1, edges2[ii, 2] ] ],
                              [ points[ 2, edges2[ii, 1] ], points[ 2, edges2[ii, 2] ] ],
                               lw = 1.5, color = :gray, labels="", sp = 2*j
                        )
                  end
            end

            for ii in axes( trians, 1 )
                  if checkTrian(filters[i], trians[ii, :], points)
                        if sum( trians[ii, :] .<= N ) >=2
                              plot!(
                                    [ points[ 1, trians[ii, 1] ], points[ 1, trians[ii, 2] ], points[ 1, trians[ii, 3] ], points[ 1, trians[ii, 1] ] ],
                                    [ points[ 2, trians[ii, 1] ], points[ 2, trians[ii, 2] ], points[ 2, trians[ii, 3] ], points[ 2, trians[ii, 1] ] ],
                                    line = 0,
                                    fill = (0, cols[1]),
                                    labels = "",
                                    fillalpha = 0.2,
                                    sp = 2*j
                              )
                        else
                              plot!(
                                    [ points[ 1, trians[ii, 1] ], points[ 1, trians[ii, 2] ], points[ 1, trians[ii, 3] ], points[ 1, trians[ii, 1] ] ],
                                    [ points[ 2, trians[ii, 1] ], points[ 2, trians[ii, 2] ], points[ 2, trians[ii, 3] ], points[ 2, trians[ii, 1] ] ],
                                    line = 0,
                                    fill = (0, cols[2]),
                                    labels = "",
                                    fillalpha = 0.2,
                                    sp = 2*j
                              )
                        end
                  end
            end
             
            
      end

      #=histogram!(
            [ decompD[:, 100] decompU[:, 100] ], bins = 1:size(distances,1), color = [cols[1] cols[2]], labels="",
            sp = 1
      )=#
      
      xticks!(:none)
      yticks!(:none)
      plot!(size = (1050, 2700))

end

#savefig("filter12ring2.pdf")



begin
      
      plot()

      plot!( sum(decompU; dims=2) ./ (size(distances, 1):-1:1), 
      labels="overall curl", color = cols[2], lw = 2 )
      plot!( sum(decompD; dims=2) ./ (size(distances, 1):-1:1), 
      labels="overall gradient", color = cols[1], lw = 2 )
      plot!( sum(decompH; dims=2) ./ (size(distances, 1):-1:1), 
      labels="overall harmonic", color = cols[4], lw = 2 )

      xlabel!("edges")

      plot!( size = (800, 400 ), legend = :topright)
end

#savefig("average_flows_per_edge.pdf")

begin
      plot(layout = grid(1, 3))
      w = abs.( sum(decompH; dims=2) ) ./ (size(distances, 1):-1:1) * 100
      w = w[invperm(perm)]

      for ii in axes(edges2, 1)
            plot!(
                  [ points[ 1, edges2[ii, 1] ], points[ 1, edges2[ii, 2] ] ],
                  [ points[ 2, edges2[ii, 1] ], points[ 2, edges2[ii, 2] ] ],
                  lw = 3.5*w[ii], color = cols[4], labels="", alpha=w[ii], sp = 1
            )
      end
      title!("harmonic part",  sp=1)

      
      w = abs.( sum(decompD; dims=2) ) ./ (size(distances, 1):-1:1) * 5
      w .-= w[end]
      w = w[invperm(perm)]
      for ii in axes(edges2, 1)
            plot!(
                  [ points[ 1, edges2[ii, 1] ], points[ 1, edges2[ii, 2] ] ],
                  [ points[ 2, edges2[ii, 1] ], points[ 2, edges2[ii, 2] ] ],
                  lw = 3.5*w[ii], color = cols[1], labels="", alpha=w[ii], sp = 2
            )
      end
      title!("gradient part",  sp=2)

      w = abs.( sum(decompU; dims=2) ) ./ (size(distances, 1):-1:1) * 4
      w .-= w[1]
      w = w[invperm(perm)]
      for ii in axes(edges2, 1)
            plot!(
                  [ points[ 1, edges2[ii, 1] ], points[ 1, edges2[ii, 2] ] ],
                  [ points[ 2, edges2[ii, 1] ], points[ 2, edges2[ii, 2] ] ],
                  lw = 3.5*w[ii], color = cols[2], labels="", alpha=w[ii], sp=3
            )
      end
      title!("curl part",  sp=3)

      scatter!( points[1, :], points[2, :],
            color = :black, marker = 6, labels = "", sp = 1)
      scatter!( points[1, :], points[2, :],
            color = :black, marker = 6, labels = "", sp = 2)
      scatter!( points[1, :], points[2, :],
            color = :black, marker = 6, labels = "", sp = 3)

      plot!(size=(1800, 600))
end
#savefig("graph_3flows.pdf")

# 2, 8 = 31th
# 2, 3 = 12th
# 3, 9 = 29th
# 8, 9 = 19th

begin
      plot(layout = grid(2,2))

      plot!(decompH[31, :], color=cols[4], labels="harmonic part", lw=2.5, sp=1)
      plot!(decompD[31, :], color=cols[1], labels="gradient part", lw=2.5, sp=1)
      plot!(decompU[31, :], color=cols[2], labels="curl part", lw=2.5, sp=1)
      title!(L"[2,8]", sp=1)

      plot!(decompH[12, :], color=cols[4], labels="", lw=2.5, sp=2)
      plot!(decompD[12, :], color=cols[1], labels="", lw=2.5, sp=2)
      plot!(decompU[12, :], color=cols[2], labels="", lw=2.5, sp=2)
      title!(L"[2,3]", sp=2)

      plot!(decompH[29, :], color=cols[4], labels="harmonic part", lw=2.5, sp=3)
      plot!(decompD[29, :], color=cols[1], labels="gradient part", lw=2.5, sp=3)
      plot!(decompU[29, :], color=cols[2], labels="curl part", lw=2.5, sp=3)
      title!(L"[3,9]", sp=3)

      plot!(decompH[19, :], color=cols[4], labels="harmonic part", lw=2.5, sp=4)
      plot!(decompD[19, :], color=cols[1], labels="gradient part", lw=2.5, sp=4)
      plot!(decompU[19, :], color=cols[2], labels="curl part", lw=2.5, sp=4)
      title!(L"[8,9]", sp=4)

      plot!(size = (1200, 800), legend = :topleft)
end

#savefig("around_donute.pdf")