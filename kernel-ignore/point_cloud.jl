using LinearAlgebra
using Random, Distributions
using SparseArrays
using Polynomials
using StatsBase
using GR: delaunay

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


"""
    getMomentsExact( σ1; moments_num = 1000 )

TBW
"""
function getMomentsExact( σ1; moments_num = 1000 )
      return [ mean( cos.( m * acos.( σ1 ) ) ) for m in 0:moments_num ]
end


"""
    getMomentsTraces( H; moments_num = 1000 )

TBW
"""
function getMomentsTraces( H; moments_num = 1000 )
      m1 = size( H, 1 )
      dm_traces = zeros( moments_num + 1 )
      dm_traces[ 1 ] = 1
      dm_traces[ 2 ] = tr( H ) / m1
      tm1 = I; tm2 = H 
      for i in 3:moments_num+1
            tm3 = 2 * H * tm2 - tm1 
            dm_traces[ i ] = tr( tm3 ) / m1
            tm1 = tm2
            tm2 = tm3
      end
      return dm_traces
end


"""
    getMomentsSample( H; Nz = 20, moments_num = 1000 )

TBW
"""
function getMomentsSample( H; Nz = 20, moments_num = 1000, regime = "sign" )
      m1 = size(H, 1)
      if regime == "sign"
            Z = sign.( rand( m1, Nz ) .- 0.5 )
      else
            Z = randn( m1, Nz )
      end

      cZ = zeros(moments_num, Nz)

      TZp = Z
      TZk = H * Z
      cZ[1, :] = mean(Z .* TZp; dims=1)
      cZ[2, :] = mean(Z .* TZk; dims=1)
      for k = 3:moments_num
            TZ = 2 * H* TZk - TZp
            TZp = TZk
            TZk = TZ
            cZ[k, :] = mean(Z .* TZk; dims=1)
      end

      c  = mean(cZ; dims=2) 
      cs = std(cZ; dims=2) / sqrt(Nz)
      return c, cs
end


"""
    getHApprox( c, σ1; Nbins = 50)

TBW
"""
function getHApprox( c, σ1; Nbins = 50)
      moments_num = size(c, 1)
      x = collect( range( σ1[1], σ1[end] , Nbins+1 ) )
      xm = ( x[1:end-1] + x[2:end] ) / 2

      h = zero(xm)
      txx = acos.( x )
      yy = c[1]*(txx) / 2
      for np = 2:moments_num
            n = np-1
            yy = yy + c[np] * sin.(n*txx)/n
      end
      
      h = -2/π * yy
      return diff(h) * size(σ1, 1)
end


"""
    getHExact( σ1; Nbins = 50 )

TBW
"""
function getHExact( σ1; Nbins = 50 )
      x = collect( range( σ1[1], σ1[end] , Nbins+1 ) )
      hist = StatsBase.fit(Histogram, σ1, x )
      return hist.weights
end







#----------------------------------------------------------------------------#
#---------------------End of functions space---------------------------------#
#----------------------------------------------------------------------------#



N = 50
H, σ1, L1, σ, m1 =  generateMatrices( N );


moments_num = 5000
Nz = 200
dm_exact = getMomentsExact( σ1; moments_num = moments_num - 1 )
dm_traces = getMomentsTraces( H; moments_num = moments_num - 1 )
dm_sample, cs = getMomentsSample( H; Nz = Nz, moments_num = moments_num, regime = "sign" )


c = dm_sample

h = getHApprox( dm_sample, σ1 )
ht = getHExact( σ1 )



Ms = [10; 20; 50; 75; 100; 200; 500; 750; 1000; 2000; 5000]
errors50 = zeros(size(Ms))
errors_decomp50 = zeros(size(Ms))
ht = getHExact( σ1 )

for i in eachindex(Ms)
      moments_num = Ms[i]
      h = getHApprox( dm_sample[1:moments_num], σ1 )
      h2 = getHApprox( dm_exact[1:moments_num], σ1 )

      errors50[ i ] = norm( h - ht ) / norm( ht )
      errors_decomp50[ i ] = norm( h2 - ht ) / norm( ht )
end


begin
      plot()

      plot!( Ms, errors10, marker=(:circle, 8), lw=4, color=cols[1], labels = L"m_1=167, \mathrm{\; sampled \; moments}")
      plot!( Ms, errors_decomp10, marker=(:square, 8), lw=4, line=:dash, color=cols[1], labels = L"m_1=167, \mathrm{\; exact \; moments}" )

      plot!( Ms, errors20, marker=(:circle, 8), lw=4, color=cols[2], labels = L"m_1=207, \mathrm{\; sampled \; moments}")
      plot!( Ms, errors_decomp20, marker=(:square, 8), lw=4, line=:dash, color=cols[2], labels = L"m_1=207, \mathrm{\; exact \; moments}" )

      plot!( Ms, errors50, marker=(:circle, 8), lw=4, color=cols[6], labels = L"m_1=1136, \mathrm{\; sampled \; moments}")
      plot!( Ms, errors_decomp50, marker=(:square, 8), lw=4, line=:dash, color=cols[6], labels = L"m_1=1136, \mathrm{\; exact \; moments}" )
      
      xlabel!(L"\mathrm{number \; of \; moments, \; M}")
      ylabel!(L"\mathrm{relative \;}L_1-\mathrm{error}")

      plot!( 
            size=(800, 600),
            xscale = :log10, yscale = :log10,
            legend = :topright
      )
      title!( L"\mathrm{DoS, \;} N_z = 200" )
end

savefig("error_plots_200samples_eps15.pdf")





ϵ = 0.3
N = 50
moments_num = 1000
Nz = 100

ϵs = collect( range( 0.3, 2, 75 ) )

profile = zeros( size(ϵs, 1 ), 50 ) 
profile_down = zeros( size(ϵs, 1 ), 50 ) 
profile_up = zeros( size(ϵs, 1 ), 50 ) 


for i in eachindex(ϵs)
      ϵ = ϵs[i]
      H, Hd, Hu, σ1, L1, Ld, Lu, σ, m1 =  generateAllMatrices( N; ϵ = ϵ  );

      dm_sample, cs = getMomentsSample( H; Nz = Nz, moments_num = moments_num, regime = "sign" )
      dm_sample_down, cs = getMomentsSample( Hd; Nz = Nz, moments_num = moments_num, regime = "sign" )
      dm_sample_up, cs = getMomentsSample( Hu; Nz = Nz, moments_num = moments_num, regime = "sign" )

      h = getHApprox( dm_sample, σ1 )
      hd = getHApprox( dm_sample_down, σ1 )
      hu = getHApprox( dm_sample_up, σ1 )

      profile[ i, : ] = h
      profile_down[ i, : ] = hd
      profile_up[ i, : ] = hu
end


profile_down_clean = deepcopy(profile_down)
profile_down_clean[ profile_down_clean .> 2*N] .= 2*N


begin
      plot( layout=(1,3))

      heatmap!(profile, c = :acton,
            sp = 1
      )
      heatmap!(profile_down_clean[:, 2:end], c = :acton,
            sp = 2
      )
      heatmap!(profile_up[:, 2:end], c = :acton,
            sp = 3
      )

      xlabel!(L"\sigma(H)\mathrm{, \; spectrum}")
      ylabel!(L"\varepsilon\mathrm{, \; percolation}", sp = 1)
      yticks!([1, 10, 32, 75], [L"0.3", L"0.5", L"1.0", L"2.0"], sp = 1)
      yticks!([1, 10, 32, 75], ["", "", "", ""], sp = 2)
      yticks!([1, 10, 32, 75], ["", "", "", ""], sp = 3)

      title!(
            L"L_1",
            sp=1
      )
      title!(
            L"L^{\downarrow}_1",
            sp=2
      )
      title!(
            L"L^{\uparrow}_1",
            sp=3
      )

      plot!(
            size=(1200, 400)
      )
end
savefig("percolation_example.pdf")


#=========#
 1

include("generateDelaunay.jl")
N=4
n, points, ν_init, edges2, trians = sparseDelaunay( N=N, ν =0.5  )
n = N + 4
δ = 0.1
B1 = B1fromEdges( n, edges2 )
B2 = B2fromTrig( edges2, trians)

L1 = B1' * B1 + B2 * B2'
Ld = B1' * B1
Lu = B2 * B2'
σ =  eigvals( Matrix( L1 ) )

U, S, V = svd( Matrix(Ld))

P = U[:, S.> 1e-6]
Q = U[:, S.< 1e-6]

qr_decomp = qr( I - P * P' )
r_diag = diag(Matrix(qr_decomp.R))
Q2 = Matrix(qr_decomp.Q)[:, abs.(r_diag) .> 1e-6 ]






























H = 2 * ( 1 - δ) / ( σ[end] - σ[1] + 1e-12 ) * L1 - (1-δ) * ( σ[end] + σ[1] ) / ( σ[end] - σ[1] + 1e-6 ) * I
Hd = 2 * ( 1 - δ) / ( σ[end] - σ[1] + 1e-12 ) * Ld - (1-δ) * ( σ[end] + σ[1] ) / ( σ[end] - σ[1] + 1e-6 ) * I
Hu = 2 * ( 1 - δ) / ( σ[end] - σ[1] + 1e-12 ) * Lu - (1-δ) * ( σ[end] + σ[1] ) / ( σ[end] - σ[1] + 1e-6 ) * I

σ1 =  eigvals( Matrix( H ) )

m1 = size(L1, 1)

moments_num = 1000
Nz = 250
Nbins = 20
#N = 50

H, Hd, Hu, σ1, L1, Ld, Lu, σ, m1 = generateDelaunayMatrices( N );
dm_sample, cs = getMomentsSample( H; Nz = Nz, moments_num = moments_num, regime = "sign" )
dm_sample_down, cs = getMomentsSample( Hd; Nz = Nz, moments_num = moments_num, regime = "sign" )
dm_sample_up, cs = getMomentsSample( Hu; Nz = Nz, moments_num = moments_num, regime = "sign" )

σ1down = eigvals( Matrix( Hd ) )
σ1up = eigvals( Matrix( Hu ) )

h = getHApprox( dm_sample, σ1; Nbins = Nbins )
hd = getHApprox( dm_sample_down, σ1; Nbins = Nbins )
hu = getHApprox( dm_sample_up, σ1; Nbins = Nbins )

x = collect( range( -1, 1 , Nbins+1 ) )
xm = ( x[1:end-1] + x[2:end] ) / 2

begin
      plot( layout = grid( 1,3 ))
      
      histogram!( σ1, bins = x, labels=L"h_i\mathrm{, \;exact \; DoS} " ,
            sp = 1
      )
      scatter!( xm, h, labels=L"\tilde{h}_i\mathrm{, \;approx. \; DoS} " ,
            sp = 1
      )

      histogram!( σ1down, bins = x, labels=L"h_i\mathrm{, \;exact \; DoS} " ,
            sp = 2
      )
      scatter!( xm[hd .> 0], hd[ hd .> 0], labels=L"\tilde{h}_i\mathrm{, \;approx. \; DoS} " ,
            sp = 2
      )

      histogram!( σ1up, bins = x, labels=L"h_i\mathrm{, \;exact \; DoS} " ,
            sp = 3
      )
      scatter!( xm, hu, labels=L"\tilde{h}_i\mathrm{, \;approx. \; DoS} " ,
            sp = 3
      )

      plot!(
            size = (1350, 450),
            legend = :topright
      )
      title!(
            L"L_1",
            sp=1
      )
      title!(
            L"L^{\downarrow}_1",
            sp=2
      )
      title!(
            L"L^{\uparrow}_1",
            sp=3
      )
      xlabel!("spectrum")
      ylabel!("counts")
end

savefig("triangulation.pdf")

















N = 20
points = getPoints(N)
n, edges2, trians = getEdgesTriansFromPoints(N, points)

begin
      
      plot()

      scatter!(
            points[1, 1:N], points[2, 1:N],
            color = cols[1], marker = 6, labels = "",
      )

      scatter!(
            points[1, N+1:2*N], points[2, N+1:2*N],
            color = cols[2], marker = 6, labels = "",
      )

      for i in axes(edges2, 1)
            plot!(
                  [ points[ 1, edges2[i, 1] ], points[ 1, edges2[i, 2] ] ],
                  [ points[ 2, edges2[i, 1] ], points[ 2, edges2[i, 2] ] ],
                   lw = 1, color = :gray, labels=""
            )
      end

      for i in axes( trians, 1 )
            if sum( trians[i, :] .<= N ) >=2
                  plot!(
                        [ points[ 1, trians[i, 1] ], points[ 1, trians[i, 2] ], points[ 1, trians[i, 3] ], points[ 1, trians[i, 1] ] ],
                        [ points[ 2, trians[i, 1] ], points[ 2, trians[i, 2] ], points[ 2, trians[i, 3] ], points[ 2, trians[i, 1] ] ],
                        line = 0,
                        fill = (0, cols[1]),
                        labels = "",
                        fillalpha = 0.2
                  )
            else
                  plot!(
                        [ points[ 1, trians[i, 1] ], points[ 1, trians[i, 2] ], points[ 1, trians[i, 3] ], points[ 1, trians[i, 1] ] ],
                        [ points[ 2, trians[i, 1] ], points[ 2, trians[i, 2] ], points[ 2, trians[i, 3] ], points[ 2, trians[i, 1] ] ],
                        line = 0,
                        fill = (0, cols[2]),
                        labels = "",
                        fillalpha = 0.2
                  )
            end
      end

      plot!(
            size = ( 600, 600 )
      )
end

savefig("complex_example.pdf")






begin
      
      plot()
      
      histogram!( σ1, bins = x, labels=L"h_i\mathrm{, \;exact \; DoS} " )
      #scatter!( xm, yy )
      scatter!( xm, h, labels=L"\tilde{h}_i\mathrm{, \;approx. \; DoS} " )
      plot!(
            size = (650, 450),
            legend = :topright
      )
      xlabel!("spectrum")
      ylabel!("counts")
end

savefig("dos_example.pdf")














