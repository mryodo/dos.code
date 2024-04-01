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
    getMomentsSampleSubspace( H; Nz = 20, moments_num = 1000, regime = "sign" )

TBW
"""
function getMomentsSampleSubspace( H, PPt; Nz = 20, moments_num = 1000, regime = "sign" )
      m1 = size(H, 1)
      if regime == "sign"
            Z = sign.( rand( m1, Nz ) .- 0.5 )
      else
            Z = randn( m1, Nz )
      end

      Z = PPt * Z

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
      x = collect( range( -1+1e-6, 1-1e-6, Nbins+1 ) )
      xm = ( x[1:end-1] + x[2:end] ) / 2

      h = zero(xm)
      txx = acos.( x )
      yy = c[1]*(txx) / 2
      for np = 2:moments_num
            n = np-1
            yy = yy + c[np] * sin.(n*txx)/n
      end
      
      h = -2/π * yy
      return diff(h) # * size(σ1, 1)
end


"""
    getHExact( σ1; Nbins = 50 )

TBW
"""
function getHExact( σ; Nbins = 50 )
      x = collect( range( -1, 1 , Nbins+1 ) )
      hist = StatsBase.fit(Histogram, σ, x )
      return hist.weights
end







#----------------------------------------------------------------------------#
#---------------------End of functions space---------------------------------#
#----------------------------------------------------------------------------#






begin
      moments_num = 5000
      Nz = 500
      Nbins = 50
      N = 54

      H, Hd, Hu, σ1, L1, Ld, Lu, σ, m1, α, B1, B2 = generateDelaunayMatrices( N );



      dm_sample, cs = getMomentsSample( H; Nz = Nz, moments_num = moments_num, regime = "sign" )
      dm_sample_down, cs = getMomentsSample( Hd; Nz = Nz, moments_num = moments_num, regime = "sign" )
      dm_sample_up, cs = getMomentsSample( Hu; Nz = Nz, moments_num = moments_num, regime = "sign" )




      Q2 = Matrix( qr(B1').Q )

      Q = getSampledRange( B1'*B1, N+4 )
      P = nullspace( Q2' )
      PPt = getSampledComplement( Q2 )
      dm_sample_up_fixed, cs = getMomentsSampleSubspace( Hu, PPt; Nz = Nz, moments_num = moments_num, regime = "sign" )

      dm_sample_up_fixed2, cs = getMomentsSample( P'*Hu*P; Nz = Nz, moments_num = moments_num, regime = "sign" )




      Q3 = Matrix( qr(B2).Q )
      PPt2 = getSampledComplement( Q3 )
      dm_sample_down_fixed, cs = getMomentsSampleSubspace( Hd, PPt2; Nz = Nz, moments_num = moments_num, regime = "sign" )



      σ1down = eigvals( Matrix( Hd ) )
      σ1up = eigvals( Matrix( Hu ) )

      #h_fixed[1] = 1 - ( norm(h_fixed, 1) - h_fixed[1] )
      hd = getHApprox( dm_sample_down, σ1; Nbins = Nbins )
      hd_fixed = m1/(m1-size(Q3, 2))*getHApprox( dm_sample_down_fixed, σ1; Nbins = Nbins )
      hd_fixed[1] += ( 1 - (N+3)/m1)

      hu = getHApprox( dm_sample_up, σ1; Nbins = Nbins )

      hu_fixed = getHApprox( dm_sample_up_fixed, σ1; Nbins = Nbins ) 
      #hu_fixed2 = getHApprox( dm_sample_up_fixed2, σ1; Nbins = Nbins ) 
      hu_fixed[1] += (N+3) / m1


      h = getHApprox( dm_sample, σ1; Nbins = Nbins )
      tmα = cos.( collect(0:moments_num-1) * acos(α) )
      h_fixed = getHApprox( dm_sample_up_fixed + dm_sample_down_fixed - tmα, σ1; Nbins = Nbins )


      x = collect( range( -1, 1 , Nbins+1 ) )
      xm = ( x[1:end-1] + x[2:end] ) / 2

      h_exact = getHExact( σ1; Nbins = Nbins ) / m1
      hd_exact = getHExact( σ1down; Nbins = Nbins ) / m1
      hu_exact = getHExact( σ1up; Nbins = Nbins ) / m1



      begin
            plot( layout = grid(1, 3))

            histogram!( σ1, bins = x, labels=L"h_i\mathrm{, \;exact \; DoS} " , normalize = :probability, sp = 1
            )
            scatter!( xm, h, labels=L"\tilde{h}_i\mathrm{, \;approx. \; DoS} " , sp = 1
            )
            scatter!( xm, h_fixed, labels=L"\tilde{h}_i\mathrm{, \;approx. \; DoS \; FIXED} " , color= cols[6] , sp = 1
            )

            histogram!( σ1down, bins = x, labels=L"h_i\mathrm{, \;exact \; DoS} " , normalize = :probability, sp = 2
            )
            scatter!( xm, hd, labels=L"\tilde{h}_i\mathrm{, \;approx. \; DoS} " , sp = 2
            )
            #scatter!( xm, hu_fixed, labels=L"\tilde{h}_i\mathrm{, \;approx. \; DoS \; FIXED} " , color= cols[6]
            #)
            scatter!( xm, hd_fixed, labels=L"\tilde{h}_i\mathrm{, \;approx. \; DoS \; FIXED} " , color= cols[6] , sp = 2
            )

            histogram!( σ1up, bins = x, labels=L"h_i\mathrm{, \;exact \; DoS} " , normalize = :probability, sp = 3
            )
            scatter!( xm, hu, labels=L"\tilde{h}_i\mathrm{, \;approx. \; DoS} " , sp = 3
            )
            #scatter!( xm, hu_fixed, labels=L"\tilde{h}_i\mathrm{, \;approx. \; DoS \; FIXED} " , color= cols[6]
            #)
            scatter!( xm, hu_fixed, labels=L"\tilde{h}_i\mathrm{, \;approx. \; DoS \; FIXED} " , color= cols[6] , sp = 3
            )

            plot!(
                  size = (1350, 450),
                  legend = :topright
            )
            title!(
                  L"L^{\uparrow}_1", sp = 3
            )
            title!(
                  L"L^{\downarrow}_1", sp = 2
            )
            title!(
                  L"L_1", sp = 1
            )
            xlabel!("spectrum")
            ylabel!("counts")
      end

      println(
            "L1:  ", norm(h-h_exact, 1)/norm(h_exact, 1), " / ", norm(h_fixed-h_exact, 1)/norm(h_exact, 1)
      )
      println(
            "L1down:  ", norm(hd-hd_exact, 1)/norm(hd_exact, 1), " / ", norm(hd_fixed-hd_exact, 1)/norm(hd_exact, 1)
      )
      println(
            "L1up:  ", norm(hu-hu_exact, 1)/norm(hu_exact, 1), " / ", norm(hu_fixed-hu_exact, 1)/norm(hu_exact, 1)
      )
      println()
end






 tmα = cos.( collect(0:moments_num-1) * acos(α) )

dm_sample - (
      dm_sample_up + dm_sample_down - tmα
)

norm( 2*H^2 - I - ( 2 *Hd^2 - I + 2*Hu^2 - I - 2 *(I) - (2*α^2-1)*I ) )

norm( 4*H^3 - 3*H   - ( 4*Hd^3 - 3*Hd + 4*Hu^3 - 3 *Hu  - 4*( α * I)^3  + 3  * (α *I)  ) )


function getSampledRange( A, l )
      n = size(A, 2)
      Ω = randn( n , l)
      Y = A * Ω
      return Matrix( qr(Y).Q )
end


function getSampledComplement( Q )
      return I - Q*Q'
end




######################################################################
1


|
























begin
      plot( layout = grid( 1,3 ))
      
      histogram!( σ1, bins = x, labels=L"h_i\mathrm{, \;exact \; DoS} " ,
            sp = 1, normalize = :probability,
      )
      scatter!( xm, h, labels=L"\tilde{h}_i\mathrm{, \;approx. \; DoS} " ,
            sp = 1
      )

      histogram!( σ1down, bins = x, labels=L"h_i\mathrm{, \;exact \; DoS} " ,
            sp = 2, normalize = :probability,
      )
      scatter!( xm, hd, labels=L"\tilde{h}_i\mathrm{, \;approx. \; DoS} " ,
            sp = 2
      )

      histogram!( σ1up, bins = x, labels=L"h_i\mathrm{, \;exact \; DoS} " ,
            sp = 3, normalize = :probability,
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










