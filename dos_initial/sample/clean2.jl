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

#----------------------------------------------------------------------#
#------------------------Functions space-------------------------------#
#----------------------------------------------------------------------#


include("pointCloudGenerator.jl")
include("generateDelaunay.jl")
include("DoS_functions.jl")


#----------------------------------------------------------------------#
#------------------------End of Functions space------------------------#
#----------------------------------------------------------------------#



moments_num = 100
Nz = 50
Nbins = 35
N = 50

H, Hd, Hu, σ1, L1, Ld, Lu, σ, m1, α, B1, B2, L0, H0 = generateDelaunayMatrices( N ); # generated complex and matrices
σ1down = eigvals( Matrix( Hd ) )
σ1up = eigvals( Matrix( Hu ) )
#σ10 = eigvals( Matrix( H0 ) )

#@assert domainCheck( H, Hd, Hu) "Domain scaling failed!"


dm_exact = getMomentsExact( σ1; moments_num = moments_num )
dm_exact_down = getMomentsExact( σ1down; moments_num = moments_num )
dm_exact_up = getMomentsExact( σ1up; moments_num = moments_num )

dm_sample, cs = getMomentsSample( H; Nz = Nz, moments_num = moments_num, regime = "sign" )
dm_sample_down, cs = getMomentsSample( Hd; Nz = Nz, moments_num = moments_num, regime = "sign" )
dm_sample_up, cs = getMomentsSample( Hu; Nz = Nz, moments_num = moments_num, regime = "sign" )


tmα = cos.( collect(0:moments_num-1) * acos(α) ) #decomposition shift

@assert norm( dm_sample - ( dm_sample_down + dm_sample_up - tmα ), Inf) < 2.5*1e-2  "sampled decomposition didn't work!"
@assert norm( dm_exact - ( dm_exact_down + dm_exact_up - tmα ), Inf) < 1e-2  "decomposition didn't work! weird, huh?"



#Q = getExactRange( B1' ) 
Q = getSampledRange( B1', N+13 )
P = nullspace( Q' )
PPt = getSampledComplement( Q )

dm_sample_up_fixed, cs = getMomentsSample( P' * Hu * P; Nz = Nz, moments_num = moments_num, regime = "sign"  )
dm_sample_up_fixed_sp, cs = getMomentsSampleSubspace( Hu, PPt ; Nz = Nz, moments_num = moments_num, regime = "sign"  )

#norm(  size( P, 2 ) / m1 * dm_sample_up_fixed - dm_sample_up_fixed_sp ) / moments_num


hu = getHApprox( dm_sample_up, σ1; Nbins = Nbins )
hu_exact = getHExact( σ1up; Nbins = Nbins ) / m1
hu_fixed = getHApprox( dm_sample_up_fixed_sp, σ1; Nbins = Nbins )
#hu_fixed[1] = 1 - norm(hu_fixed, 1) + abs(hu_fixed[1])

[ hu_exact hu hu_fixed ]

norm(hu_exact[2:end] - hu[2:end]), norm(hu_exact[2:end] - hu_fixed[2:end])





#Q = getExactRange( Matrix( B2 )  ) 
Q = getSampledRange( B2, m1 )
P = nullspace( Q' )
PPt = getSampledComplement( Q )

#dm_sample_down_0, cs = getMomentsSample( H0; Nz = Nz, moments_num = moments_num, regime = "sign"  )


dm_sample_down_fixed, cs = getMomentsSample( P' * Hd * P; Nz = Nz, moments_num = moments_num, regime = "sign"  )
dm_sample_down_fixed_sp, cs = getMomentsSampleSubspace( Hd, PPt ; Nz = Nz, moments_num = moments_num, regime = "sign"  )

#norm(  size( P, 2 ) / m1 * dm_sample_up_fixed - dm_sample_up_fixed_sp ) / moments_num


hd = getHApprox( dm_sample_down, σ1; Nbins = Nbins )
hd_exact = getHExact( σ1down; Nbins = Nbins ) / m1
hd_fixed = getHApprox( dm_sample_down_fixed_sp, σ1; Nbins = Nbins )
#hd_fixed[1] = 1 - norm(hd_fixed, 1) + abs(hd_fixed[1])
#hd0 = getHApprox( dm_sample_down_0, σ1; Nbins = Nbins )

#[ hd_exact hd hd_fixed hd0 ]

norm(hd_exact[2:end] - hd[2:end]), norm(hd_exact[2:end] - hd_fixed[2:end])#, norm(hd_exact[2:end] - hd0[2:end])


h = getHApprox( dm_sample, σ1; Nbins = Nbins )
#h_fixed = getHApprox( dm_sample_up_fixed + dm_sample_down_fixed - tmα, σ1; Nbins = Nbins )
#h_fixed[1] = 1 - norm(h_fixed, 1) + abs(h_fixed[1])

h_fixed = hd_fixed + hu_fixed# - getHApprox( tmα, σ1; Nbins = Nbins ) / m1
h_exact = getHExact( σ1; Nbins = Nbins ) / m1

[ h_exact h h_fixed ]
norm(h_exact[2:end] - h[2:end]), norm(h_exact[2:end] - h_fixed[2:end])

gr()
begin
      x = collect( range( -1-1e-6, 1+1e-6 , Nbins+1 ) )
      xm = ( x[1:end-1] + x[2:end] ) / 2


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
      ylims!( (0.0, 1.1*maximum(hd_fixed[2:end])),  sp = 2 )
      ylims!( (0.0, 1.1*maximum(hu_fixed[2:end])),  sp = 3 )
      ylims!( (0.0, 1.1*maximum(h_fixed[2:end])),  sp = 1 )

      plot!(
            size = (1350, 550),
            legend = :outerbottom
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






savefig("test.png")





