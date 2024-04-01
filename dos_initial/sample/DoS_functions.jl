"""
    getMomentsExact( σ1; moments_num = 1000 )

TBW
"""
function getMomentsExact( σ1; moments_num = 1000 )
      return [ mean( cos.( m * acos.( σ1 ) ) ) for m in 0:moments_num-1 ]
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
      x = collect( range( -1-1e-6, 1+1e-6 , Nbins+1 ) )
      x[1] = -1
      x[end] = 1
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
      x = collect( range( -1-1e-6, 1+1e-6 , Nbins+1 ) )
      hist = StatsBase.fit(Histogram, σ, x )
      #weights = zeros( size(x, 1) - 1)
      #for i in eachindex(weights)
      #      weights[i] = sum(  )
      #end
      return hist.weights #weights #
end



"""
    getSampledRange( A, l )

TBW
"""
function getSampledRange( A, l )
      n = size(A, 2)
      Ω = randn( n , l)
      Y = A * Ω
      Q, R  = qr(Y)

      return Q[ :, 1:findall( abs.(diag(Matrix(R))) .< 1e-8 )[1] ]
end


"""
    getExactRange( A )

TBW
"""
function getExactRange( A )
      Q, R  = qr( A )
      return Q[ :, 1:findall( abs.(diag(Matrix(R))) .< 1e-8 )[1] ] 
end


"""
    getSampledComplement( Q )

TBW
"""
function getSampledComplement( Q )
      return I - Q*Q'
end



"""
    domainCheck( H, Hd, Hu )

TBW
"""
function domainCheck( H, Hd, Hu )
      s = eigvals( Matrix( H ) )
      sd = eigvals( Matrix( Hd ) )
      su = eigvals( Matrix( Hu ) )
      return ( s[1] > -1 ) && ( s[end] < 1 )  && ( sd[1] > -1 ) && ( sd[end] < 1 ) && ( su[1] > -1 ) && ( su[end] < 1 ) 
end