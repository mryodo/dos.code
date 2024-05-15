
"""
    B1fromEdges(n, edges)

TBW
"""
function B1fromEdges(n, edges)
      m = size(edges, 1);
      B1 = spzeros(n, m);
      
      for i in 1:m
          B1[edges[i, 1], i] = -1;
          B1[edges[i, 2], i] = 1;
      end
      return B1
end
  
"""
    B2fromTrig(edges, trigs)

TBW
"""
function B2fromTrig(edges, trigs)
      
      m = size(edges, 1);
      if length(trigs)==1
          return spzeros(m, 1)
      end
      del = size(trigs, 1);
      B2 = spzeros(m, del);
      
      for i in 1:del
          B2[findfirst( all( [trigs[i, 1], trigs[i, 2]]' .== edges, dims=2 )[:, 1] ), i]=1;
          B2[findfirst( all( [trigs[i, 1], trigs[i, 3]]' .== edges, dims=2 )[:, 1] ), i]=-1;
          B2[findfirst( all( [trigs[i, 2], trigs[i, 3]]' .== edges, dims=2 )[:, 1] ), i]=1;
      end 
      return B2
end



"""
    getPoints( N; ϵ = 1.5, dist = 3, center1 = [0; 0], σ0 = 1.0, σ2 = 1.0 )

TBW
"""
function getPoints( N; ϵ = 1.5, dist = 3, center1 = [0; 0], σ0 = 1.0, σ2 = 1.0 )
      center2 = [dist; dist]
      return [ rand( MvNormal(center1, σ0*I), N ) rand( MvNormal(center2, σ2*I), N ) ] 
end

function getPoints3( N; ϵ = 1.5, dist = 3, center1 = [0; 0], σ0 = 1.0, σ2 = 1.0, σ3 = 1.0 )
      center2 = [dist; dist]
      center3 = [0; 2*dist ]
      return [ rand( MvNormal(center1, σ0*I), N ) rand( MvNormal(center2, σ2*I), N ) rand( MvNormal(center3, σ3*I), N )  ] 
end



"""
    getEdgesTriansFromPoints( N, points )

TBW
"""
function getEdgesTriansFromPoints( m0, points; ϵ = 1.5 )
      
      edges2 = Array{Int64}(undef, 0, 2)

      for i in 1:m0-1
            for j in i+1:m0
                  if norm( points[:, i] - points[:, j] ) <= ϵ
                        edges2 = [edges2; i j]
                  end
            end
      end

      trians = Array{Int64}(undef, 0, 3)

      for i in 1:size(edges2, 1)-2
            for j in i+1:size(edges2, 1)-1
                  if edges2[i, 1] == edges2[j, 1]
                        if sum( all( edges2 .== sort([edges2[i, 2]; edges2[j, 2]])', dims = 2) ) == 1
                              trians = [ trians; sort( [ edges2[i, 1]; edges2[i, 2]; edges2[j, 2] ] )' ]
                        end
                  end
            end
      end

      return m0, edges2, trians
end


"""
    generateMatrices( N;  ϵ = 1.5, dist = 3, center1 = [0; 0], σ0 = 1.0, σ2 = 1.0  )

TBW
"""
function generateMatrices( N;  ϵ = 1.5, dist = 3, center1 = [0; 0], σ0 = 1.0, σ2 = 1.0  )
      points = getPoints( N;  ϵ = ϵ, dist = dist, center1 = center1, σ0 = σ0, σ2 = σ2  )
      n, edges2, trians = getEdgesTriansFromPoints( N, points; ϵ = ϵ )
      B1 = B1fromEdges( n, edges2 )
      B2 = B2fromTrig( edges2, trians)
      
      L1 = B1' * B1 + B2 * B2'
      σ =  eigvals( Matrix( L1 ) )

      H = (  L1  - ( σ[1]+σ[end] )/2 * I ) / ( ( σ[end] - σ[1] + 1e-12) / 2 )
      σ1 =  eigvals( Matrix( H ) )

      m1 = size(L1, 1)
      
      return H, σ1, L1, σ, m1
end



function generateAllMatrices( N;  ϵ = 1.5, dist = 3, center1 = [0; 0], σ0 = 1.0, σ2 = 1.0, δ = 0.1  )
      points = getPoints( N;  ϵ = ϵ, dist = dist, center1 = center1, σ0 = σ0, σ2 = σ2  )
      n, edges2, trians = getEdgesTriansFromPoints( N, points; ϵ = ϵ )
      B1 = B1fromEdges( n, edges2 )
      B2 = B2fromTrig( edges2, trians)
      

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

      #H = 2 * ( 1 - δ) / ( σ[end] - σ[1] + 1e-12 ) * L1 - (1-δ) * ( σ[end] + σ[1] ) / ( σ[end] - σ[1] + 1e-6 ) * I
      #Hd = 2 * ( 1 - δ) / ( σ[end] - σ[1] + 1e-12 ) * Ld - (1-δ) * ( σ[end] + σ[1] ) / ( σ[end] - σ[1] + 1e-6 ) * I
      #Hu = 2 * ( 1 - δ) / ( σ[end] - σ[1] + 1e-12 ) * Lu - (1-δ) * ( σ[end] + σ[1] ) / ( σ[end] - σ[1] + 1e-6 ) * I

      σ1 =  eigvals( Matrix( H ) )

      m1 = size(L1, 1)
      return H, Hd, Hu, σ1, L1, Ld, Lu, σ, m1,  - 1 + 1e-6 , B1, B2, L0, H0
end




function generateDelauney(N=4)
      points=rand(N, 2)*0.8.+0.1;
      points=[points; [0 0]; [0 1]; [1 0]; [1 1]];
  
      num, tri=delaunay(points[:, 1], points[:, 2]);
      preedges=Array{Integer}(undef, 0, 2);
      trian=Array{Integer}(undef, num, 3);
      
      for i in axes(tri, 1)
          tmp=tri[i, :]; tmp=sort(tmp)'; trian[i, :]=tmp;
          preedges=[preedges; [tmp[1] tmp[2]];     [tmp[1] tmp[3]]; [tmp[2] tmp[3]]];   
      end
  
      preedges=unique(preedges, dims=1);
      preedges=sort(preedges, dims = 2); 
      preedges=sortslices(preedges, dims = 1);
  
      trian=sort(trian, dims = 2); 
      trian=sortslices(trian, dims = 1);  
      return points, preedges, trian
end


function generateDelaunayMatrices(N = 4; δ = 0.1, ν = 0.7 )
      #points, edges2, trians = generateDelauney( N )
      
      #n = N + 4

      #for i in 1:Int(round(size(trians, 1)/ 3))
      #      indx = getIndx2Kill( edges2 )
      #      edges2, trians = killEdge(indx, n, edges2, trians)
      #end
      #Δ = size(trians, 1) 
      #for i in 1:Int( round( Δ * 39 / 40 ) )
      #      ind = rand(1:size(trians, 1))
      #      trians = trians[1:end .!= ind, :]
      #end
      
      n, points, ν_init, edges2, trians = sparseDelaunay( N = N, ν = ν )

      B1 = B1fromEdges( n, edges2 )
      B2 = B2fromTrig( edges2, trians)

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

      #H = 2 * ( 1 - δ) / ( σ[end] - σ[1] + 1e-12 ) * L1 - (1-δ) * ( σ[end] + σ[1] ) / ( σ[end] - σ[1] + 1e-6 ) * I
      #Hd = 2 * ( 1 - δ) / ( σ[end] - σ[1] + 1e-12 ) * Ld - (1-δ) * ( σ[end] + σ[1] ) / ( σ[end] - σ[1] + 1e-6 ) * I
      #Hu = 2 * ( 1 - δ) / ( σ[end] - σ[1] + 1e-12 ) * Lu - (1-δ) * ( σ[end] + σ[1] ) / ( σ[end] - σ[1] + 1e-6 ) * I

      σ1 =  eigvals( Matrix( H ) )

      m1 = size(L1, 1)
      return H, Hd, Hu, σ1, L1, Ld, Lu, σ, m1,  - 1 + 1e-6 , B1, B2, L0, H0, edges2, trians
end



function getFractalTriangle( N )

      trians = [ 1 2 3; ]
      edges2 = [ 1 2; 1 3; 2 3]

      for cur in 4:N
            tri = trians[1, :]
            edges2 = [edges2; tri[1] cur; tri[2] cur; tri[3] cur;]
            trians = [ trians; tri[1] tri[2] cur; tri[1] tri[3] cur; tri[2] tri[3] cur ] 
            trians = trians[2:end, :]
      end

      preedges=sort(edges2, dims = 2); 
      preedges=sortslices(preedges, dims = 1);

      trian=sort(trians, dims = 2); 
      trian=sortslices(trians, dims = 1);  

      return preedges, trian
end



function generateFractalMatrices(N = 4; δ = 0.1)
      edges2, trians = getFractalTriangle( N )
      n = N + 4
      B1 = B1fromEdges( n, edges2 )
      B2 = B2fromTrig( edges2, trians)

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

      #H = 2 * ( 1 - δ) / ( σ[end] - σ[1] + 1e-12 ) * L1 - (1-δ) * ( σ[end] + σ[1] ) / ( σ[end] - σ[1] + 1e-6 ) * I
      #Hd = 2 * ( 1 - δ) / ( σ[end] - σ[1] + 1e-12 ) * Ld - (1-δ) * ( σ[end] + σ[1] ) / ( σ[end] - σ[1] + 1e-6 ) * I
      #Hu = 2 * ( 1 - δ) / ( σ[end] - σ[1] + 1e-12 ) * Lu - (1-δ) * ( σ[end] + σ[1] ) / ( σ[end] - σ[1] + 1e-6 ) * I

      σ1 =  eigvals( Matrix( H ) )

      m1 = size(L1, 1)
      return H, Hd, Hu, σ1, L1, Ld, Lu, σ, m1,  - 1 + 1e-6 , B1, B2, L0, H0
end