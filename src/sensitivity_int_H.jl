# util functions for neural generated hamiltonian
"""
    returns a dfunc for an ode solver,
    function feeds pulse amplitudes into a neural network and generates a hermitian hamiltonian
    and then applies du .= hamiltonian * u

    the neural network is as defined by NN.jl 
    
    S::System , the system
    pulses:: , a flattened pulse sequence
    nn::NN,  a neural network as defined in NN.jl
    static_op ,
"""

##cell
function make_dfunc_neural(
    S::System,
    pulses,
    nn::NN;# makeNN([Dense(2 * S.N, 2 * S.dim * S.dim + 10, relu), Dense(2 * S.dim * S.dim + 10, 2 * S.dim * S.dim, softmax)]),
    ) 
    S = deepcopy(S)
    nn = deepcopy(nn)
    pulses = deepcopy(pulses)
    repart = S.repart
    impart = S.impart
    opsize = S.dim * S.dim
    output_l = zeros(S.dim * S.dim)
    #output_l = zeros(2*S.dim * S.dim)
    sym = 1:S.dim*(S.dim+1)÷2
    antisym = (S.dim*(S.dim+1)÷2+1):(S.dim*S.dim)
    H_re = zeros(S.dim, S.dim)
    H_im = zeros(S.dim, S.dim)
    amps = zeros(2 * S.N) #2 amps per qubit
    scratch = zeros(2 * S.dim)
    #some 2 qubit specific ops 
    Z = [1.0 0.0
         0.0 -1.0]
    X = [0.0 1.0
         1.0 0.0]
    iY = [0.0 -1.0 #todo check this sign once more
          1.0 0.0]

    Id = I(2)
    ZX = kron(X,Z)
    ZI = kron(I(2),Z)

    Δ12 = ω(S,1) - ω(S,2)
    J12 = J(S,1)

    Ωs1 = Ω(S,1)
    Ωs2 = Ω(S,2)

    δ1 = Δ(S,1)
    function dfunc(du, u, params, t)
        #set the neural net params
        du .= 0
        set_nnparams(nn, params)

        ##assemble input into `amps`
        for i in 1:S.N
            amps[2i-1] = get_It(t, i, pulses, S)
            amps[2i] = get_Qt(t, i, pulses, S)
        end

        #apply neural net on input, get value into output_l
        applyNN!(nn, amps, output_l)
        applyNN!(nn, amps, output_l)

        symmetrize(H_re,view(output_l,sym))
        antisymmetrize(H_im,view(output_l,antisym))

        #output_l .= output_l
        scratch .=0
        ####apply hamiltonian on u, store into du
        mul!(view(scratch,repart),H_re,view(u,repart),1.0,1.0)
        mul!(view(scratch,repart),H_im,view(u,impart),-1.0,1.0)

        mul!(view(scratch,impart),H_re,view(u,impart),1.0,1.0)
        mul!(view(scratch,impart),H_im,view(u,repart),1.0,1.0)

        ##2 qubit specific
        Ωi = Ωs1*amps[1]
        #Ωq = Ωs1#*amps[3]

        ##w_zi = 0.5*(Δ12 - sqrt(Δ12^2 + Ωi^2))  
        ##w_zx = -0.5*J12*Ωi/sqrt(Δ12^2 + Ωi^2) 
        ##w_zx = 0.5*J12*Ωi*(1/(Δ12-Δ(sys1,1)) - 1/Δ12 ) 
        ##w_zi = 0.5*Ωi^2*(1/(Δ12-Δ(sys1,1)) - 1/Δ12 ) 
        ##w_zi = 0.5*w_zx* Ωi/J12
        #
        #w_zx = 0.5*J12*Ωi*(1/(Δ12-δ1) - 1/Δ12 ) 
        #w_zi = 0.5*Ωi^2*(1/(Δ12-δ1) - 1/Δ12 ) 

        #mul!(view(scratch,repart), ZX , view(u, repart), w_zx , 1.0)
        #mul!(view(scratch,impart), ZX , view(u, impart), w_zx , 1.0)

        #mul!(view(scratch,repart), ZI , view(u, repart), w_zi , 1.0)
        #mul!(view(scratch,impart), ZI , view(u, impart), w_zi , 1.0)

        du[repart] .= -1.0.*view(scratch,impart)
        du[impart] .= view(scratch,repart)
        nothing
    end
    return dfunc, nn
end

"""
    makes a problem for neural hamiltonian
"""
function make_neuralH_prob(
    S::System,
    pulses,
    nn::NN=makeNN([Dense(2 * S.N, 2 * S.dim * S.dim + 10, relu), Dense(2 * S.dim * S.dim + 10, 2 * S.dim * S.dim, softmax)]),
)
    tspan::Tuple = pulse_tspan(S, pulses)
    dfunc, nn = make_dfunc_neural(S, pulses, nn)
    u0 = zeros(2 * S.dim)
    u0[1] = 1.0
    nnparams = get_nnparams(nn)
    prob = ODEProblem{true}(dfunc, u0, tspan, nnparams)
    return prob, nn, nnparams
end

"""
    takes an array of n(n+1)/2 elements and makes an n*n symmetric matrix
"""
function symmetrize(SM,arr)
    N = size(SM)[1]
    count=1
    for i in 1:N
        for j in 1:i
            SM[i,j] = arr[count]
            SM[j,i] = SM[i,j]
            count+=1
        end
    end
    nothing
end


"""
    takes an array of n(n-1)/2 elements and makes an n*n antisymmetric matrix
"""
function antisymmetrize(SM,arr)
    N = size(SM)[1]
    SM.=0
    count=1
    for i in 1:N
        for j in 1:i-1
            SM[i,j] = arr[count]
            SM[j,i] = -SM[i,j]
            count+=1
        end
    end
    nothing
end
