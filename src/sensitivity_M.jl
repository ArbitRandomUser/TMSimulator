"""
    setter function for MDparams
    MD params is an flat array that contains all the elements of M and D matrices
    mutates MDparams, sets the M section with matrix M
"""
function setM!(S, MDparams, M::Matrix)
    bound = S.dim^2 #boundary in MDparams demarcating matrices
    MDparams[1:bound] .= reshape(M, (bound, 1))
    nothing
end

"""
    setter function for MDparams sets D,
    mutates MDparams
"""
function setD!(S, i, MDparams, D)
    bound = S.dim^2
    MDparams[bound+(i-1)*bound:bound+(i)*bound] .= reshape(D, (bound, 1))
    nothing
end

"""
    mutates MDparams
    setter function for MDparams, sets all D's
"""
function setDs!(S, MDparams, Ds::Array{Matrix})
    for i = 1:S.N
        setD!(S, i, MDparams, Ds[i])
    end
    nothing
end

"""
    makes dfunc for evolving (dψ = -iM + Σ_i uwave_i(t) * D_i ψ)
    uwave_i is the microwave amplitude , D_i is the drive operator
"""
function make_dfunc_M_2(S, pulses)
    S = deepcopy(S)
    M = zeros(S.dim, S.dim)
    Ds = [zeros(S.dim, S.dim) for _ = 1:S.N]
    repart = S.repart
    impart = S.impart
    H_0 = get_H0(S, S.params) 
    H_J = get_HJ(S, S.params) 
    scratch = zeros(2 * S.dim)
    drives = get_drives(S, S.params)
    function dfunc(dstate, state, MDparams, t) #MD params is the parameters of M and D matrices as params
        scratch .= 0
        bound = S.dim^2
        M .= reshape(view(MDparams, 1:bound), size(M))
        M .+= transpose(reshape(view(MDparams, 1:bound), size(M))) ##HERMITIAN
        M .= M./2
        for i = 1:length(Ds)
            Ds[i] .=
                reshape(view(MDparams, bound+1+(i-1)*bound:bound+(i)*bound), size(Ds[i]))
            Ds[i] .+=
                transpose(reshape(view(MDparams, bound+1+(i-1)*bound:bound+(i)*bound), size(Ds[i]))) ##HERMITIAN
            Ds[i] .= Ds[i]./2
        end
        H_0 .= view(MDparams,1:S.dim)
        coeffs = uwavecoeffs(S, pulses, t)
        dstate[repart] .=
            cos.(-t .* H_0) .* view(state, repart) .- sin.(-t .* H_0) .* view(state, impart)
        dstate[impart] .=
            cos.(-t .* H_0) .* view(state, impart) .+ sin.(-t .* H_0) .* view(state, repart)

        mul!(view(scratch, repart), M, view(dstate, repart), 1.0, 1.0)
        mul!(view(scratch, impart), M, view(dstate, impart), 1.0, 1.0)

        mul!(view(scratch, repart), H_J, view(dstate, repart), 1.0, 1.0)
        mul!(view(scratch, impart), H_J, view(dstate, impart), 1.0, 1.0)

        for (i, coeff, drive) in zip(1:S.N, coeffs, Ds)
            mul!(view(scratch, repart), drive, view(dstate, repart), coeff, 1.0)
            mul!(view(scratch, impart), drive, view(dstate, impart), coeff, 1.0)
        end

        for (i, coeff, drive) in zip(1:S.N, coeffs, drives)
            mul!(view(scratch, repart), drive, view(dstate, repart), coeff, 1.0)
            mul!(view(scratch, impart), drive, view(dstate, impart), coeff, 1.0)
        end
        dstate[impart] .=
            -1.0 .* cos.(t .* H_0) .* view(scratch, repart) .+
            sin.(t .* H_0) .* view(scratch, impart)
        dstate[repart] .=
            cos.(t .* H_0) .* view(scratch, impart) .+
            sin.(t .* H_0) .* view(scratch, repart)
        nothing
    end
end

"""
    makes a dfunc for ode solver,
    the hamiltonian is the ibm hamiltonian evolved in the interaction picture.
    there is a correction term M and D to the Hamiltonian whose elements are passed in as 
    MDparams.
    train_M , toggles training static correction (default is off)
    train_Ds , a tuple of integers specifying which dynamic corrections are to be trained (default is just the second qubit)

    M is hermitianized, and its diagnols are set to 0.
    D is hermitianized.
"""
function make_dfunc_M(S,pulses,train_M=false,train_Ds=(2,))
    S = deepcopy(S)
    M = zeros(S.dim, S.dim)
    Ds = [zeros(S.dim, S.dim) for _ = 1:S.N]
    repart = S.repart
    impart = S.impart
    H_0 = get_H0(S, S.params) 
    H_J = get_HJ(S, S.params) 
    scratch = zeros(2 * S.dim)
    drives = get_drives(S, S.params)
    @assert max(train_Ds...)<S.N
    function dfunc(dstate, state, MDparams, t) #MD params is the parameters of M and D matrices as params
        scratch .= 0
        bound = S.dim^2
        if !train_M #disable M training
            M .= 0 
        end
        M .= reshape(view(MDparams, 1:bound), size(M))
        M .+= transpose(reshape(view(MDparams, 1:bound), size(M))) ##HERMITIAN
        M .= M./2
        for i in 1:S.dim #disables diagonals .
            M[i,i] = 0.0
        end
        for i in train_Ds 
            Ds[i] .=
                reshape(view(MDparams, bound+1+(i-1)*bound:bound+(i)*bound), size(Ds[i]))
            Ds[i] .+=
                transpose(reshape(view(MDparams, bound+1+(i-1)*bound:bound+(i)*bound), size(Ds[i]))) ##HERMITIAN
            Ds[i] .= Ds[i]./2
        end
        coeffs = uwavecoeffs(S, pulses, t)
        dstate[repart] .=
            cos.(-t .* H_0) .* view(state, repart) .- sin.(-t .* H_0) .* view(state, impart)
        dstate[impart] .=
            cos.(-t .* H_0) .* view(state, impart) .+ sin.(-t .* H_0) .* view(state, repart)

        #mul!(view(scratch, repart), M, view(dstate, repart), 1.0, 1.0)
        #mul!(view(scratch, impart), M, view(dstate, impart), 1.0, 1.0)

        mul!(view(scratch, repart), H_J, view(dstate, repart), 1.0, 1.0)
        mul!(view(scratch, impart), H_J, view(dstate, impart), 1.0, 1.0)

        for (i, coeff, drive) in zip(1:S.N, coeffs, Ds)
            mul!(view(scratch, repart), drive, view(dstate, repart), coeff, 1.0)
            mul!(view(scratch, impart), drive, view(dstate, impart), coeff, 1.0)
        end

        for (i, coeff, drive) in zip(1:S.N, coeffs, drives)
            mul!(view(scratch, repart), drive, view(dstate, repart), coeff, 1.0)
            mul!(view(scratch, impart), drive, view(dstate, impart), coeff, 1.0)
        end
        dstate[impart] .=
            -1.0 .* cos.(t .* H_0) .* view(scratch, repart) .+
            sin.(t .* H_0) .* view(scratch, impart)
        dstate[repart] .=
            cos.(t .* H_0) .* view(scratch, impart) .+
            sin.(t .* H_0) .* view(scratch, repart)
        nothing
    end
end

"""
    make_MDprob , makes an ODEproblem with correction matrices , the matrices are defined by a flat array of 1+no_of_qubits matrices, one static correction and one dynamic correction for each qubit.
"""
function make_MDprob(
    S::System,
    pulses::Array;
    MDparams::Array = zeros((1+ S.N)*S.dim^2),
    u0 =  begin state = zeros(2*S.dim); state[1] = 1.0; state end
)
    tspan::Tuple = pulse_tspan(S, pulses)
    dfunc = make_dfunc_M(S, pulses)
    prob = ODEProblem{true}(dfunc, u0, tspan, MDparams)
    return prob, MDparams
end
