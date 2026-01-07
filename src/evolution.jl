function evolve_oncomplex(
    S::System,
    pulses::Array,
    init_state::Array;
    reltol = 1e-6,
    abstol = 1e-6,
    dense = true,
    save_everystep = true,
)
    scratch = zero(init_state)
    H_0 = get_H0(S, S.params)
    H_J = get_HJ(S, S.params)
    drives = get_drives(S, S.params)
    function dfunc(dstate, state, params, t)
        coeffs = uwavecoeffs(S, pulses, t)
        dstate .= exp.((-im * t) .* H_0) .* state
        mul!(scratch, H_J, dstate, 1.0, 0.0)
        for (coeff, drive) in zip(coeffs, drives)
            mul!(scratch, drive, dstate, coeff, 1.0)
            nothing
        end
        dstate .= -im .* exp.((im * t) .* H_0) .* scratch
    end
    tspan = pulse_tspan(S, pulses)
    prob = ODEProblem{true}(dfunc, init_state, tspan, S.params)
    sol = solve(
        prob,
        Tsit5(),
        maxiters = 1e7,
        reltol = reltol,
        abstol = abstol,
        dense = dense,
        save_everystep = save_everystep,
    )
    return sol
end

function evolve_oncomplex(
    S::System,
    init_state::Array;
    reltol = 1e-6,
    abstol = 1e-6,
    dense = true,
    save_everystep = true,
)
    @assert length(S.pulses) != 0
    evolve_oncomplex(
        S,
        S.pulses,
        init_state;
        reltol = reltol,
        abstol = abstol,
        dense = dense,
        save_everystep = save_everystep,
    )
end

function gen_dfunc_pulses(S, pulses)
    init_state = zeros(2 * S.dim)
    init_state[1] = 1.0
    scratch = zero(init_state)
    repart = S.repart
    impart = S.impart
    H_0 = get_H0(S, S.params)
    H_J = get_HJ(S, S.params)
    drives = get_drives(S, S.params)
    function dfunc(dstate, state, pulses, t)
        coeffs = uwavecoeffs(S, pulses, t)
        dstate[repart] .=
            cos.(-t .* H_0) .* view(state, repart) .- sin.(-t .* H_0) .* view(state, impart)
        dstate[impart] .=
            cos.(-t .* H_0) .* view(state, impart) .+ sin.(-t .* H_0) .* view(state, repart)
        #dstate_repart =
        #    cos.(-t .* H_0) .* state[repart] .- sin.(-t .* H_0) .* state[impart]
        #dstate_impart =
        #    cos.(-t .* H_0) .* state[impart] .+ sin.(-t .* H_0) .* state[repart]
        mul!(view(scratch, repart), H_J, view(dstate, repart), 1.0, 0.0)
        mul!(view(scratch, impart), H_J, view(dstate, impart), 1.0, 0.0)
        #scratch_repart = H_J * dstate_repart 
        #scratch_impart = H_J * dstate_impart 
        for (coeff, drive) in zip(coeffs, drives)
            mul!(view(scratch, repart), drive, view(dstate, repart), coeff, 1.0)
            mul!(view(scratch, impart), drive, view(dstate, impart), coeff, 1.0)
            #scratch_repart = coeff.*drive*dstate_repart .+ scratch_repart
            #scratch_impart = coeff.*drive*dstate_impart .+ scratch_impart
        end
        dstate[impart] .=
            -1.0 .* cos.(t .* H_0) .* view(scratch, repart) .+
            sin.(t .* H_0) .* view(scratch, impart)
        dstate[repart] .=
            cos.(t .* H_0) .* view(scratch, impart) .+
            sin.(t .* H_0) .* view(scratch, repart)
        #dstate_impart =
        #    -1.0 .* cos.(t .* H_0) .* scratch_repart .+
        #    sin.(t .* H_0) .* scratch_impart
        #dstate_repart =
        #    cos.(t .* H_0) .* scratch_impart .+
        #    sin.(t .* H_0) .* scratch_repart
        #dstate .= [dstate_repart;dstate_impart]
        nothing
    end
    return dfunc
end

function evolve_real_pulses(
    S::System,
    pulses;
    reltol = 1e-6,
    abstol = 1e-6,
    sensealg = ZygoteVJP(),
)
    init_state = zeros(2 * S.dim)
    init_state[1] = 1.0
    scratch = zero(init_state)
    repart = S.repart
    impart = S.impart
    H_0 = get_H0(S, S.params)
    H_J = get_HJ(S, S.params)
    drives = get_drives(S, S.params)
    dfunc = gen_dfunc_pulses(S, pulses)
    tspan = pulse_tspan(S, pulses)
    prob = ODEProblem(dfunc, init_state, tspan, S.pulses)
    sol = solve(prob, Tsit5(), reltol = reltol, abstol = abstol, sensealg = sensealg)#InterpolatingAdjoint(autojacvec=EnzymeVJP()))
    return sol
end
