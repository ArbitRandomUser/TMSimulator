module TMSimulator
include("NN.jl")
include("plotting.jl")
using .SimpleNN

using DifferentialEquations
using Enzyme
using LinearAlgebra
using JSON
Jmap = Dict{Tuple{Int,Int},Int} #type for mapping couplings to coupling_ops


"""
    creation operator for an n level system
"""
function cr_op(n) #creation operator
    ret = zeros(ComplexF64, n, n)
    for i = 2:n
        ret[i, i-1] = sqrt(i - 1)
    end
    return ret
end

"""
    annhilation operator for an n level system
"""
function an_op(n) #annhilation operator
    ret = zeros(ComplexF64, n, n)
    for i = 2:n
        ret[i-1, i] = sqrt(i - 1)
    end
    return ret
end

"""
m -> local operator on qubit
i -> specifies which qubit it m acts on
n -> global system no of qubits
lvls -> cutoff
"""
function glb_op(m, i, n, lvls)
    #kron([j == i ? m : Matrix{ComplexF64}(I, lvls, lvls) for j = 1:n]...)
    kron(reverse([j == i ? m : Matrix{ComplexF64}(I, lvls, lvls) for j = 1:n])...)
end

"""
    N:number of qubits
    cr_ops::creation operators
    an_ops::anhilation operators
    number_ops::Number operators
    drive_ops::drive operators
    dt:: system dt
    dim:: system dimension (lvls^N)
    twoindices:: indices of basis states (000,001,010... etc) in the state for this system)
    repart:: real part
"""
mutable struct System{TF1,TF2}
    N::Int ##number of qubits
    cr_ops::Vector{Matrix{TF1}} #creation operators
    an_ops::Vector{Matrix{TF1}} #anhilation operatos
    number_ops::Vector{Matrix{TF1}} #number operators
    drive_ops::Vector{Matrix{TF1}} #
    coupling_ops::Vector{Matrix{TF1}} #coupling operators
    coupling_map::Vector{Tuple{Int,Int}}
    dt::TF2
    dim::Int
    lvls::Int
    twoindices::Vector{Int64}
    repart::UnitRange{Int64}
    impart::UnitRange{Int64}
    params::Array{TF1,1}
    pulses::Array{TF1,1}
    prob::Any
end

function System(
    N,
    couplings::Array{Tuple{Int,Int}};
    lvls,
    dt,
    TF = Float64,
)
    cr_ops = []
    an_ops = []
    number_ops = []
    drive_ops = []
    coupling_ops = []
    dim = lvls^N
    repart = 1:dim
    impart = (dim+1):(2*dim)
    adag = cr_op(lvls)
    a = an_op(lvls)
    glb_oph(m, i) = glb_op(m, i, N, lvls)
    for i = 1:N
        push!(cr_ops, glb_oph(adag, i))
        push!(an_ops, glb_oph(a, i))
        push!(number_ops, glb_oph(adag * a, i))
        push!(drive_ops, glb_oph(adag + a, i))
    end
    for c in couplings
        push!(
            coupling_ops,
            #(glb_oph(adag, c[1]) + glb_oph(a, c[1])) *
            #(glb_oph(a, c[2]) + glb_oph(adag, c[2])),
            glb_oph(adag,c[1])*glb_oph(a,c[2])+glb_oph(adag,c[2])*glb_oph(a,c[1])
        )
    end
    params = zeros(3 * N + length(couplings))
    pulses = []
    twoindices = [parse(Int,s,base=lvls)+1 for s in (string(k,base=2,pad=N) for k in 0:(2^N-1))]
    return System{TF,TF}(
        N,
        cr_ops,
        an_ops,
        number_ops,
        drive_ops,
        coupling_ops,
        couplings,
        dt,
        dim,
        lvls,
        twoindices,
        repart,
        impart,
        params,
        pulses,
        nothing,
    )
end

probdist(u,s::System) = abs.(cstate(u,s)).^2

"""
    gets a complex vector from realvector
"""
function cstate(state::Vector{Float64}, S)
    #@assert mod(length(state),2) == 0
    return state[S.repart] + im * state[S.impart]
end

function cstate(state::Vector{ComplexF64}, S)
    return state
end

## param/pulse functions

"""
parameter array format
    ω     values : N #qubit freqs
    Δ     values : N #qubit anharm
    Ω     values : N #drive amps
    J_ij  values : ~ #couplings number depends on system
"""

"""
pulse array format
    format [ p1_1_I p1_1_Q  p1_1_wd p1_2_I p1_2_Q ..... p1_n_I p1_n_q ... p2_1_I ... ]
    where p_i_j_I is inphase amplitude on i'th qubit at j'th dt
"""

"""
    validates if p has the right size
"""
function validate_pulse(S, pulse)
    @assert length(pulse) % (3 * S.N) == 0
end

function validate_pulse(S)
    @assert length(S.pulse) % (3 * S.N) == 0
end

"""
    assert atleast w's deltas and drivesamps are specified
"""
function validate_params(S)
    @assert length(S.params) >= (3 * S.N)
end

function validate_params(S, params)
    @assert length(params) >= (3 * S.N)
end


"""
    get i'th dt inphase amplitude of qubit q
"""
function get_I(i, q, pulse, S)
    n = pulse_count(S, pulse)
    return pulse[3*(q-1)n+3(i)-2]
end

"""
    get the inphase amp of qubit q at time t
"""
function get_It(t,q,pulse,S)
    n = pulse_count(S, pulse)
    ind = min(floor(Int, t / S.dt) + 1, n)
    return pulse[3*(q-1)n+3(ind)-2]
end
"""
    get i'th dt quadrature (out of phase quadrature) amplitude of qubit q
"""
function get_Q(i, q, pulse, S)
    n = pulse_count(S, pulse)
    return pulse[3*(q-1)n+3i-1]
end

"""
    get the offset quadrature amp of qubit q at time t
"""
function get_Qt(t,q,pulse,S)
    n = pulse_count(S,pulse)
    ind = min(floor(Int,t/S.dt)+1,n)
    return pulse[3*(q-1)n+3ind-1]
end

function get_ωd(i, q, pulse, S)
    n = pulse_count(S, pulse)
    return pulse[3*(q-1)*n+3(i)]
end

function pulse_tspan(S, pulse)
    return (0.0, pulse_count(S, pulse) * S.dt)
end

function pulse_tspan(S)
    return pulse_tspan(S, S.pulses)
end

"""
    get the tspan from pulse array
"""
function pulse_count(S, pulse)
    return length(pulse) ÷ (3 * S.N)
end


## util functions to get parameters
## TODO: using single alphabets (especially greek) for function names was probably a bad idea 
## refactor this!! 
"""
    get the qubit freq
"""
function ω(S, params, q)
    return params[q]
end

function ω(S::System, q)
    return ω(S, S.params, q)
end

"""
  get qubit anharm
"""
function Δ(S, params, q)
    return params[S.N+q]
end

function Δ(S::System, q)
    return Δ(S, S.params, q)
end

"""
  get qubit drive amp.
"""
function Ω(S, params, q)
    return params[2*S.N+q]
end

function Ω(S::System, q)
    return Ω(S, S.params, q)
end

"""
  get qubit couplings
"""
function J(S, params, q)
    return params[3*S.N+q]
end
function J(S::System, q)
    return J(S, S.params, q)
end


## util functions
"""
    get the static hamiltonian diagonal
    constructs ω N + Δ/2.0 (N²-N)

    typically used as H_0 in interaction evolution
"""
function get_H0(S, p)
    mat = zeros(S.dim, S.dim)
    for i = 1:(S.N)
        mat +=
            (ω(S, p, i)) .* S.number_ops[i] .+
            0.5 * (Δ(S, p, i)) .* (S.number_ops[i]^2 .- S.number_ops[i])
    end
    #print(mat)
    return diag(mat)
end

function set_H0(mat, S, p)
    for i = 1:(S.N)
        mat .+=
            (ω(S, p, i)) .* S.number_ops[i] .+
            0.5 * (Δ(S, p, i)) .* (S.number_ops[i]^2 .- S.number_ops[i])
    end
end

"""
    get static hamiltonian other elements
"""
function get_HJ(S, params)
    mat = zeros(S.dim, S.dim)
    for i = 1:length(S.coupling_ops)
        mat += J(S, params, i) * S.coupling_ops[i]
    end
    return mat
end

function set_HJ(mat, S, params)
    for i = 1:length(S.coupling_ops)
        mat .+= J(S, params, i) * S.coupling_ops[i]
    end
end

function get_drives(S, params)
    drives = Array{Matrix{Float64}}([])
    for i = 1:(S.N)
        push!(drives, Ω(S, params, i) * S.drive_ops[i])
    end
    return drives
end

function uwave_envelop(S, pulses, t)
    n = pulse_count(S, pulses)
    ind = min(floor(Int, t / S.dt) + 1, n)
    return (
        (get_I(ind, q, pulses, S), get_Q(ind, q, pulses, S), get_ωd(ind, q, pulses, S)) for
        q = 1:(S.N)
    )
end

function uwavecoeffs(S, pulse, t)
    n = pulse_count(S, pulse)
    ind = min(floor(Int, t / S.dt) + 1, n)
    return (
        get_I(ind, q, pulse, S) * cos(get_ωd(ind, q, pulse, S) * t) - ## TODO  figure this out +/-
        get_Q(ind, q, pulse, S) * sin(get_ωd(ind, q, pulse, S) * t) for q = 1:(S.N)
    )
end

"""
    makes a system from dictionary, typically load the IBMQ hamiltonian json as a disctionary
    and pass to make_system , for large system you might want to trim away parameters
    specify qubits as needed , for example for qubits 0,2,3 only as a subsystem pass qubits = [0,2,3]
"""
function make_system(d::Dict, qubits; lvls = 3, dt = 0.5)
    #regex
    omegas = get_omegas(d)
    Omegas = get_Omegas(d)
    deltas = get_deltas(d)
    Js = get_js(d)
    if qubits == :all
        N = length(omegas)
        qubits = 1:N
    else
        qubits = qubits
    end
    connections = [i[1] for i in Js if (i[1][1] in qubits) && (i[1][2] in qubits)]
    nconnections = [
        (findfirst(x -> x == i[1], qubits), findfirst(x -> x == i[2], qubits)) for
        i in connections
    ]
    jvals = [i[2] for i in Js if (i[1][1] in qubits) && (i[1][2] in qubits)]
    S = System(length(qubits), nconnections, lvls = lvls, dt = dt)
    select(arr, inds) = [arr[i] for i in inds]
    params =
        [
            select(omegas, qubits .+ 1)
            select(deltas, qubits .+ 1)
            select(Omegas, qubits .+ 1)
            jvals
        ] ./ (1e9)
    validate_params(S, params)
    S.params = params
    return S
end

function get_deltas(d)
    rdelta = r"delta(?<q>[0-9]+)"
    regresults = [match(rdelta, key) for key in keys(d) if match(rdelta, key) !== nothing]
    sortf(x, y) = begin
        parse(Int, x.captures[1]) < parse(Int, y.captures[1])
    end
    sorted_regresults = sort(regresults; lt = sortf)
    sorted_keys = [reg.match for reg in sorted_regresults]
    [d[key] for key in sorted_keys]
end

function get_omegas(d)
    romega = r"wq(?<q>[0-9]+)"
    regresults = [match(romega, key) for key in keys(d) if match(romega, key) !== nothing]
    sortf(x, y) = begin
        parse(Int, x["q"]) < parse(Int, y["q"])
    end
    sorted_regresults = sort(regresults; lt = sortf)
    sorted_keys = [reg.match for reg in sorted_regresults]
    [d[key] for key in sorted_keys]
end

function get_Omegas(d)
    rOmega = r"omegad(?<q>[0-9]+)"
    regresults = [match(rOmega, key) for key in keys(d) if match(rOmega, key) !== nothing]
    sortf(x, y) = begin
        parse(Int, x["q"]) < parse(Int, y["q"])
    end
    sorted_regresults = sort(regresults; lt = sortf)
    sorted_keys = [reg.match for reg in sorted_regresults]
    [d[key] for key in sorted_keys]
end

function get_js(d)
    rj = r"jq(?<q1>[0-9]+)q(?<q2>[0-9]+)"
    regresults = [match(rj, key) for key in keys(d) if match(rj, key) !== nothing]
    ret = []
    for reg in regresults
        push!(ret, ((parse(Int, reg["q1"]), parse(Int, reg["q2"])), d[reg.match]))
    end
    ret
end

"""
    set pulses , 
    S is used to determine the system
    pulse is an array of floats that holds all the inphase , quadrature and frequencies in a sequence
    the format is 
    [ <qubit 1 inphase array> <qubit 1 quad array> <qubit 1 frequences> <qubit 2 inphase> <qubit 2 quad > ...] 
    set_pulse! will set qubit `q`'s inphase/quad/freq array to the `arr`
"""
function set_pulse!(S, pulse, q, sym::Symbol, arr::Array)
    @assert length(arr) <= pulse_count(S, pulse)
    n = pulse_count(S, pulse)
    if sym == :inphase
        pulse[(1+3*(q-1)*n):(3):(3*q*n)] .= [arr; zeros(n - length(arr))]
    elseif sym == :quad
        pulse[(2+3*(q-1)*n):(3):(3*q*n)] .= [arr; zeros(n - length(arr))]
    elseif sym == :freq
        pulse[(3+3*(q-1)*n):(3):(3*q*n)] .= [arr; zeros(n - length(arr))]
    else
        throw("sym should either be :inphase, :quad or :freq")
    end
    nothing
end

"""
    sets S.pulses with array @ref set_pulse!
"""
function set_pulse!(S, q::Int, sym::Symbol, arr::Array)
    set_pulse!(S, S.pulses, q, sym, arr)
end

function set_pulse_all!(S::System, arr::Array, freqs)
    @assert length(arr) % S.N == 0
    plen = length(arr) ÷ S.N
    for q = 1:S.N
        set_pulse!(S, q, :inphase, arr[])
        set_pulse!(S, q, :quad, arr[])
        set_pulse!(S, q, :freq, freqs[i])
    end
end

function get_pulse(S, pulse, q, sym::Symbol)
    n = pulse_count(S, pulse)
    if sym == :inphase
        return pulse[(1+3*(q-1)*n):(3):(3*q*n)]
    elseif sym == :quad
        return pulse[(2+3*(q-1)*n):(3):(3*q*n)]
    elseif sym == :freq
        return pulse[(3+3*(q-1)*n):(3):(3*q*n)]
    else
        throw("sym should either be :inphase, :quad or :freq")
    end
end

set_pulse_freq!(S, pulse, q, arr) = set_pulse!(S, pulse, q, :freq, arr)
set_pulse_inphase!(S, pulse, q, arr) = set_pulse!(S, pulse, q, :inphase, arr)
set_pulse_quad!(S, pulse, q, arr) = set_pulse!(S, pulse, q, :quad, arr)

include("evolution.jl")
include("sensitivity_M.jl")
include("sensitivity_int_H.jl")

export System,make_system,make_system,set_pulse_freq!,set_pulse_inphase!,set_pulse_quad!,set_pulse!,pulse_tspan
export evolve_oncomplex
export make_dfunc_neural,make_neuralH_prob
export make_dfunc_M,make_MDprob!
export makeNN
export plotsol,plotsol!,plotdrive,probdist
end
