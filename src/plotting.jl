using Plots

function plotsol(sol, S, amps = :all; title = "fig", legend = :outerright)
    if amps == :all
        plotamps = 0:(S.dim)-1
    else
        plotamps = amps
    end
    plt = Plots.plot()
    for amp in plotamps
        Plots.plot!(
            sol.t,
            [abs(cstate(sol(t), S)[amp+1])^2 for t in sol.t],
            linewidth = 1,
            label = string(amp , base = Int(round((S.dim)^(1 / S.N))), pad = S.N),
            ticks = :native,
            title = title,
        )
    end
    Plots.plot!(sol.t, [norm(sol(t)) for t in sol.t], label = "norm", legend = legend)
    plt
end

function plotsol!(plt, sol, S, amps = :all; title = "fig", legend = :outerright)
    if amps == :all
        plotamps = 0:(S.dim)-1
    else
        plotamps = amps
    end
    for amp in plotamps
        Plots.plot!(
            sol.t,
            [abs(cstate(sol(t), S)[amp+1])^2 for t in sol.t],
            linewidth = 1,
            label = string(amp , base = Int(round((S.dim)^(1 / S.N))), pad = S.N),
            ticks = :native,
            title = title,
        )
    end
    Plots.plot!(sol.t, [norm(sol(t)) for t in sol.t], label = "norm", legend = legend)
    plt
end

function plotdrive(pulses,S ; qubits = 1:S.N,syms=[:inphase,:quad],title ="")
    plt = Plots.plot(title=title,ticks=:native)
    tspan = pulse_tspan(S, pulses)
    n = length(get_pulse(S, pulses, 1, :inphase))
    tarray = LinRange(tspan[1], tspan[2], n)
    for i in qubits
        :inphase in syms ? plot!(tarray, get_pulse(S, pulses, i, :inphase), label = "$i inphase") : nothing
        :quad in syms ? plot!(tarray, get_pulse(S, pulses, i, :quad), label = "$i quad") : nothing
        :freq in syms ? plot!(tarray, get_pulse(S, pulses, i, :freq), label = "$i freq") : nothing
    end
    plt
end
