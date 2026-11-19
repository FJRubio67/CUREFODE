#=
****************************************************************************
Hazard-response excess hazard model: colonDC data, no covariates

Data: colonDC (R package cuRe) with the population hazard at exit taken
from rstpm2::popmort (Finnish life table), exported from R as
colonDC_hp.csv (see colonDC_hp.R).
****************************************************************************
=#

#=
****************************************************************************
Required packages
****************************************************************************
=#

using CSV, DataFrames, Statistics, LinearAlgebra, FreqTables
using OrdinaryDiffEqRosenbrock, SciMLBase, Optim, LogExpFunctions
using StaticArrays, DiffResults, ForwardDiff
using Plots

# Additional routines
include("routines.jl")

#=
****************************************************************************
Data preparation
****************************************************************************
=#

df = CSV.read("colonDC_hp.csv", DataFrame; missingstring = "NA")

# Remove patients with unknown stage
freqtable(df.stage)                                   # check the labels first
df = df[lowercase.(string.(df.stage)) .!= "unknown", :]

# Complete records for the variables used
dropmissing!(df, [:FU, :status, :mort_h, :stage, :age, :sex])

# Valid population hazard
df = df[isfinite.(df.mort_h) .& (df.mort_h .> 0), :]

# Follow-up in years. FU is in days: zero follow-up is set to half a day
# instead of being dropped, so very early deaths stay in the data.
println("Zero follow-up times: ", sum(df.FU .== 0))
df.time = max.(Float64.(df.FU), 0.5) ./ 365.24

# Sort by time
sort!(df, :time)

times  = Float64.(df.time)
status = collect(Bool, df.status .== 1)
hp     = Float64.(df.mort_h)
n      = length(times)

println("n = ", n, ", events = ", sum(status),
        ", max follow-up = ", round(maximum(times), digits = 2), " years")
freqtable(df.stage)

# Check that follow-up is not heaped on a coarse grid (unlike breastla)
println("Distinct follow-up times (days): ", length(unique(df.FU)))

# Crude early excess hazard: (observed - expected deaths) / person-time,
# using the population hazard at exit (rough; only to inform h0)
for c in [30, 91, 182, 365] ./ 365.24
    pt = min.(times, c)
    d  = sum(status .& (times .<= c))
    e  = sum(hp .* pt)
    println("[0, ", round(c, digits = 3), "] years: observed ", d,
            ", expected ", round(e, digits = 1),
            ", crude excess hazard ", round((d - e) / sum(pt), digits = 4))
end

#=
****************************************************************************
Fitting the model without covariates: MLE
Several starting values for each h0 (the likelihood can be multimodal or
have ridges); the best fit for each h0 is kept.
****************************************************************************
=#

q0      = 1.0e-6
h0_grid = [1.0e-3, 1.0e-2, 1.0e-1, 0.3, 1.0]

# Starting values for log(λ, κ, α, β)
starts = [zeros(4),
          log.([0.1, 0.1, 0.1, 0.1]),
          log.([1.0, 0.1, 5.0, 2.0]),
          log.([0.5, 1.0, 1.0, 1.0])]

results = DataFrame(h0 = Float64[], start = Int[], loglik = Float64[],
                    converged = Bool[], logkappa = Float64[])
best_fits = Dict{Float64, Any}()

for h0 in h0_grid
    best = nothing
    for (j, s) in enumerate(starts)
        f  = EHRMLE(times, status, hp, [h0, q0], 10000, s)
        ll = -Optim.minimum(f[1])
        push!(results, (h0, j, ll, Optim.converged(f[1]), f[1].minimizer[2]))
        if best === nothing || ll > best.loglik
            best = (fit = f, loglik = ll)
        end
    end
    best_fits[h0] = best
end

show(results, allrows = true)

# Large |log κ| suggests the optimiser is running along the κ → ∞ ridge
for h0 in h0_grid
    lk = best_fits[h0].fit[1].minimizer[2]
    abs(lk) > 20 && @warn "h0 = $h0: log κ = $(round(lk, digits = 1)); possible ridge"
end

# Overall best
h0_best = h0_grid[argmax([best_fits[h0].loglik for h0 in h0_grid])]
LMLE0   = best_fits[h0_best].fit[1].minimizer
println("\nBest h0 = ", h0_best, ", log-likelihood = ", best_fits[h0_best].loglik)

#=
****************************************************************************
Fitted ODE solutions: checks
****************************************************************************
=#

# Solve the fitted model on a grid that is fine near t = 0 (log-spaced + linear)
function solve_fit(par, h0q0; tmax = maximum(times), alg = Rodas5P())
    u0    = [log(h0q0[1]), log(h0q0[2]), 0.0]
    tgrid = sort(unique(vcat(0.0,
                             exp10.(range(-4, log10(tmax), length = 400)),
                             range(0.0, tmax, length = 400))))
    sol = solve(ODEProblem(HRJL, u0, (0.0, tmax), exp.(par)), alg;
                saveat = tgrid, reltol = 1e-8, abstol = 1e-10)
    U = Array(sol)
    return (t = sol.t, h = exp.(U[1, :]), q = exp.(U[2, :]), H = U[3, :],
            retcode = sol.retcode)
end

p1 = plot(title = "Excess hazard", xlabel = "Time (years)", ylabel = "h_E(t)")
p2 = plot(title = "Excess hazard (log–log)", xlabel = "Time (years)", ylabel = "h_E(t)",
          xscale = :log10, yscale = :log10)
p3 = plot(title = "Response (fraction of κ)", xlabel = "Time (years)", ylabel = "q(t)/κ",
          xscale = :log10, yscale = :log10)
p4 = plot(title = "Net survival", xlabel = "Time (years)", ylabel = "S_N(t)", ylim = (0, 1))

for h0 in h0_grid
    f   = best_fits[h0].fit
    par = f[1].minimizer
    λ, κ, α, μ = exp.(par)
    a   = α * κ
    reg = attractor(λ, κ, α, μ)
    s   = solve_fit(par, [h0, q0])
    lab = "h0 = $h0"

    println("\n", lab, "  (retcode: ", s.retcode, ")")
    println("  loglik      = ", f[2](par))
    println("  λ, κ, α, μ  = ", round.([λ, κ, α, μ], sigdigits = 4))
    println("  a = ακ      = ", round(a, sigdigits = 4), "   regime: ", reg)
    println("  peak h_E    = ", round(maximum(s.h), sigdigits = 4),
            " at t = ", round(s.t[argmax(s.h)], sigdigits = 3))
    println("  S_N(tmax)   = ", round(exp(-s.H[end]), sigdigits = 4))
    if reg == :response_wins
        # Tail correction: h_E(t) ≈ h_E(T) e^{-(a-λ)(t-T)} beyond T
        Hinf = s.H[end] + s.h[end] / (a - λ)
        println("  cure fraction exp(-H_E(∞)) ≈ ", round(exp(-Hinf), sigdigits = 4))
    end

    pos = (s.t .> 0) .& (s.h .> 0) .& (s.q .> 0)   # positive values for log axes
    plot!(p1, s.t, s.h, lw = 2, label = "$lab ($reg)")
    plot!(p2, s.t[pos], s.h[pos], lw = 2, label = false)
    plot!(p3, s.t[pos], s.q[pos] ./ κ, lw = 2, label = false)
    plot!(p4, s.t, exp.(-s.H), lw = 2, label = false)
end

plot(p1, p2, p3, p4, layout = (2, 2), size = (1000, 750), legend = :topright)


#=
****************************************************************************
Data preparation for model with covariates
****************************************************************************
=#

# Standardisation (renamed from `scale` to avoid clashing with Distributions.scale);
# move this to routines.jl
standardise(x) = (x .- mean(x)) ./ std(x)

# Treatment-coded dummies: one column per level, except the reference level
function dummies(x; ref = minimum(unique(x)))
    levs = sort(setdiff(unique(x), [ref]))
    return Float64.(x .== permutedims(levs)), levs
end

# Covariates: age (standardised) + race + grade
age_mean, age_sd = mean(df.age), std(df.age)   # keep these for predictions
race_d,  race_levs  = dummies(df.race)          # reference: lowest race code
grade_d, grade_levs = dummies(df.grade)         # reference: lowest grade

# Full design matrix, intercept included
X = hcat(ones(n), (df.age .- age_mean) ./ age_sd, race_d, grade_d)
colnames = vcat("(Intercept)", "age (std)",
                "race = " .* string.(race_levs),
                "grade = " .* string.(grade_levs))

@assert rank(X) == size(X, 2) "Design matrix is rank deficient: check the coding"
freqtable(df.grade), freqtable(df.race)          # check the levels and reference categories

# Design matrices for each parameter: all covariates enter every regression
# log(λ_i) = des_l[i,:]' β_λ,   log(κ_i) = des_k[i,:]' β_κ,
# log(α_i) = des_a[i,:]' β_α,   log(β_i) = des_b[i,:]' β_β
des_l = X
des_k = X
des_a = X
des_b = X

# Parameter-vector layout: [β_λ; β_κ; β_α; β_β], each block starting with its intercept
p       = [size(des_l, 2), size(des_k, 2), size(des_a, 2), size(des_b, 2)]
ends    = cumsum(p)
starts  = ends .- p .+ 1
blocks  = [starts[j]:ends[j] for j in 1:4]
indint  = starts                          # intercept positions
indbeta = setdiff(1:ends[end], indint)    # regression-coefficient positions
npar    = ends[end]

# Selection index for the saturated model (rows: λ, κ, α, β; columns: columns of X)
index_sel0 = ones(Int, 4, size(X, 2))

# Initial values: no-covariate MLE in the intercepts, zero slopes
init = zeros(npar)
init[indint] .= LMLE0


#=
****************************************************************************
Fitting the model with covariates: MLE
****************************************************************************
=#

Threads.nthreads()                       # should be > 1

# Saturated model: every covariate in every parameter
index_sel0 = ones(Int, 4, size(X, 2))

# Initial values: no-covariate MLE in the intercepts, zero slopes
init = zeros(npar)
init[indint] .= LMLE0

# Time one likelihood evaluation before running the optimiser
tmp = EHRMLE_cov(times, status, hp, X, index_sel0, h0q0, 0, init)
@time tmp.log_lik(init)
@time ForwardDiff.gradient(tmp.log_lik, init)

# MLE
fit    = EHRMLE_cov(times, status, hp, X, index_sel0, h0q0, 200, init)
optmle = fit.optimiser
Optim.converged(optmle)
MLE    = optmle.minimizer

Hess = fit.neg_hess(MLE)                        # exact observed information
SE   = sqrt.(diag(inv(Symmetric(Hess))))

θhat = hcat((exp.(X[:, fit.sel[k]] * MLE[fit.blocks[k]]) for k in 1:4)...)
freqtable(attractor.(θhat[:, 1], θhat[:, 2], θhat[:, 3], θhat[:, 4]))

