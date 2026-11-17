#=
****************************************************************************
Required packages
****************************************************************************
=#

using Plots
using DifferentialEquations
using LinearAlgebra
using CSV
using LSODA
using Optim
using Distributions
using Random
using AdaptiveMCMC
using Tables
using DelimitedFiles
using Statistics
using Survival
using DataFrames
using FreqTables
using Sundials
using ForwardDiff
using Turing
using StatsPlots
using StatsFuns
using JLD2
using RData
using StatsFuns
using HazReg
using LogExpFunctions
using NetSurvival
using StaticArrays, DiffResults 

# Additional routines
include("routines.jl")



#=
****************************************************************************
Data preparation
****************************************************************************
=#


#= Data =#
df_full = load("breastla.rda")
df = df_full["breastla"]


# Removing patients with missing grade Data
indobs = zeros(size(df)[1])
for i in 1:size(df)[1]
    if (df.grade[i] == 9)
        indobs[i] = 0
    else
        indobs[i] = 1
    end
end
indobs = collect(Bool, (indobs))

df = df[indobs,:]

# Removing patients with infinite mortality hazard
df = df[.!isinf.(df.mort_h), :]

# Removing patients with na in mort_h
df = df[.!isnan.(df.mort_h), :]

# Removing patients with time equal to zero
df = df[df.Time .> 0.0, :]

# Sorting df by time
sorted_indices = sortperm(df[:, :Time])

df = df[sorted_indices,:]

# Sample size
n = size(df)[1]

#= Vital status =#
status = collect(Bool, (df.Status));

#= grades matrix =#
n = length(df.grade)
k = 4  # number of categories (columns)
grades = zeros(Float64, n, k)

for i in 1:n
    grades[i, Int(df.grade[i])] = 1.0
end

#= Survival times =#
times = copy(df.Time);

#= Population hazard =#
hp = copy(df.mort_h);


#=
****************************************************************************
Fitting the model without covariates: MLE 
****************************************************************************
=#

# Initial conditions on the natural scale
h0q0  = [1.0e-3, 1.0e-6]

# Starting values for log(λ, κ, α, β): avoid all zeros (degenerate case)
init0 = log.([1.0, 1.0, 1.0, 1.0])

# MLE: no covariates
optmle0 = EHRMLE(times, status, hp, h0q0, 10000, init0)
LMLE0   = optmle0[1].minimizer
Optim.converged(optmle0[1])


#=
****************************************************************************
Fitted ODE solutions: checks
****************************************************************************
=#

# Solve the fitted model on a grid that is fine near t = 0 (log-spaced + linear)
function solve_fit(par; h0q0 = h0q0, tmax = maximum(times), alg = Rodas5P())
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

# Fits to compare (drop one if you only have a single fit)
fits = [("start: zeros", fitA), ("start: init0", fitB)]

p1 = plot(title = "Excess hazard", xlabel = "Time (years)", ylabel = "h_E(t)")
p2 = plot(title = "Excess hazard (log–log)", xlabel = "Time (years)", ylabel = "h_E(t)",
          xscale = :log10, yscale = :log10)
p3 = plot(title = "Response (fraction of κ)", xlabel = "Time (years)", ylabel = "q(t)/κ",
          xscale = :log10, yscale = :log10)
p4 = plot(title = "Net survival", xlabel = "Time (years)", ylabel = "S_N(t)", ylim = (0, 1))

for (lab, f) in fits
    par = f[1].minimizer
    λ, κ, α, μ = exp.(par)
    a   = α * κ
    reg = attractor(λ, κ, α, μ)
    s   = solve_fit(par)

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

# Observed event times, as a rug on the hazard panel
vline!(p1, times[status], alpha = 0.03, color = :grey, label = false)

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

