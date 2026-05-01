using DifferentialEquations
using DiffEqGPU

# ---------------------------------------------------------
# 1. Define the ODE system
# ---------------------------------------------------------
function HazRespL(dlh, lh, p, t)
    λ, κ, α, β = p
    dlh[1] = λ * (1 - exp(lh[1]) / κ) - α * exp(lh[2]) # log hazard
    dlh[2] = β * (1 - exp(lh[2]) / κ) - α * exp(lh[1]) # log response
    dlh[3] = exp(lh[1])                                # cumulative hazard
    return nothing
end

# ---------------------------------------------------------
# 2. Jacobian function
# ---------------------------------------------------------
function jacHRL(J, u, p, t)
    λ, κ, α, β = p
    lh, lq = u[1], u[2]

    J[1, 1] = -λ * exp(lh) / κ
    J[1, 2] = -α * exp(lq)
    J[1, 3] = 0.0
    J[2, 1] = -α * exp(lh)
    J[2, 2] = -β * exp(lq) / κ
    J[2, 3] = 0.0
    J[3, 1] = exp(lh)
    J[3, 2] = 0.0
    J[3, 3] = 0.0
    nothing
end

# ---------------------------------------------------------
# 3. Define ODEFunction with explicit Jacobian
# ---------------------------------------------------------
HRJL = ODEFunction(HazRespL; jac = jacHRL)

# ---------------------------------------------------------
# 4. Example setup for N different problems
# ---------------------------------------------------------
N = 10000  # number of simulations

# Each trajectory has its own parameters and max time
tmaxs = rand(1.0:0.5:10.0, N)  # different time horizons
params = [ [1.5 + 0.1*randn(), 1.0, 0.75 + 0.05*randn(), 1.0 + 0.1*randn()] for i in 1:N ]

lu0 = [log(1e-3), log(1e-6), 0.0] # shared initial condition

# ---------------------------------------------------------
# 5. Define the base problem
# ---------------------------------------------------------
base_prob = ODEProblem(HRJL, lu0, (0.0, 1.0), params[1])

# ---------------------------------------------------------
# 6. Define how to generate new problems
# ---------------------------------------------------------
function prob_func(prob, i, repeat)
    ODEProblem(prob.f, prob.u0, (0.0, tmaxs[i]), params[i])
end

# ---------------------------------------------------------
# 7. Create the EnsembleProblem
# ---------------------------------------------------------
ensemble_prob = EnsembleProblem(base_prob; prob_func = prob_func)

# ---------------------------------------------------------
# 8. Solve on GPU!
# ---------------------------------------------------------
sol = solve(
    ensemble_prob,
    Rodas5P(),  # stiff solver, GPU-compatible
    EnsembleGPUArray(),  # run on GPU
    trajectories = N,
    batch_size = 1024,   # tune this depending on GPU memory
    saveat = 0.1
)


using DifferentialEquations

sol = solve(
    ensemble_prob,
    Rodas5P(),          # same stiff solver
    EnsembleThreads(),  # parallel execution across CPU threads
    trajectories = N,
    batch_size = 50,    # adjust depending on number of cores
    saveat = 0.1
)










using DifferentialEquations, BenchmarkTools, Base.Threads

# Define your ODE and Jacobian (as before)
function HazRespL(dlh, lh, p, t)
    λ, κ, α, β = p
    dlh[1] = λ * (1 - exp(lh[1]) / κ) - α * exp(lh[2])
    dlh[2] = β * (1 - exp(lh[2]) / κ) - α * exp(lh[1])
    dlh[3] = exp(lh[1])
    return nothing
end

HRJL = ODEFunction(HazRespL)
lu0 = [log(1e-3), log(1e-6), 0.0]
N = 1000
tmaxs = rand(1.0:0.5:10.0, N)
params = [[1.5, 1.0, 0.75, 1.0] for _ in 1:N]

base_prob = ODEProblem(HRJL, lu0, (0.0, 1.0), params[1])
function prob_func(prob, i, repeat)
    ODEProblem(prob.f, prob.u0, (0.0, tmaxs[i]), params[i])
end
ensemble_prob = EnsembleProblem(base_prob; prob_func)

# --- EnsembleThreads() ---
@btime solve($ensemble_prob, Rodas5P(), EnsembleThreads(), trajectories=$N, save_everystep=false);

# --- Threads.@threads for loop ---
@btime begin
    sols = Vector{ODESolution}(undef, $N)
    @threads for i in 1:$N
        prob = ODEProblem(HRJL, lu0, (0.0, tmaxs[i]), params[i])
        sols[i] = solve(prob, Rodas5P(), save_everystep=false)
    end
    sols
end




function mlog_likLMT_opt(par)
    # --- 1. Expand parameters for all trajectories ---
    # precompute odeparams (n × 4)
    odeparams = exp.(hcat(
        des_l * par[1:p_l],
        des_k * par[(p_l+1):(p_l+p_k)],
        des_a * par[(p_l+p_k+1):(p_l+p_k+p_a)],
        des_b * par[(p_l+p_k+p_a+1):(p_l+p_k+p_a+p_b)]
    ))

    # --- 2. Preallocate output array ---
    OUT = zeros(Float64, n, 3)

    # --- 3. Threaded loop over all trajectories ---
    Threads.@threads for i in 1:n
        # Create local copy of parameters to avoid thread race
        p_i = @view odeparams[i, :]

        # Use a fresh ODEProblem per trajectory, but reuse lu0
        prob_i = ODEProblem(HRJL, lu0, (tspan0[i, 1], tspan0[i, 2]), p_i)

        # Solve ODE with stiff solver
        sol = solve(prob_i; alg_hints=[:stiff], save_everystep=false)

        # Directly store final state
        @inbounds OUT[i, :] = sol.u[end]
    end

    # --- 4. Compute log-likelihood ---
    a = OUT[status, 1]
    b = log_mort_h[status]

    # Log-hazard term
    ll_haz = sum(logsumexp((a[i], b[i])) for i in eachindex(a))

    # Cumulative hazard term
    ll_chaz = sum(OUT[:, 3])

    # Return negative log-likelihood
    return -ll_haz + ll_chaz
end
