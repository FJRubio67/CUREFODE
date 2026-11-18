#=
****************************************************************************
****************************************************************************
Additional functions
****************************************************************************
****************************************************************************
=#

# Standardisation function
scale(x) = (x .- mean(x)) ./ std(x)

function key2matrix(key)
    out = reshape(parse.(Int, split(key, "_")), (4, 5))
    return out
end

#=
****************************************************************************
****************************************************************************
Hazard-Response model
****************************************************************************
****************************************************************************
=#

# Hazard-Response ODE model
function HazResp(dh, h, p, t)
    # Model parameters
    lambda, kappa, alpha, beta = p

    # ODE System
    dh[1] = lambda * h[1] * (1 - h[1] / kappa) - alpha * h[1] * h[2] # hazard
    dh[2] = beta * h[2] * (1 - h[2] / kappa) - alpha * h[1] * h[2] # response
    dh[3] = h[1] # cumulative hazard
    return nothing
end

# Jacobian for Hazard-Response model

function jacHR(J, u, p, t)
    # Parameters
    lambda, kappa, alpha, beta = p
    # state variables
    h = u[1]
    q = u[2]

    # Jacobian
    J[1, 1] = lambda * (1 - 2 * h / kappa) - alpha * q
    J[1, 2] = -alpha * h
    J[1, 3] = 0.0
    J[2, 1] = -alpha * q
    J[2, 2] = beta * (1 - 2 * q / kappa) - alpha * h
    J[2, 3] = 0.0
    J[3, 1] = 1.0
    J[3, 2] = 0.0
    J[3, 3] = 0.0
    nothing
end

# Hazard-Response model with explicit Jacobian
HRJ = ODEFunction(HazResp; jac=jacHR)

#=
****************************************************************************
****************************************************************************
Hazard-Response model (log h and log q)
****************************************************************************
****************************************************************************
=#

# Hazard-Response ODE model
function HazRespL(dlh, lh, p, t)
    # Model parameters
    lambda, kappa, alpha, beta = p

    # ODE System
    dlh[1] = lambda * (1 - exp(lh[1]) / kappa) - alpha * exp(lh[2]) # log hazard
    dlh[2] = beta * (1 - exp(lh[2]) / kappa) - alpha * exp(lh[1]) # log response
    dlh[3] = exp(lh[1]) # cumulative hazard
    return nothing
end

# Jacobian for Hazard-Response model

function jacHRL(J, u, p, t)
    # Parameters
    lambda, kappa, alpha, beta = p
    # state variables
    lh = u[1]
    lq = u[2]

    # Jacobian
    J[1, 1] = -lambda * exp(lh) / kappa
    J[1, 2] = -alpha * exp(lq)
    J[1, 3] = 0.0
    J[2, 1] = -alpha * exp(lh)
    J[2, 2] = -beta * exp(lq) / kappa
    J[2, 3] = 0.0
    J[3, 1] = exp(lh)
    J[3, 2] = 0.0
    J[3, 3] = 0.0
    nothing
end

# Hazard-Response model with explicit Jacobian
HRJL = ODEFunction(HazRespL; jac=jacHRL)

#=
****************************************************************************
****************************************************************************
Functions: no covariates
****************************************************************************
****************************************************************************
=#


# Negative log likelihood function: original formulation (h and q)
function mlog_lik0(par::Vector{Float64}, times::Vector{Float64}, status::AbstractVector{Bool}, hp::Vector{Float64}, u0::Vector{Float64})
    # Parameters for the ODE
    odeparams = exp.(par)
    tmax = maximum(times)

    sol = solve(ODEProblem(HRJ, u0, [0.0, tmax], odeparams); alg_hints=[:stiff])
    OUT = sol(times)

    # h_E(t_i | θ) for all i, and for uncensored only
    hE_all  = OUT[1, :]          # length n
    HE_all  = OUT[3, :]          # length n  (cumulative hazard)

    # log(hp_i + h_E_i) via logsumexp for numerical stability
    # logsumexp([a, b]) = log(exp(a) + exp(b))
    ll_haz = sum(
        logsumexp(log(hp[i]), log(hE_all[i]))
        for i in eachindex(times) if status[i]
    )

    ll_chaz = sum(HE_all)

    return -ll_haz + ll_chaz
end


# Negative log likelihood function (log h and log q)

function mlog_lik0L(par::Vector{Float64}, times::Vector{Float64}, status::AbstractVector{Bool}, hp::Vector{Float64}, u0::Vector{Float64})
    # Parameters for the ODE
    odeparams = exp.(par)
    tmax = maximum(times)

    sol = solve(ODEProblem(HRJL, lu0, [0.0, tmax], odeparams); alg_hints=[:stiff])
    OUT = sol(times)

    # log h_E(t_i | θ) for all i, and for uncensored only
    lhE_all  = OUT[1, :]          # length n
    HE_all  = OUT[3, :]          # length n  (cumulative hazard)

    # log(hp_i + h_E_i) via logsumexp for numerical stability
    # logsumexp([a, b]) = log(exp(a) + exp(b))
    ll_haz = sum(
        logsumexp(log(hp[i]), lhE_all[i])
        for i in eachindex(times) if status[i]
    )

    ll_chaz = sum(HE_all)

    return -ll_haz + ll_chaz
end




#=
********************************************************************************************************************************************************
Function to find the MLE for the Hazard-Response excess hazard model (no covariates)

Arguments:
  times   : Vector of observed times t_i
  status  : BitVector of censoring indicators δ_i (1 = event, 0 = censored)
  hp      : Vector of population hazard values hp_i(t_i)
  lu0     : Initial conditions for the ODE (log scale)
  M       : Maximum number of iterations for the optimiser
  init    : Initial values for the parameters (log scale)
  log_scale : Boolean indicating whether to use log scale (default: true)
  opt_method  : Optimisation method (default: NelderMead())
Returns:
  optimiser : Optim optimisation object
  log_lik   : The log likelihood function (callable)
********************************************************************************************************************************************************
=#

function EHRMLE(
    times::Vector{Float64},
    status::AbstractVector{Bool},
    hp::Vector{Float64},
    u0::Vector{Float64},
    M::Int,
    init::Vector{Float64};
    log_scale::Bool = true,
    opt_method = NelderMead()
)
    # Pre-compute log population hazard for uncensored observations (outside the likelihood)
    log_hp = log.(hp[status])
    tmax = maximum(times)
    n = length(times)

    # Select ODE function based on formulation
    ode_fun = log_scale ? HRJL : HRJ

    # Log likelihood function
    log_lik = function (par::Vector{Float64})

        odeparams = exp.(par)  # [λ, κ, α, β]

        sol = solve(ODEProblem(ode_fun, u0, [0.0, tmax], odeparams); alg_hints=[:stiff])
        OUT = sol(times)

        # If log_scale: OUT[1, :] is already log h_E, so use directly
        # If not:       OUT[1, :] is h_E, so take log
        log_hE_uncens = log_scale ? OUT[1, status] : log.(OUT[1, status])

        ll_haz = sum(logsumexp(log_hp[i], log_hE_uncens[i]) for i in eachindex(log_hE_uncens))
        ll_chaz = sum(OUT[3, :])

        return ll_haz - ll_chaz
    end

    mlog_lik = par -> -log_lik(par)

    optimiser = optimize(mlog_lik, init, opt_method, Optim.Options(iterations = M))

    return optimiser, log_lik
end




















