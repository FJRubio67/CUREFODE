#=
****************************************************************************
****************************************************************************
Routines for the hazard-response excess hazard model

Functions only: no data are loaded or created in this file, so it can be
include()d at the top of the analysis script.

Required packages:
    using OrdinaryDiffEqRosenbrock, SciMLBase, Optim, LogExpFunctions,
          Statistics, LinearAlgebra, StaticArrays, DiffResults, ForwardDiff
****************************************************************************
****************************************************************************
=#

#=
****************************************************************************
Additional functions
****************************************************************************
=#

# Standardisation (named `standardise` to avoid clashing with Distributions.scale)
standardise(x) = (x .- mean(x)) ./ std(x)

# Treatment-coded dummies: one column per level, except the reference level
function dummies(x; ref = minimum(unique(x)))
    levs = sort(setdiff(unique(x), [ref]))
    return Float64.(x .== permutedims(levs)), levs
end

function key2matrix(key)
    out = reshape(parse.(Int, split(key, "_")), (4, 5))
    return out
end

#=
****************************************************************************
Hazard-response model, in-place versions
u = (h, q, H) or (log h, log q, H),  p = (λ, κ, α, β)
****************************************************************************
=#

# Natural scale
function HazResp(dh, h, p, t)
    lambda, kappa, alpha, beta = p
    dh[1] = lambda * h[1] * (1 - h[1] / kappa) - alpha * h[1] * h[2]   # hazard
    dh[2] = beta * h[2] * (1 - h[2] / kappa) - alpha * h[1] * h[2]     # response
    dh[3] = h[1]                                                       # cumulative hazard
    return nothing
end

function jacHR(J, u, p, t)
    lambda, kappa, alpha, beta = p
    h, q = u[1], u[2]
    J[1, 1] = lambda * (1 - 2 * h / kappa) - alpha * q
    J[1, 2] = -alpha * h
    J[1, 3] = 0.0
    J[2, 1] = -alpha * q
    J[2, 2] = beta * (1 - 2 * q / kappa) - alpha * h
    J[2, 3] = 0.0
    J[3, 1] = 1.0
    J[3, 2] = 0.0
    J[3, 3] = 0.0
    return nothing
end

HRJ = ODEFunction(HazResp; jac = jacHR)

# Log scale for h and q
function HazRespL(dlh, lh, p, t)
    lambda, kappa, alpha, beta = p
    dlh[1] = lambda * (1 - exp(lh[1]) / kappa) - alpha * exp(lh[2])   # log hazard
    dlh[2] = beta * (1 - exp(lh[2]) / kappa) - alpha * exp(lh[1])     # log response
    dlh[3] = exp(lh[1])                                               # cumulative hazard
    return nothing
end

function jacHRL(J, u, p, t)
    lambda, kappa, alpha, beta = p
    lh, lq = u[1], u[2]
    J[1, 1] = -lambda * exp(lh) / kappa
    J[1, 2] = -alpha * exp(lq)
    J[1, 3] = 0.0
    J[2, 1] = -alpha * exp(lh)
    J[2, 2] = -beta * exp(lq) / kappa
    J[2, 3] = 0.0
    J[3, 1] = exp(lh)
    J[3, 2] = 0.0
    J[3, 3] = 0.0
    return nothing
end

HRJL = ODEFunction(HazRespL; jac = jacHRL)

#=
****************************************************************************
Hazard-response model on the log scale, static-array (out-of-place) version
Used by EHRMLE_cov
****************************************************************************
=#
function HazRespL_s(u, p, t)
    λ, κ, α, β = p
    h, q = exp(u[1]), exp(u[2])
    return SVector(λ * (1 - h / κ) - α * q,
                   β * (1 - q / κ) - α * h,
                   h)
end

function jacHRL_s(u, p, t)
    λ, κ, α, β = p
    h, q = exp(u[1]), exp(u[2])
    T = promote_type(eltype(u), eltype(p))
    z = zero(T)
    # Column-major order: column 1, column 2, column 3
    return SMatrix{3,3,T}(-λ * h / κ, -α * h, h,
                          -α * q, -β * q / κ, z,
                          z, z, z)
end

# The system is autonomous
tgradHRL_s(u, p, t) = zero(u)

HRJL_s = ODEFunction{false}(HazRespL_s; jac = jacHRL_s, tgrad = tgradHRL_s)

#=
********************************************************************************
EHRMLE: MLE for the hazard-response excess hazard model (no covariates)

  times     : observed times t_i
  status    : event indicators δ_i
  hp        : population hazard h_P(age_i + t_i)
  h0q0      : (h0, q0) on the NATURAL scale
  M         : maximum number of iterations
  init      : initial values for log(λ, κ, α, β)
  log_scale : solve for (log h, log q) (default) or (h, q)

Returns (optimiser, log_lik)
********************************************************************************
=#
function EHRMLE(
    times::AbstractVector{<:Real},
    status::AbstractVector{Bool},
    hp::AbstractVector{<:Real},
    h0q0::AbstractVector{<:Real},
    M::Int,
    init::AbstractVector{<:Real};
    log_scale::Bool = true,
    opt_method = NelderMead(),
    alg = Rodas5P(),
    reltol = 1e-8,
    abstol = 1e-10,
)
    n = length(times)
    @assert length(status) == n && length(hp) == n
    @assert all(>(0), h0q0[1:2])

    # Initial condition on the ODE's scale; H(0) = 0 always
    u0 = log_scale ? [log(h0q0[1]), log(h0q0[2]), 0.0] :
                     [float(h0q0[1]), float(h0q0[2]), 0.0]
    ode_fun = log_scale ? HRJL : HRJ

    tgrid     = sort(unique(Float64.(times)))
    ev        = findall(status)
    log_hp_ev = log.(hp[ev])

    prob = ODEProblem(ode_fun, u0, (0.0, tgrid[end]), exp.(init))

    function log_lik(par::AbstractVector)
        sol = solve(remake(prob; p = exp.(par)), alg;
                    saveat = tgrid, reltol = reltol, abstol = abstol)
        SciMLBase.successful_retcode(sol) || return -Inf

        idx = searchsortedfirst.(Ref(sol.t), times)
        U   = Array(sol)

        if log_scale
            lhE = U[1, idx[ev]]
        else
            hE = U[1, idx[ev]]
            any(<=(0), hE) && return -Inf
            lhE = log.(hE)
        end

        ll = sum(logaddexp.(log_hp_ev, lhE)) - sum(U[3, idx])
        return isfinite(ll) ? ll : -Inf
    end

    optimiser = optimize(p -> -log_lik(p), collect(float.(init)), opt_method,
                         Optim.Options(iterations = M))
    return optimiser, log_lik
end

#=
****************************************************************************
Group individuals with identical rows of X
Returns the G × p matrix of unique rows and, for each group:
  tg  : sorted unique observed times in the group
  idx : position in tg of each member's time
  d   : event indicators of the members
  lhp : log population hazard of the members
****************************************************************************
=#
function make_groups(times, status, hp, X)
    n    = length(times)
    rows = [X[i, :] for i in 1:n]
    ux   = unique(rows)
    pos  = Dict(r => g for (g, r) in enumerate(ux))
    members = [Int[] for _ in ux]
    for i in 1:n
        push!(members[pos[rows[i]]], i)
    end
    groups = map(members) do m
        tg = sort(unique(Float64.(times[m])))
        (tg  = tg,
         idx = searchsortedfirst.(Ref(tg), times[m]),
         d   = collect(Bool, status[m]),
         lhp = log.(hp[m]))
    end
    Xg = reduce(vcat, permutedims.(ux))
    return Xg, groups
end

#=
****************************************************************************
Log-likelihood contribution of one group as a function of η = log θ
****************************************************************************
=#
function group_ll(η::SVector{4,T}, g, u0, alg, reltol, abstol) where {T}
    θ    = exp.(η)
    prob = ODEProblem{false}(HRJL_s, SVector{3,T}(u0), (0.0, g.tg[end]), θ)
    sol  = solve(prob, alg; saveat = g.tg, save_start = false,
                 reltol = reltol, abstol = abstol)
    SciMLBase.successful_retcode(sol) || return T(-Inf)
    length(sol.u) == length(g.tg) || return T(-Inf)

    ll = zero(T)
    @inbounds for (j, m) in enumerate(g.idx)
        u = sol.u[m]                       # (log h_E(t), log q(t), H_E(t))
        ll += (g.d[j] ? logaddexp(g.lhp[j], u[1]) : zero(T)) - u[3]
    end
    return ll
end

#=
********************************************************************************
EHRMLE_cov: MLE for the hazard-response excess hazard model with covariates

log(θ_{k,i}) = X[i, index_sel[k,:] .== 1]' β_k,   θ = (λ, κ, α, β)

  times     : observed times t_i
  status    : event indicators δ_i
  hp        : population hazard h_P(age_i + t_i)
  X         : n × p design matrix, first column = intercept
  index_sel : 4 × p 0/1 matrix (rows λ, κ, α, β); column 1 must be 1
  h0q0      : (h0, q0) on the natural scale
  M         : maximum number of iterations
  init      : initial values, layout [β_λ; β_κ; β_α; β_β]

Individuals are grouped by the covariates used by at least one parameter,
so dropping a covariate from all four regressions reduces the number of
ODE solves. With intercepts only, there is a single solve.

Returns a named tuple with
  optimiser : Optim result
  log_lik   : par -> log-likelihood
  neg_hess  : par -> Hessian of the negative log-likelihood (observed information)
  sel       : selected columns of X for each parameter
  blocks    : positions of each parameter's coefficients in par
  ngroups   : number of distinct covariate patterns (ODE solves per evaluation)
********************************************************************************
=#
function EHRMLE_cov(
    times::AbstractVector{<:Real},
    status::AbstractVector{Bool},
    hp::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    index_sel::AbstractMatrix{<:Integer},
    h0q0::AbstractVector{<:Real},
    M::Int,
    init::AbstractVector{<:Real};
    opt_method = NewtonTrustRegion(),
    alg = Rodas5P(),
    reltol = 1e-8,
    abstol = 1e-10,
)
    n = length(times)
    @assert size(X, 1) == n && length(status) == n && length(hp) == n
    @assert size(index_sel) == (4, size(X, 2))
    @assert all(index_sel[:, 1] .== 1) "All four intercepts must be included"
    @assert all(>(0), h0q0[1:2])

    # Columns used by at least one parameter; group on those only
    used = findall(vec(any(index_sel .== 1; dims = 1)))
    Xg, groups = make_groups(times, status, hp, X[:, used])
    G = length(groups)

    # Selected columns (w.r.t. X, for predictions) and group design matrices
    sel    = [findall(index_sel[k, :] .== 1) for k in 1:4]
    desg   = [Xg[:, findall(index_sel[k, used] .== 1)] for k in 1:4]   # G × p_k
    p      = length.(sel)
    ends   = cumsum(p)
    starts = ends .- p .+ 1
    blocks = [starts[k]:ends[k] for k in 1:4]
    npar   = ends[end]
    @assert length(init) == npar "init must have length $npar"

    u0 = SVector(log(float(h0q0[1])), log(float(h0q0[2])), 0.0)

    # Value (order 0), + gradient (1), + Hessian (2) w.r.t. η for every group
    function group_derivs(par, order)
        E     = hcat((desg[k] * par[blocks[k]] for k in 1:4)...)   # G × 4
        vals  = zeros(G)
        grads = zeros(G, 4)
        hess  = zeros(G, 4, 4)
        Threads.@threads for g in 1:G
            ηg = SVector{4,Float64}(E[g, 1], E[g, 2], E[g, 3], E[g, 4])
            f  = η -> group_ll(η, groups[g], u0, alg, reltol, abstol)
            if order == 0
                vals[g] = f(ηg)
            elseif order == 1
                res = ForwardDiff.gradient!(DiffResults.GradientResult(ηg), f, ηg)
                vals[g]     = DiffResults.value(res)
                grads[g, :] = DiffResults.gradient(res)
            else
                res = ForwardDiff.hessian!(DiffResults.HessianResult(ηg), f, ηg)
                vals[g]       = DiffResults.value(res)
                grads[g, :]   = DiffResults.gradient(res)
                hess[g, :, :] = DiffResults.hessian(res)
            end
        end
        return vals, grads, hess
    end

    # Objective (negative log-likelihood), gradient and Hessian for Optim
    function fgh!(F, Gr, H, par)
        order = H !== nothing ? 2 : (Gr !== nothing ? 1 : 0)
        vals, gr, hs = group_derivs(par, order)
        ll = sum(vals)

        if !isfinite(ll) || any(!isfinite, gr) || any(!isfinite, hs)
            Gr === nothing || fill!(Gr, 0.0)
            H  === nothing || fill!(H, 0.0)
            return F === nothing ? nothing : Inf
        end

        if Gr !== nothing
            for k in 1:4
                Gr[blocks[k]] .= -(desg[k]' * gr[:, k])
            end
        end
        if H !== nothing
            for k in 1:4, l in 1:4
                H[blocks[k], blocks[l]] .= -(desg[k]' * (hs[:, k, l] .* desg[l]))
            end
        end
        return F === nothing ? nothing : -ll
    end

    log_lik(par)  = sum(group_derivs(par, 0)[1])
    neg_hess(par) = (H = zeros(npar, npar); fgh!(nothing, nothing, H, par); H)

    optimiser = optimize(Optim.only_fgh!(fgh!), collect(float.(init)), opt_method,
                         Optim.Options(iterations = M))

    return (optimiser = optimiser, log_lik = log_lik, neg_hess = neg_hess,
            sel = sel, blocks = blocks, ngroups = G)
end

#=
****************************************************************************
Attractor regime (Theorem 1); boundary cases have measure zero
****************************************************************************
=#
function attractor(λ, κ, α, μ)
    a = α * κ
    λ < a && μ > a && return :response_wins   # h_E → 0 exponentially, H_E(∞) < ∞
    λ > a && μ < a && return :hazard_wins
    λ > a && μ > a && return :coexistence
    λ < a && μ < a && return :bistable        # depends on (h0, q0): use Theorem 1(iv)(d)
    return :boundary
end