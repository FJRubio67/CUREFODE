
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

# log-population hazard
log_mort_h = log.(df.mort_h);

# Time grids
tspan0 = hcat(zeros(n), df.Time);

tspan00 = vcat(0.0, df.Time);
tmax = maximum(df.Time)

# Initial conditions (h,q,H)
u0 = [1.0e-3, 1.0e-6, 0.0]

# Initial conditions (log h,log q,H)
lu0 = [log(1.0e-3), log(1.0e-6), 0.0]

#=
****************************************************************************
Data preparation for best model
****************************************************************************
=#

# Design matrix including variables of interest for model building
# age + sizes + nodes + trt
des_full = hcat( scale(df.age), df.race .- 1.0, grades[:,1:3])

# Design matrices for the regression models for each parameter
#(intercept,age,sizes,nodes,trt)
# lambda: intercept + nodes + trt
#des_l = hcat(ones(n), des_full[:,3:4]);
des_l = copy(des_full)
p_l = size(des_l)[2];
# kappa: intercept + nodes
des_k = copy(des_full)
p_k = size(des_k)[2];
# alpha: intercept + nodes + trt
des_a = copy(des_full)
p_a = size(des_a)[2];
# beta: intercept + age + sizes + trt
des_b = copy(des_full)
p_b = size(des_b)[2];

# Intercept positions
indint = [1, p_l + 1, p_l + p_k + 1, p_l + p_k + p_a + 1]
indbeta = deleteat!(collect(1:(p_l+p_k+p_a+p_b)), indint)

#=
****************************************************************************
Kaplan-Meier estimate
****************************************************************************
=#

# Kaplan-Meier estimator 
km = fit(KaplanMeier, df.Time, df.Status)
ktimes = sort(unique(times))
ksurvival = km.survival

# Comparison
plot(ktimes, ksurvival,
    xlabel = "Time (years)", ylabel = "Population Survival", title = "",
  xlims = (0.0001,maximum(times)),   xticks = 0:2:maximum(times), label = "", 
  xtickfont = font(16, "Courier"),  ytickfont = font(16, "Courier"),
  xguidefontsize=18, yguidefontsize=18, linewidth=3,
  linecolor = "gray", ylims = (0,1), linestyle=:solid)

#=
****************************************************************************
Fitting the model without covariates: MLE 
****************************************************************************
=#

# MLE: No covariates
optmle0 = optimize(mlog_likL0, [0.0,0.0,0.0,0.0], method=NelderMead(), iterations=10000)

MLE0 = optmle0.minimizer


#=
****************************************************************************
Fitting the model with covariates: MLE 
****************************************************************************
=#

# Index for saturated model
index_sel0 =  vcat([1,1,1,1,1,1]',[1,1,1,1,1,1]',[1,1,1,1,1,1]',[1,1,1,1,1,1]')

# Initial values for optimization
# init = vec(permutedims(hcat(MLE0, zeros(length(MLE0), 5))))
init = zeros(p_l + p_k + p_a + p_b + 4)

# MLE: No covariates
optmle = HRMLEMT(des_full, index_sel0, 10000, init)

MLE = optmle.minimizer

#=
****************************************************************************
Fitting the model without covariates: MAPs 
****************************************************************************
=#

# MAP: No covariates
optmap0 = optimize(mlog_postL0, [0.0,0.0,0.0,0.0], method=NelderMead(), iterations=1000)

MAP0 = optmap0.minimizer