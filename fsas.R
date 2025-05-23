rm(list = ls())

source("routinesC.R")



lambda0 = 5
kappa0 = 1
delta0 = 2
eps0 = 0.005

tvec = seq(0,10,by = 0.01)


params  <- c(lambda = lambda0, kappa = kappa0, delta = delta0, eps = eps0)
init  <- c(H = 0)

library(deSolve)

out <- ode(init, tvec, fsas, params, method = "lsode")


# Get the solution for H(t)
H_values <- out[, 2]


# Extract the derivatives (h(t) = H'(t))
h_values <- get_derivatives(tvec, H_values, fsas, params)



plot(tvec,exp(-H_values), type = "l", lwd = 2, ylim = c(0,1))


plot(tvec,h_values, type = "l", lwd = 2, ylim = c(0,2))

min(h_values)
