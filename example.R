rm(list=ls())

library(smcure)
library(deSolve)
library(HazReg)

source("routinesC.R")

data(e1684)

# Kaplan-Meier estimator for the survival times
km <- survfit(Surv(e1684$FAILTIME,e1684$FAILCENS) ~ 1)

plot(km$time, km$surv, type = "l", col = "black", lwd = 2, lty = 1, 
     ylim = c(0,1), xlab = "Time", ylab = "Survival")


survtimes <- as.vector(sort(e1684$FAILTIME))
status <- as.logical(e1684$FAILCENS)


log_likFL(c(0,0,-1))

OPT = nlminb(c(0,0,-1), log_likFL, control = list(iter.max = 10000))

      
tvec = seq(0,10,by = 0.01)


params  <- c(lambda = exp(OPT$par[1]), kappa = exp(OPT$par[2]), eps = exp(OPT$par[3]))
init  <- c(H = 0)


out <- ode(init, tvec, flogis, params, method = "lsode")


# Get the solution for H(t)
H_values <- out[, 2]


# Extract the derivatives (h(t) = H'(t))
h_values <- get_derivatives(tvec, H_values, flogis, params)

plot(tvec,H_values, type = "l", lwd = 2, ylim = c(0,10))


plot(tvec,exp(-H_values), type = "l", lwd = 2, ylim = c(0,1))


plot(tvec,h_values, type = "l", lwd = 2, ylim = c(0,1.5))



plot(km$time, km$surv, type = "l", col = "black", lwd = 2, lty = 1, 
     ylim = c(0,1), xlab = "Time", ylab = "Survival")
points(tvec,exp(-H_values), type = "l", lwd = 2, ylim = c(0,1),
       col = "blue")




OPTPGW <- GHMLE(c(0,0,0), survtimes, status, hstr = "baseline", dist = "PGW", maxit = 10000, method = "nlminb")


fspgw <- Vectorize(function(t) exp(-chpgw(t,exp(OPTPGW$OPT$par[1]),exp(OPTPGW$OPT$par[2]),exp(OPTPGW$OPT$par[3]))))


plot(km$time, km$surv, type = "l", col = "black", lwd = 2, lty = 1, 
     ylim = c(0,1), xlab = "Time", ylab = "Survival")
curve(fspgw,0,10, add = T, lwd = 2, ylim = c(0,1),
       col = "red")

curve(tempf1,-2,-1)

tempf2 <- Vectorize(function(par) log_likFL(c(OPT$par[1],par,OPT$par[3])))

curve(tempf2,1,5)


tempf3 <- Vectorize(function(par) log_likFL(c(OPT$par[1],OPT$par[2],par)))

curve(tempf3,-1.5,1.5)

