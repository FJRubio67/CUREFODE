rm(list=ls())

library(smcure)
library(deSolve)

source("routinesC.R")

data(e1684)

# Kaplan-Meier estimator for the survival times
km <- survfit(Surv(e1684$FAILTIME,e1684$FAILCENS) ~ 1)

plot(km$time, km$surv, type = "l", col = "black", lwd = 2, lty = 1, 
     ylim = c(0,1), xlab = "Time", ylab = "Survival")


survtimes <- as.vector(sort(e1684$FAILTIME))
status <- as.logical(e1684$FAILCENS)


log_likFL(c(0,0,-1))

OPT = nlminb(c(0,25,-1), log_likFL, control = list(iter.max = 10000))

      
tvec = seq(0,10,by = 0.01)


params  <- c(lambda = exp(OPT$par[1]), kappa = exp(OPT$par[2]), eps = exp(OPT$par[1]))
init  <- c(H = 0)


out <- ode(init, tvec, flogis, params, method = "lsode")


# Get the solution for H(t)
H_values <- out[, 2]


# Extract the derivatives (h(t) = H'(t))
h_values <- get_derivatives(tvec, H_values, flogis, params)



plot(tvec,exp(-H_values), type = "l", lwd = 2, ylim = c(0,1))


plot(tvec,h_values, type = "l", lwd = 2, ylim = c(0,1.5))



plot(km$time, km$surv, type = "l", col = "black", lwd = 2, lty = 1, 
     ylim = c(0,1), xlab = "Time", ylab = "Survival")
points(tvec,exp(-H_values), type = "l", lwd = 2, ylim = c(0,1),
       col = "blue")







tempf1 <- Vectorize(function(par) log_likFL(c(par,OPT$par[2],OPT$par[3])))

curve(tempf1,-3,-1)

tempf2 <- Vectorize(function(par) log_likFL(c(OPT$par[1],par,OPT$par[3])))

curve(tempf2,15,25)


tempf3 <- Vectorize(function(par) log_likFL(c(OPT$par[1],OPT$par[2],par)))

curve(tempf3,-1.5,0)

