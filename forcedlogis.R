
# Forced logistic ODE
flogis <- function(t, y, par) {
  # state variables
  CH <- y[1]
  
  # parameters
  lambda <- par[1]
  kappa <- par[2]
  eps <- par[3]

  
  # model equations
  dCH <-  lambda*CH*(1 - CH/kappa) + eps

  
  # result
  return( list(c(dCH)) )
  
}



lambda0 = 5
kappa0 = 1
eps0 = 0.01

tvec = seq(0,5,by = 0.1)


  params  <- c(lambda = lambda0, kappa = kappa0, eps = eps0)
  init      <- c(H = 0 )
  
  library(deSolve)
  
  out <- ode(init, tvec, flogis, params, method = "lsode")
  

  plot(tvec,exp(-out[,2]), type = "l", lwd = 2, ylim = c(0,1))
