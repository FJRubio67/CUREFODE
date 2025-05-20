
# Forced logistic ODE
flogis <- function(t, y, par) {
  # state variables
  CH <- y[1]

  
  # parameters
  lambda <- par[1]
  kappa <- par[2]
  eps <- par[3]

  
  # model equations
  dCH <-  lambda*CH*(1 - CH/kappa) + eps/(1+t)
  
  
  # result
  return( list(c(dCH)) )
  
}



lambda0 = 5
kappa0 = 1
eps0 = 0.01

tvec = seq(0,5,by = 0.01)


  params  <- c(lambda = lambda0, kappa = kappa0, eps = eps0)
  init      <- c(H = 0 )
  
  library(deSolve)
  
  out <- ode(init, tvec, flogis, params, method = "lsode")
  
  
  # Get the solution for H(t)
  H_values <- out[, 2]
  
  # Function to extract H'(t) after solving the ODE
  get_derivatives <- function(times, y, func, parms) {
    n <- length(times)
    derivs <- numeric(n)
    
    for (i in 1:n) {
      # Calculate derivative at each time point
      derivs[i] <- unlist(func(times[i], y[i], parms))
    }
    
    return(derivs)
  }
  
  # Extract the derivatives (h(t) = H'(t))
  h_values <- get_derivatives(tvec, H_values, flogis, params)
  

  
  plot(tvec,exp(-H_values), type = "l", lwd = 2, ylim = c(0,1))

  
  plot(tvec,h_values, type = "l", lwd = 2, ylim = c(0,1.5))
  