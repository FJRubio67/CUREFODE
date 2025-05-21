# Forced logistic ODE
flogis <- function(t, y, par) {
  # state variables
  CH <- y[1]
  
  
  # parameters
  lambda <- par[1]
  kappa <- par[2]
  eps <- par[3]
  
  
  # model equations
  # dCH <-  lambda*CH*(1 - CH/kappa) + eps/(1+t)^2
  
  dCH0 <-  lambda*CH*(1 - CH/kappa) + eps/(1+t)^2
  dCH <- ifelse(dCH0 > 0, dCH0, 1e-6)
  
  # result
  return( list(c(dCH)) )
  
}

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


#--------------------------------------------------------------------------------------------------
# Forced logistic ODE -log-likelihood function: Solver 
#--------------------------------------------------------------------------------------------------

log_likFL <- function(par){
  # Numerical solution for the ODE
  params  <- c(lambda = exp(par[1]), kappa = exp(par[2]), eps = exp(par[3]))
  init  <- c(H = 0)
  times  <- c(0,survtimes) 
  out <- ode(init, times, flogis, params, method = "lsode")
  
  # Get the solution for H(t)
  H_values <- as.vector(out[, 2])
  
  # Extract the derivatives (h(t) = H'(t))
  h_values <- get_derivatives(times, H_values, flogis, params)
  
  # Terms in the log log likelihood function
  ll_haz <- sum(log(as.vector(h_values[status])))
  
  ll_chaz <- sum(as.vector(H_values))
  
  log_lik <- -ll_haz + ll_chaz
  
  return(log_lik)
}
