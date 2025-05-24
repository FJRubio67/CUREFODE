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
  
  force = eps*exp(-2*t)
  
  dCH0 <-  lambda*CH*(1 - CH/kappa) + force
  dCH <- as.numeric(ifelse(dCH0 > force, dCH0, force))
  
  # result
  return( list(c(dCH)) )
  
}


# Forced Gompertz ODE
fgomp <- function(t, y, par) {
  # state variables
  CH <- y[1]
  
  
  # parameters
  lambda <- par[1]
  kappa <- par[2]
  eps <- par[3]
  
  
  # model equations
  # dCH <-  lambda*CH*(1 - CH/kappa) + eps/(1+t)^2
  
  force = eps*exp(-2*t)
  
  dCH0 <-  lambda*CH*log(kappa/(CH+eps*exp(-2*t))) + force
  dCH <- as.numeric(ifelse(dCH0 > force, dCH0, force))
  
  # result
  return( list(c(dCH)) )
  
}


# Forced Richards ODE
frich <- function(t, y, par) {
  # state variables
  CH <- y[1]
  
  
  # parameters
  lambda <- par[1]
  kappa <- par[2]
  beta <- par[3]
  eps <- par[4]
  
  # model equations
  # dCH <-  lambda*CH*(1 - CH/kappa) + eps/(1+t)^2
  
  force = eps*exp(-3*t)
  
  dCH0 <-  lambda*CH*(1 - (CH/kappa)^beta) + force
  dCH <- as.numeric(ifelse(dCH0 > force, dCH0, force))
  
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




# Forced SAS ODE (in progress)
fsas <- function(t, y, par) {
  # state variables
  CH <- y[1]
  
  
  # parameters
  lambda <- par[1]
  kappa <- par[2]
  delta <- par[3]
  eps <- par[4]
  
  # model equations
  # dCH <-  lambda*CH*(1 - CH/kappa) + eps/(1+t)^2
  cons = sinh(delta*asinh(1))
  
  force = eps*exp(-3*t)
  
  dCH0 <-  lambda*CH*(1 - sinh(delta*asinh(CH/kappa))/cons ) + force
  dCH <- as.numeric(ifelse(dCH0 > force, dCH0, force))
  
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
