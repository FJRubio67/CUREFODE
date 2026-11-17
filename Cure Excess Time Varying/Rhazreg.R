library(HazReg)
library(matrixStats)

load("breastla.rda")


times = breastla$Time
status = breastla$Status
hp = breastla$mort.h

ind_bad = sort(c(which(is.na(hp)), which(times == 0), which(hp == 0), which(is.infinite(hp))))

times <- times[-ind_bad]
status <- status[-ind_bad]
hp <- hp[-ind_bad]


################################################################################
# No covariates
################################################################################

# PGWGH
OPTPGWGH <- GEHMLE(init = rep(0.0, 3), times = times, status = status, hp = hp, 
                     hstr = "baseline", dist = "PGW", method = "nlminb", 
                   control = list(iter.max = 10000))
MLEPGWGH <- exp(OPTPGWGH$OPT$par)


PGWGHhaz <- Vectorize(function(t) hpgw(t, MLEPGWGH[1], MLEPGWGH[2], MLEPGWGH[3]) ) 

curve(PGWGHhaz,1e-6, max(times), xlab = "Time (years)", ylab = "Baseline Hazard", main = "",
      cex.axis = 1.5, cex.lab = 1.5, lwd = 2, ylim = c(0,0.05), n = 1000)


net_surv <- Vectorize(function(t){
  out <- mean( exp( - chpgw(t, MLEPGWGH[1], MLEPGWGH[2], MLEPGWGH[3])  )  )
  return(out)
})


# Comparison
curve(net_surv,0.001,max(times), type = "l", col = "black", lwd = 2, lty = 1, ylim = c(0,1),
      xlab = "Time (years)", ylab = "Net Survival", main = "",
      cex.axis = 1.5, cex.lab = 1.5)

