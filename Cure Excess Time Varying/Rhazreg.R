rm(list=ls())

# Required packages
library(cuRe)
library(HazReg)
library(matrixStats)

# Data preparation
data(colonDC)
head(colonDC)
colonDC$bhaz <- general.haz(time = "FU",
                            rmap = list(age = "agedays", sex = "sex", year = "dx"),
                            data = colonDC, ratetable = survexp.dk)

# Variables
times = colonDC$FUyear
status = colonDC$status
hp = colonDC$bhaz
stage = as.numeric(colonDC$stage)
age = colonDC$age
sex = as.numeric(colonDC$sex)
ind_miss_stage = which(colonDC$stage=="Unknown")

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
      cex.axis = 1.5, cex.lab = 1.5, lwd = 2, ylim = c(0,1), n = 2000)


net_surv <- Vectorize(function(t){
  out <- mean( exp( - chpgw(t, MLEPGWGH[1], MLEPGWGH[2], MLEPGWGH[3])  )  )
  return(out)
})


# Comparison
curve(net_surv,0.001,max(times), type = "l", col = "black", lwd = 2, lty = 1, ylim = c(0,1),
      xlab = "Time (years)", ylab = "Net Survival", main = "",
      cex.axis = 1.5, cex.lab = 1.5)


################################################################################
# Prepare data for Julia


library(cuRe); library(rstpm2)
data(colonDC, package = "cuRe")
data(popmort, package = "rstpm2")

d <- colonDC
d$sexn   <- ifelse(d$sex == "male", 1, 2)
d$a_age  <- pmin(floor(d$age + d$FUyear), max(popmort$age))       # attained age
d$a_year <- as.numeric(format(d$dx + d$FU, "%Y"))                    # calendar year at exit
d$a_year <- pmin(pmax(d$a_year, min(popmort$year)), max(popmort$year))

d <- merge(d, popmort[, c("sex", "age", "year", "rate")],
           by.x = c("sexn", "a_age", "a_year"), by.y = c("sex", "age", "year"),
           all.x = TRUE, sort = FALSE)
names(d)[names(d) == "rate"] <- "mort_h"

summary(d$mort_h)                       # check: no NAs, plausible yearly rates
write.csv(d, "colonDC_hp.csv", row.names = FALSE)
