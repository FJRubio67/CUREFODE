# Required packages

library(haven)
library(HazReg)
library(PTCMGH)
library(haven)
library(dplyr)
library(survival)

# Read the raw data set
df <- read_sas("all_finalb.sas7bdat")
head(df)

# Data
df_raw <- read_sas("all_finalb.sas7bdat")

df <- df_raw %>%
  # Restrict to eligible patients only
  filter(elig == 2) %>%
  mutate(
    # --- Outcome variables ---
    os_time   = survmos,
    os_event  = survstat,       # 0=alive, 1=dead  
    dfs_time  = dfsmos,
    dfs_event = dfsstat,        # 0=no event, 1=event  
    
    # --- Treatment (primary exposures) ---
    agent  = factor(agent,  levels = c(0,1), labels = c("CA", "T")),
    length = factor(length, levels = c(0,1), labels = c("4 cycles", "6 cycles")),
    
    # --- Stratification / baseline covariates ---
    menopause = factor(stra1, levels = c(1,2), labels = c("Pre", "Post")),
    receptor  = factor(stra2, levels = c(1,2), labels = c("Pos/Unk", "Neg")),
    her2      = factor(stra3, levels = c(1,2,3), labels = c("Positive","Negative","Unknown")),
    
    # --- Tumour characteristics ---
    er_status   = factor(OH003, levels = c(1,2), labels = c("Negative","Positive")),
    pgr_status  = factor(OH004, levels = c(1,2), labels = c("Negative","Positive")),
    grade       = factor(OH005, levels = c(1,2,3), labels = c("Low","Intermediate","High")),
    tumor_size  = factor(tsize, levels = c(1,2,3), labels = c("<2cm","2-5cm",">5cm")),
    num_pos_nodes = num_pos_nodes,   # keep as-is or convert to numeric (note ">3" category)
    
    # --- Demographics ---
    age_cat = factor(agecat, levels = 1:6,
                     labels = c("20-29","30-39","40-49","50-59","60-69","70+")),
    race    = factor(RACE_ID, levels = c(1,3,4,5,99),
                     labels = c("White","Black","Asian","Other","Unknown")),
    ethnicity = factor(ETHNIC_ID, levels = c(1,2,9),
                       labels = c("Hispanic","Non-Hispanic","Unknown")),
    
    # --- Surgery / staging ---
    surgery_type = factor(OH028, levels = c(1,2), labels = c("Lumpectomy","Mastectomy")),
    sentinel_bx  = factor(OH032, levels = c(1,2), labels = c("No","Yes")),
    axillary_diss = factor(OH037, levels = c(1,2), labels = c("No","Yes")),
    
    # --- Amendment (important! 6-cycle arms were later closed) ---
    preamend = factor(preamend, levels = c(0,1), labels = c("Post-amend","Pre-amend"))
  )

dim(df)


# Kaplan-Meier estimator for the overall survival times
km <- survfit(Surv(df$os_time, df$os_event) ~ 1)
plot(km$time, km$surv, type = "l", col = "black", lwd = 2, lty = 1, 
     ylim = c(0,1), xlab = "Time", ylab = "Survival")


# Complete cases for variables of interest
dfc <- data.frame(agent = df$agent, length = df$length, menopause = df$menopause,
                  receptor = df$receptor, tumor_size = df$tumor_size, grade = df$grade,
                  num_pos_nodes = as.numeric(df$num_pos_nodes), age_cat = df$age_cat,
                  indrx = df$indrx, her2 = df$her2,
                  os_time = df$os_time, os_event = df$os_event)

dfc <- dfc[complete.cases(dfc),]
dim(dfc)


# # Cox model
# cox_os <- coxph(
#   Surv(os_time, os_event) ~
#     factor(agent) + factor(length) + factor(menopause) +
#     factor(receptor) + factor(her2) + factor(tumor_size) +
#     factor(grade),
#   data = dfc
# )
# summary(cox_os)

################################################################################
# Parametric analysis using PTCMGH
################################################################################

# Build the design matrix from the Cox model formula
X <- model.matrix(
  ~ factor(agent) + factor(length) + factor(menopause) +
    factor(receptor) + factor(her2) + factor(tumor_size) +
    factor(grade),
  data = dfc
)

# View the first few rows
head(X)
dim(X)


# Model fitting
OPT <- PTCMMLE(init = c(0,0,0),
               times = dfc$os_time,
               status = dfc$os_event,
               hstr = "baseline",
               dist = "LogNormal",
               des_theta = NULL,
               des_t = NULL,
               des_h = NULL,
               des = NULL,
               method = "nlminb", 
               maxit = 10000) 

# MLE
MLE <- c(OPT$OPT$par[1],exp(OPT$OPT$par[2]),exp(OPT$OPT$par[3]))

# Fitted survival function 
spt <- Vectorize(function(t) exp(- MLE[3]*(1-exp(-chlnorm(t,MLE[1],MLE[2])))) )

plot(km$time, km$surv, type = "l", col = "black", lwd = 2, lty = 1, 
     ylim = c(0,1), xlab = "Time", ylab = "Survival")
curve(spt,0,150, col = "red", add= T) 
legend("topright", legend = c("KM","Baseline"), col = c("black","red"), 
       lwd = c(2,2))



# Model fitting
OPT_PH <- PTCMMLE(init = c(0,0,0,0,0,0,0),
                  times = times,
                  status = status,
                  hstr = "PH",
                  dist = "LogNormal",
                  des_theta = des0,
                  des_t = NULL,
                  des_h = NULL,
                  des = des0[,-1],
                  method = "nlminb", 
                  maxit = 10000) 

# MLE
MLE_PH <- c(OPT_PH$OPT$par[1],exp(OPT_PH$OPT$par[2]),OPT_PH$OPT$par[-c(1:2)])

# Population survival function 
spt_ph <- Vectorize(function(t){
  
  theta_i <- as.vector(exp(des0 %*% MLE_PH[3:5]))
  F_i <- 1 - exp(-chlnorm(t,MLE_PH[1],MLE_PH[2])*exp(des0[,-1]%*%MLE_PH[6:7]))
  survs <- exp(-theta_i*F_i)
  return(mean(survs))
}) 

plot(km$time, km$surv, type = "l", col = "black", lwd = 2, lty = 1, 
     ylim = c(0,1), xlab = "Time", ylab = "Survival")
curve(spt_ph,0,6, col = "red", add= T)  
legend("topright", legend = c("KM","PH"), col = c("black","red"), 
       lwd = c(2,2))

