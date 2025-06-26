#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: run_tgp_sens.R <method> <dgm> <response_col>")
}

secondOrder <- TRUE
tgpsens <- FALSE


method <- args[1]
dgm <- args[2]
response_col <- args[3]

suppressPackageStartupMessages({
  library(tgp)
  library(sensitivity)
  library(yaml)
})

param_specs <- list(
  "kl_div_x1d" = list(
    names = c("log_kl_multiplier", "log_sigma", "num_steps", "num_weights", "initial_samples", "log_lr", "prior_mu"),
    ranges = list(c(-1, 1), c(-0.5, 0.5), c(2000, 20000), c(2, 100), c(1, 25), c(-3.3, -0.3), c(-2, 2))
  ),
  "kl_div_x2d" = list(
    names = c("log_kl_multiplier", "log_sigma", "num_steps", "num_weights", "initial_samples", "log_lr", "prior_mu"),
    ranges = list(c(-1, 1), c(-0.5, 0.5), c(2000, 20000), c(2, 100), c(1, 25), c(-3.3, -0.3), c(-2, 2))
  ),
  "alpha_renyi_x1d" = list(
    names = c("alpha", "log_sigma", "num_steps", "num_weights", "initial_samples", "log_lr", "prior_mu"),
    ranges = list(c(0, 1), c(-0.5, 0.5), c(2000, 20000), c(2, 100), c(1, 25), c(-3.3, -0.3), c(-2, 2))
  ),
  "alpha_renyi_x2d" = list(
    names = c("alpha", "log_sigma", "num_steps", "num_weights", "initial_samples", "log_lr", "prior_mu"),
    ranges = list(c(0, 1), c(-0.5, 0.5), c(2000, 20000), c(2, 100), c(1, 25), c(-3.3, -0.3), c(-2, 2))
  )
)

param_specs_raw <- yaml::read_yaml("param_specs_v1.yaml")

# Convert to flat key format like before
param_specs <- list()
for (m in names(param_specs_raw)) {
  for (d in names(param_specs_raw[[m]])) {
    key <- paste(m, d, sep = "_")
    param_specs[[key]] <- param_specs_raw[[m]][[d]]
  }
}




combo <- paste(method, dgm, sep = "_")
param_names <- param_specs[[combo]]$names
param_ranges <- param_specs[[combo]]$ranges

# Load data
input_path <- file.path("..", "..", "results", paste0(method, "_", dgm), "merged_results.csv")
output_path <- file.path("..", "..", "results", paste0(method, "_", dgm), paste0("tgp_", response_col, "_sens.RData"))
tmpdir <- file.path("..", "tgp_tmp", paste0(method, "_", dgm, "_", response_col))
dir.create(tmpdir, recursive = TRUE, showWarnings = FALSE)
setwd(tmpdir)

data <- read.csv(input_path)

lhs_path <- file.path("..", "..", "lhs", paste0(method, "_", dgm, "_lhs.csv"))

lhs_df <- read.csv(lhs_path)

X <- lhs_df
Y <- data[[response_col]]

# Predictive grid
set.seed(123)

rect <- do.call("rbind",param_specs[[combo]]$ranges)

BTE <- c(2000, 100000, 10)

if(tgpsens){
sensfit <- sens(X = X, Z = Y, model = btgp, nn.lhs = 10, BTE = BTE,
                rect = rect,
                pred.n = FALSE,
                improv = FALSE,
                krige = FALSE,
                meanfn = "linear", trace = FALSE)

save(sensfit, file = output_path)

}
## Additional calculation for 2nd order sens


if(secondOrder){

## Note we did not achieve good results from this method after lengthy computation
## this numerical method returned some negative indices, but we left the option here

n <- 7000



X1 <- X1nat <- data.frame(lhs(n, rect = rect))
X2 <- X2nat <- data.frame(lhs(n, rect = rect))



for(j in 1:ncol(X2nat)){

  r <- rect[j,]
  m <- r[1]
  r <- r[2] - r[1]

  X1[,j] <- (X1nat[,j] - m)/r
  X2[,j] <- (X2nat[,j] - m)/r


}

sobol_design <- sobol(
  model = NULL,
  X1 = X1,
  X2 = X2,
  order = 2,
  nboot = 0
)

XX <- unname(as.matrix(sobol_design$X))

for(j in 1:ncol(XX)){

  r <- rect[j,]
  m <- r[1]
  r <- r[2] - r[1]

  XX[,j] <- XX[,j] * r + m

}

Y_allfit <- btgp(X = X, Z = Y, XX = XX, BTE = BTE,
                pred.n = FALSE,
                krige = TRUE,
                meanfn = "linear", trace = FALSE)



Y_all <- as.numeric(drop(Y_allfit$ZZ.mean))

Y_all <- (Y_all - mean(Y_all))/sd(Y_all)

sobol_result <- tell(sobol_design, Y_all)

save(sobol_result, file = output_path)

}

if(secondOrder & tgpsens){
  save(sensfit,sobol_result,  file = output_path)
}
