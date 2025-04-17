#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: run_tgp_analysis.R <method> <dgm> <response_col>")
}

method <- args[1]
dgm <- args[2]
response_col <- args[3]

suppressPackageStartupMessages({
  library(tgp)
  library(dplyr)
  library(readr)
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

combo <- paste(method, dgm, sep = "_")
param_names <- param_specs[[combo]]$names
param_ranges <- param_specs[[combo]]$ranges

# Load data
input_path <- file.path("..", "..", "results", paste0(method, "_", dgm), "merged_results.csv")
output_path <- file.path("..", "..", "results", paste0(method, "_", dgm), paste0("tgp_", response_col, ".RData"))
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



plotparams <- c(1,2)

ngrid <- 4
r1 <- param_specs[[combo]]$ranges[[plotparams[1]]]
r2 <- param_specs[[combo]]$ranges[[plotparams[2]]]

x1 <- seq(from = r1[1], to = r1[2], length.out = ngrid)
x2 <- seq(from = r2[1], to = r2[2], length.out = ngrid)

plotgrid <- expand.grid(x1,x2)
plotgridtemp <- data.frame(matrix(NA, ncol = ncol(X), nrow = nrow(plotgrid)))

for(j in 1:ncol(plotgridtemp)){
  m <- param_specs[[combo]]$ranges[[j]]
  plotgridtemp[,j] <- mean(m)
}

plotgridtemp[, plotparams[1]] <- plotgrid[, plotparams[1]]
plotgridtemp[, plotparams[2]] <- plotgrid[, plotparams[2]]
plotgrid <- plotgridtemp

n <- 15
EIlhs <- tgp::lhs(n = n, rect = rect)

EIfit <- btgp(X = X, Z = Y, XX = EIlhs,
              BTE = c(20, 100, 1),
              pred.n = FALSE,
              improv = TRUE,
              krige = FALSE,
              #sens.p = sensparams,
              meanfn = "linear", trace = FALSE)

plotfit <- btgp(X = X, Z = Y, XX = plotgrid,
                BTE = c(20, 100, 1),
                pred.n = FALSE,
                improv = FALSE,
                krige = FALSE,
                #sens.p = sensparams,
                meanfn = "linear", trace = FALSE)

sensfit <- sens(X = X, Z = Y, model = btgp, nn.lhs = 10, BTE = c(20, 100, 1),
                pred.n = FALSE,
                improv = FALSE,
                krige = FALSE,
                #sens.p = sensparams,
                meanfn = "linear", trace = FALSE)



save(fit, X, Y, XX, sens_result, predict_result, file = output_path)
cat("Saved TGP output to:", output_path, "\n")
