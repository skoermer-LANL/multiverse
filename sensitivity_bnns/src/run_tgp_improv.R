#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: run_tgp_improv.R <method> <dgm> <response_col>")
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
output_path <- file.path("..", "..", "results", paste0(method, "_", dgm), paste0("tgp_", response_col, "_improv.RData"))
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

n <- 15
EIlhs <- tgp::lhs(n = n, rect = rect)

EIfit <- btgp(X = X, Z = Y, XX = EIlhs,
              BTE = c(20, 100, 1),
              pred.n = FALSE,
              improv = TRUE,
              krige = FALSE,
              meanfn = "linear", trace = FALSE)


save(EIfit, file = output_path)
