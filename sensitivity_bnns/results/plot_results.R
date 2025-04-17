### To do:  Check if num_steps, num_weights, and initial_samples is rounded within the sensitivity analysis.  It is not rounded for EI
### Annotate optimal point
### Group similar parameters within a plot, whether {optimizer stuff, prior parameters}
### 

### Get plots of the fits 

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tgp)

reg_types <- c("kl_div", "alpha_renyi")
data_gens <- c("x1d", "x2d")

result_files <- c("tgp_interval_score_sens.RData", "tgp_interval_score_improv_new.RData")

folders <- expand.grid(reg_types, data_gens)
folders <- paste(folders$Var1,folders$Var2, sep = "_")

results <- list(sens = list(), EI = list())

for(i in folders){
  
  load(paste0(i,"/", result_files[1]))
  results$sens[[i]] <- sensfit
  
  load(paste0(i,"/", result_files[2]))
  results$EI[[i]] <- EIfit
  
}

for(i in folders){
  
  plot(results$sens[[i]], layout = "sens", main = i)
  plot(results$sens[[i]], layout = "sens", maineff = t(1:7))
  
}

XXmins <- qmins<- list()

for(i in folders){
  m <- results$sens[[i]]$sens$ZZ.mean
  q2 <- results$sens[[i]]$sens$ZZ.q2
  q <- m
  
  XXmins[[i]] <- diag(results$sens[[i]]$sens$Xgrid[apply(q,2,which.min),])
  qmins[[i]] <- apply(q,2,min)
  
}

### Max expected improvement locations

maxEI <- list()



for(i in folders){
  
  maxEI[[i]] <- results$EI[[i]]$XX[sort(results$EI[[i]]$improv$rank, index.return = TRUE)$ix,][1:10,]
  row.names(maxEI[[i]]) <- NULL
  
}

## combine minimum of main effect and max EI

newpts <- list()

for(i in folders){
  
  newpts[[i]] <- rbind(maxEI[[i]], XXmins[[i]])
  
  newpts[[i]][c("num_steps", "num_weights", "initial_samples")] <- round(newpts[[i]][c("num_steps", "num_weights", "initial_samples")])
  
  write.csv(newpts[[i]], file = paste0(i,"/", i, "_", "extra.csv"), row.names = FALSE)
}

####### Poster plots

reg_type <- c("kl_div", "alpha_renyi")[2]
data_gen <- c("x1d", "x2d")[2]

sens_i2 <- 2 ## Sensitivity index, index
sens_type <- c("S", "T")[sens_i2] ## equivalent to c("First order", "Total")
ylab1 <- c("First Order Sensitivity Index", "Total Sensitivity Index")[sens_i2]

reg_dat <- paste0(reg_type, "_", data_gen)

layout(matrix(c(1:2,3,3), ncol = 2, byrow = TRUE), heights = c(6,1.5))


mar_use <- c(4.5, 4.5, 1 , 1.5)


kl_fctrs_use <-  c("log_kl_multiplier", 
                   "log_sigma", 
                   "num_steps", 
                   "initial_samples", 
                   "log_lr"
)

a_fctrs_use <- c("alpha", 
                 "log_sigma", 
                 "num_steps",
                 "initial_samples", 
                 "log_lr")
ufctrs <- unique(c(kl_fctrs_use, a_fctrs_use))

fctr_lookup <- data.frame(matrix(NA, nrow = length(ufctrs), ncol = 4))
pchs <- c(16,4,5,17,8,16,4,5,17,8)

names(fctr_lookup) <- c("fctr", "col","pch","label")
fctr_lookup$fctr <- ufctrs
fctr_lookup$col <- hcl.colors(n = length(ufctrs))
fctr_lookup$pch <- pchs[1:length(ufctrs)]
fctr_lookup$label <- c(as.expression(bquote("log"[10] * "(" * gamma * ")")), as.expression(bquote("log"[10] * "(" * sigma[0] * ")")), as.expression("Optimizer\n   Steps"), as.expression("Samples for\n Integration"), as.expression(bquote("log"[10] * "(LR)")), as.expression(bquote(alpha)))
#fctr_lookup$label <- c("Loss Weight\nFactor","Prior\nVariance", "Optimizer\nSteps", "Samples for\n Integration", "Learning\nRate", "alpha")



metric <- c("interval_score_calculate", "interval_score_assess", "coverage_rate", 
            "rmse")[1]

if(reg_type == "kl_div"){
fctrs <- c("log_kl_multiplier", "log_sigma", "num_steps", "num_weights", 
           "initial_samples", "log_lr", "prior_mu")
}else if(reg_type == "alpha_renyi"){
  fctrs <- c("alpha", "log_sigma", "num_steps", "num_weights", "initial_samples", 
               "log_lr", "prior_mu")
}

fctr <- "log_sigma"

first_order_stats <- data.frame(matrix(NA, ncol = length(fctrs), nrow = 2))
names(first_order_stats) <- fctrs
row.names(first_order_stats) <- c("mean", "var")

sens_fits <- results$sens[[reg_dat]]

first_order_stats[1,] <- apply(sens_fits$sens[[sens_type]],2,mean)
first_order_stats[2,] <- apply(sens_fits$sens[[sens_type]],2,var)

first_order <- sens_fits$sens[[sens_type]]


fctr_cols <- hcl.colors(n = length(fctrs), palette = "Zissou1")

if(reg_type == "kl_div"){
  fctrs_plot <- kl_fctrs_use
}else if(reg_type == "alpha_renyi"){
  fctrs_plot <- a_fctrs_use
}

fctrs_use <- fctrs_plot

first_order <- data.frame(first_order)
names(first_order) <- fctrs

#fctrs_plot <- fctrs

# Function to dynamically get jitter bounds
get_jitter_bounds <- function(data_matrix, jitter_amount = 0.2) {
  num_vars <- ncol(data_matrix)
  jittered_positions <- list()
  bounds <- matrix(NA, ncol = num_vars, nrow = 2)  # Store min/max for each variable
  
  for (i in 1:num_vars) {
    jitter_vals <- jitter(rep(i, nrow(data_matrix)), amount = jitter_amount)
    jittered_positions[[i]] <- jitter_vals
    bounds[, i] <- range(jitter_vals)  # Min and Max x-position for jittered points
  }
  
  return(list(jittered_positions = jittered_positions, bounds = bounds))
}

data_jitter <- get_jitter_bounds(first_order[fctrs_plot])$bounds


par(mar = mar_use)

#custom_labels <- c(as.expression(bquote("log"[10] * "(" * gamma * ")")), as.expression(bquote("log"[10] * "(" * sigma * ")")), "Optimizer\nSteps", "Layer\nWeights", "Monte-Carlo\nSamples", bquote("log"[10] * "(LR)"), bquote(mu["0"]))[fctrs %in% fctrs_plot]#c("A", "B", "C", "D", "E", "F", "G")

stripchart(first_order[fctrs_plot], 
           vertical = TRUE, 
           method = "jitter", 
           jitter = 0.2, 
           pch = 16, col = "gray85", 
           main = "",#bquote("p(" * theta * "|y) = " * "arg" * min("KL[" * q * "(" * theta * ")|p(" * theta * "|y)]", "q      ")),
           ylab = ylab1, xlab = "",
           xaxt = "n", ylim = c(0, range(first_order[fctrs_plot])[2]))#c(0,0.4))#,
#ylim = ylim)

#axis(1, at = 1:ncol(first_order_KL[fctrs_plot]), labels = custom_labels)  # las=2 rotates labels
#axis(1, at = 1:ncol(first_order_KL[fctrs_plot]), labels = fctr_lookup$label[match(kl_fctrs_use, fctr_lookup$fctr)])  # las=2 rotates labels
tickPositions <- 1:6
# First, draw the axis without labels.
axis(1, at = tickPositions, labels = FALSE)

# Now, add the labels manually with mtext.
# We loop over the tick positions, and for positions 3 and 4 we use a larger 'line' value.


for (i in seq_along(tickPositions)) {
  # Set extra offset for tick labels 3 and 4
  # extraLine <- if (i %in% c(3, 4)) 2 else 1
  extraline <- 1
  mtext(fctr_lookup$label[match(fctrs_use, fctr_lookup$fctr)][i], side = 1, at = tickPositions[i], line = 1 + extraline, cex = 0.8)
}


for(i in 1:length(fctrs_plot)){
  mean_use <- first_order_stats[[fctrs[fctrs %in% fctrs_plot[i]]]][1]
  #lines(data_jitter[,i],rep(mean_use, times = 2), col = "black", lwd = 3, lty = 3)
  lines(data_jitter[,i],rep(mean_use, times = 2), col = fctr_lookup$col[fctr_lookup$fctr == fctrs_plot[i]], lwd = 2, lty = 1)
  #points(mean(data_jitter[,i]), mean_use, pch = 4, cex = 1.6, col ="black", lwd = 2)
  points(mean(data_jitter[,i]), mean_use, pch = fctr_lookup$pch[fctr_lookup$fctr == fctrs_plot[i]], cex = 1.5, lwd = 1.6, col =fctr_lookup$col[fctr_lookup$fctr == fctrs_plot[i]])
}


################################################
######## main effects Sensitivity
################################################

#layout(matrix(c(1, 2, 3, 3), nrow = 2, byrow = TRUE), heights = c(4, 1))


maineff <- sens_fits$sens

xgrid <- seq(from = 0, to = 1, length.out = 100)
xcgrid <- c(0,xgrid[seq(from = 0, to = length(xgrid), length.out = 10)])


ufctrs <- fctrs_use

# fctr_lookup <- data.frame(matrix(NA, nrow = length(ufctrs), ncol = 4))
# pchs <- c(16,4,5,17,8,16,4,5,17,8)
# 
# names(fctr_lookup) <- c("fctr", "col","pch","label")
# fctr_lookup$fctr <- ufctrs
# fctr_lookup$col <- hcl.colors(n = length(ufctrs))
# fctr_lookup$pch <- pchs[1:length(ufctrs)]
# fctr_lookup$label <- c(NA, as.expression(bquote("log"[10] * "(" * sigma[0] * ")")), as.expression("Optimizer Steps"), as.expression("Samples for Integration  "), as.expression(bquote("log"[10] * "(LR)")), as.expression(bquote(alpha)))
# 
# fctr_lookup$label[1] <- as.expression(bquote("log"[10] * "(" * gamma * ")"))

#fctr_lookup$label <- c("Loss Weight Factor","Prior Variance", "Optimizer Steps", "Samples for Integration", "Learning Rate", "alpha")

mean_use <- (maineff$ZZ.mean[,fctrs %in% fctrs_use , drop = FALSE] + 0.5)*(range(sens_fits$Z)[2] - range(sens_fits$Z)[1])



par(mar = mar_use)

plot(xgrid, mean_use[,1], type = "l", col = fctr_lookup$col[fctr_lookup$fctr == fctrs_use[1]], xlim = c(0,1), ylim = range(mean_use),
     xlab = bquote("Tuning Parameter Value, Rescaled to [0,1]"), ylab = "Main Effect on Interval Score", lwd = 1.6)

points(xcgrid, mean_use[xgrid %in% xcgrid,1], pch = fctr_lookup$pch[fctr_lookup$fctr == fctrs_use[1]], col = fctr_lookup$col[fctr_lookup$fctr == fctrs_use[1]])

if(ncol(mean_use) > 1){
  for(i in 2:ncol(mean_use)){
    lines(xgrid, mean_use[,i], col = fctr_lookup$col[fctr_lookup$fctr == fctrs_use[i]], lwd = 1.6)
    points(xcgrid, mean_use[xgrid %in% xcgrid,i], pch = fctr_lookup$pch[fctr_lookup$fctr == fctrs_use[i]], col = fctr_lookup$col[fctr_lookup$fctr == fctrs_use[i]])
  }
}


par(mar = c(0, 0, 0, 0))
plot.new()

legend("center", legend = fctr_lookup$label[1:5], col = fctr_lookup$col[1:5], pch = fctr_lookup$pch[1:5], lty = 1, ncol = 3,
       bty = "n", lwd = 1.4, cex = 1.3)




