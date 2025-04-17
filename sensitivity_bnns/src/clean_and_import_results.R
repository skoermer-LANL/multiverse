# === SETUP ===
# Auto-detect project root if in RStudio
if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
  setwd("../")  # Move to project root
}

library(dplyr)
library(readr)
library(stringr)

# === DEFINE OPTIONS ===
method_dgm_combos <- c(
  "kl_div_x1d", "kl_div_x2d",
  "alpha_renyi_x1d", "alpha_renyi_x2d"
)


combo <- method_dgm_combos[4]

cat("Loading:", combo, "\n")
file_path <- file.path("results", combo, "merged_results.csv")

if(combo == method_dgm_combos[1]){

# Read file
df <- read_csv(file_path, col_types = cols(.default = "c"))  # Read as characters for cleaning

attr(df,which = "spec") <- NULL
attr(df, which = "problems") <- NULL

df <- as.data.frame(df)

df <- apply(df,2,as.numeric)
df <- as.data.frame(df)

s <- sort(df$lhs_row, index.return = TRUE)$ix

df <- df[s,]

row.names(df) <- NULL

write_csv(df, file_path)
}else{
  
  df <- read.csv(file_path)
  
  s <- sort(df$lhs_row, index.return = TRUE)$ix
  
  df <- df[s,]
  
  row.names(df) <- NULL
  
  write.csv(df,file = file_path, row.names = FALSE)
  
  
}
