# ============================================================
# MATH 4230 Capstone Project
# Script: scripts/00_setup.R
# Purpose: Load packages, set global options, and create folders
# ============================================================

# ----------------------------
# 1. Global settings
# ----------------------------

set.seed(4230)

options(scipen = 999)

# ----------------------------
# 2. Packages
# ----------------------------

packages <- c(
  "tidyverse",
  "janitor",
  "skimr",
  "broom",
  "knitr",
  "kableExtra",
  "caret",
  "rsample",
  "recipes",
  "glmnet",
  "car",
  "leaps",
  "tree",
  "randomForest",
  "gbm",
  "e1071",
  "class",
  "nnet",
  "pROC",
  "corrplot"
)

installed_packages <- rownames(installed.packages())

missing_packages <- packages[!(packages %in% installed_packages)]

if (length(missing_packages) > 0) {
  install.packages(missing_packages, dependencies = TRUE)
}

suppressPackageStartupMessages(
  lapply(packages, library, character.only = TRUE)
)

# ----------------------------
# 3. Main folder paths
# ----------------------------

dir_data_raw <- "data/raw"
dir_data_processed <- "data/processed"

dir_scripts <- "scripts"

dir_figures <- "figures"
dir_figures_eda <- "figures/eda"

dir_results <- "results"
dir_results_tables <- "results/tables"
dir_results_tables_eda <- "results/tables/eda"
dir_results_models <- "results/models"

dir_report <- "report"

folders <- c(
  dir_data_raw,
  dir_data_processed,
  dir_scripts,
  dir_figures,
  dir_figures_eda,
  dir_results,
  dir_results_tables,
  dir_results_tables_eda,
  dir_results_models,
  dir_report
)

for (folder in folders) {
  if (!dir.exists(folder)) {
    dir.create(folder, recursive = TRUE)
  }
}

# ----------------------------
# 4. Helper function for chapter folders
# ----------------------------

create_chapter_folders <- function(chapter_name) {
  dir.create(file.path("figures", chapter_name), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path("results/tables", chapter_name), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path("results/models", chapter_name), recursive = TRUE, showWarnings = FALSE)
}

# Example use inside future scripts:
# create_chapter_folders("ch04_mlr")

# ----------------------------
# 5. Plot theme
# ----------------------------

theme_capstone <- function() {
  theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", size = 18),
      plot.subtitle = element_text(size = 11, color = "gray35"),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(color = "gray25"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "gray90"),
      legend.position = "bottom",
      legend.title = element_text(face = "bold")
    )
}

theme_set(theme_capstone())

# ----------------------------
# 6. Project colors
# ----------------------------

f1_red <- "#C8102E"
f1_dark <- "gray35"
f1_gray <- "gray55"

# ----------------------------
# 7. Project targets
# ----------------------------

regression_target <- "lap_time_s"
classification_target <- "pit_next_lap"

# ----------------------------
# 8. Leakage notes
# ----------------------------

lap_time_leakage_vars <- c(
  "lap_time_delta",
  "cumulative_degradation",
  "pit_stop",
  "pit_next_lap"
)

pit_stop_leakage_vars <- c(
  "pit_stop"
)

# ----------------------------
# 9. Setup check
# ----------------------------

cat("00_setup.R ran successfully.\n")
cat("Packages loaded.\n")
cat("Folders checked/created.\n")
cat("Regression target:", regression_target, "\n")
cat("Classification target:", classification_target, "\n")