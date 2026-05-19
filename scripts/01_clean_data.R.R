# ============================================================
# MATH 4230 Capstone Project
# Script: scripts/01_clean_data.R
# Purpose: Clean raw F1 data and build modeling dataset
# ============================================================

# ----------------------------
# 1. Run setup
# ----------------------------

source("scripts/00_setup.R")

# ----------------------------
# 2. Read raw CSV
# ----------------------------

raw_files <- list.files(
  path = "data/raw",
  pattern = "\\.csv$",
  full.names = TRUE
)

if (length(raw_files) == 0) {
  stop("No CSV file found in data/raw. Put the Kaggle CSV file in data/raw first.")
}

raw_file <- raw_files[1]

cat("Reading raw file:\n")
cat(raw_file, "\n\n")

f1_raw <- read_csv(raw_file, show_col_types = FALSE)

# ----------------------------
# 3. Clean column names
# ----------------------------

f1_clean <- f1_raw %>%
  clean_names()

cat("Dimensions after reading data:\n")
print(dim(f1_clean))

cat("\nVariable names after clean_names():\n")
print(names(f1_clean))

# ----------------------------
# 4. Check missing values and duplicates
# ----------------------------

cat("\nMissing values before cleaning:\n")
print(colSums(is.na(f1_clean)))

cat("\nNumber of duplicate rows before cleaning:\n")
print(sum(duplicated(f1_clean)))

# ----------------------------
# 5. Check required variables
# ----------------------------

required_vars <- c(
  "driver",
  "lap_number",
  "compound",
  "stint",
  "tyre_life",
  "position",
  "lap_time_s",
  "race",
  "year",
  "lap_time_delta",
  "cumulative_degradation",
  "pit_stop",
  "pit_next_lap",
  "race_progress",
  "normalized_tyre_life",
  "position_change"
)

missing_vars <- setdiff(required_vars, names(f1_clean))

if (length(missing_vars) > 0) {
  cat("\nMissing required variables:\n")
  print(missing_vars)
  stop("Some required variables are missing. Check the raw CSV column names.")
}

# ----------------------------
# 6. Helper function for binary variables
# ----------------------------

to_yes_no <- function(x) {
  x <- as.character(x)
  
  case_when(
    x %in% c("1", "TRUE", "True", "true", "Yes", "yes", "Y", "y") ~ "Yes",
    x %in% c("0", "FALSE", "False", "false", "No", "no", "N", "n") ~ "No",
    TRUE ~ x
  )
}

# ----------------------------
# 7. Convert variable types
# ----------------------------

numeric_vars <- c(
  "year",
  "lap_number",
  "position",
  "lap_time_s",
  "stint",
  "tyre_life",
  "normalized_tyre_life",
  "lap_time_delta",
  "cumulative_degradation",
  "position_change",
  "race_progress"
)

factor_vars <- c(
  "race",
  "driver",
  "compound"
)

f1_clean <- f1_clean %>%
  mutate(
    across(all_of(numeric_vars), ~ as.numeric(.x)),
    across(all_of(factor_vars), ~ as.factor(.x)),
    
    pit_stop = factor(to_yes_no(pit_stop), levels = c("No", "Yes")),
    pit_next_lap = factor(to_yes_no(pit_next_lap), levels = c("No", "Yes")),
    
    race_id = paste(year, race, sep = "_"),
    race_id = as.factor(race_id),
    
    compound_encoded = as.numeric(compound)
  )

# ----------------------------
# 8. Remove duplicate and unusable rows
# ----------------------------

f1_clean <- f1_clean %>%
  distinct() %>%
  filter(
    !is.na(year),
    !is.na(race),
    !is.na(driver),
    !is.na(lap_number),
    !is.na(lap_time_s),
    !is.na(pit_next_lap)
  )

# ----------------------------
# 9. Final checks
# ----------------------------

cat("\nCleaned data dimensions:\n")
print(dim(f1_clean))

cat("\nCleaned variable names:\n")
print(names(f1_clean))

cat("\nMissing values after cleaning:\n")
print(colSums(is.na(f1_clean)))

cat("\nNumber of duplicate rows after cleaning:\n")
print(sum(duplicated(f1_clean)))

cat("\nPit next lap class balance:\n")
print(table(f1_clean$pit_next_lap, useNA = "ifany"))

cat("\nPit stop class balance:\n")
print(table(f1_clean$pit_stop, useNA = "ifany"))

cat("\nYears included:\n")
print(sort(unique(f1_clean$year)))

cat("\nNumber of race-year IDs:\n")
print(length(unique(f1_clean$race_id)))

cat("\nCompound levels:\n")
print(levels(f1_clean$compound))

# Important leakage note:
# Do not use pit_stop as a predictor when predicting pit_next_lap at first.
# pit_stop describes whether a pit stop happened on the current lap,
# so it may create leakage in the classification models.

# ----------------------------
# 10. Save cleaned modeling data
# ----------------------------

write_csv(f1_clean, "data/processed/f1_modeling_data.csv")

cat("\n01_clean_data.R ran successfully.\n")
cat("Cleaned modeling data saved to: data/processed/f1_modeling_data.csv\n")