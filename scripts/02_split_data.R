# ============================================================
# MATH 4230 Capstone Project
# Script: scripts/02_split_data.R
# Purpose: Create year-based train/test split
# ============================================================

# ----------------------------
# 1. Run setup
# ----------------------------

source("scripts/00_setup.R")

# ----------------------------
# 2. Load cleaned modeling data
# ----------------------------

clean_path <- "data/processed/f1_modeling_data.csv"

if (!file.exists(clean_path)) {
  stop("Cleaned data file not found. Run scripts/01_clean_data.R first.")
}

f1 <- read_csv(clean_path, show_col_types = FALSE)

# ----------------------------
# 3. Fix variable types after reading CSV
# ----------------------------

f1 <- f1 %>%
  mutate(
    race = as.factor(race),
    driver = as.factor(driver),
    compound = as.factor(compound),
    race_id = as.factor(race_id),
    pit_stop = as.factor(pit_stop),
    pit_next_lap = as.factor(pit_next_lap)
  )

# ----------------------------
# 4. Create train/test split
# ----------------------------

f1_train <- f1 %>%
  filter(year %in% c(2022, 2023, 2024))

f1_test <- f1 %>%
  filter(year == 2025)

# ----------------------------
# 5. Check row counts
# ----------------------------

cat("Training data dimensions:\n")
print(dim(f1_train))

cat("\nTesting data dimensions:\n")
print(dim(f1_test))

# ----------------------------
# 6. Check class balance
# ----------------------------

cat("\nTraining pit_next_lap counts:\n")
print(table(f1_train$pit_next_lap, useNA = "ifany"))

cat("\nTesting pit_next_lap counts:\n")
print(table(f1_test$pit_next_lap, useNA = "ifany"))

# ----------------------------
# 7. Check number of races
# ----------------------------

cat("\nNumber of race-year IDs in training data:\n")
print(length(unique(f1_train$race_id)))

cat("\nNumber of race-year IDs in testing data:\n")
print(length(unique(f1_test$race_id)))

cat("\nTraining years:\n")
print(sort(unique(f1_train$year)))

cat("\nTesting years:\n")
print(sort(unique(f1_test$year)))

# ----------------------------
# 8. Stop if split looks wrong
# ----------------------------

if (nrow(f1_train) == 0) {
  stop("Training data has 0 rows. Check the year values.")
}

if (nrow(f1_test) == 0) {
  stop("Testing data has 0 rows. Check whether 2025 exists in the dataset.")
}

if (length(unique(f1_train$pit_next_lap)) < 2) {
  stop("Training data does not contain both pit_next_lap classes.")
}

if (length(unique(f1_test$pit_next_lap)) < 2) {
  stop("Testing data does not contain both pit_next_lap classes.")
}

if (length(unique(f1_train$race_id)) < 2) {
  stop("Training data does not contain multiple races.")
}

if (length(unique(f1_test$race_id)) < 2) {
  stop("Testing data does not contain multiple races.")
}

# ----------------------------
# 9. Save train/test files
# ----------------------------

write_csv(f1_train, "data/processed/f1_train.csv")
write_csv(f1_test, "data/processed/f1_test.csv")

# ----------------------------
# 10. Split explanation
# ----------------------------

cat("\nWhy this split makes sense:\n")
cat("This project uses a year-based split instead of a random row split.\n")
cat("Formula 1 lap data is connected within races, so randomly splitting laps could leak race context across training and testing.\n")
cat("The model is trained on earlier seasons, 2022-2024, and tested on the later 2025 season.\n")
cat("The 2025 test set should stay untouched until final model evaluation.\n")

cat("\n02_split_data.R ran successfully.\n")
cat("Training data saved to: data/processed/f1_train.csv\n")
cat("Testing data saved to: data/processed/f1_test.csv\n")