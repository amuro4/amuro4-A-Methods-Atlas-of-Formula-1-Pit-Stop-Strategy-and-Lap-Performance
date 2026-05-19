# ============================================================
# MATH 4230 Capstone Project
# Script: scripts/08_regularization_ridge_lasso.R
# Purpose: Chapter 7 - Ridge and Lasso Regression
# ============================================================

# ----------------------------
# 1. Run setup
# ----------------------------

source("scripts/00_setup.R")

# ----------------------------
# 2. Organized output folders
# ----------------------------

chapter_name <- "ch07_regularization"

table_dir <- file.path("results/tables", chapter_name)
model_dir <- file.path("results/models", chapter_name)
figure_dir <- file.path("figures", chapter_name)

dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# 3. Helper functions
# ----------------------------

safe_ggsave <- function(plot_object, filename, width = 8, height = 5, dpi = 300) {
  main_path <- file.path(figure_dir, filename)
  
  if (file.exists(main_path)) {
    try(unlink(main_path, force = TRUE), silent = TRUE)
  }
  
  ggsave(
    filename = main_path,
    plot = plot_object,
    width = width,
    height = height,
    dpi = dpi,
    device = "png"
  )
  
  cat("Saved figure:", main_path, "\n")
}

rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

mae <- function(actual, predicted) {
  mean(abs(actual - predicted), na.rm = TRUE)
}

test_r2 <- function(actual, predicted) {
  1 - sum((actual - predicted)^2, na.rm = TRUE) /
    sum((actual - mean(actual, na.rm = TRUE))^2, na.rm = TRUE)
}

relevel_if_present <- function(x, ref_level) {
  x <- as.factor(x)
  
  if (ref_level %in% levels(x)) {
    x <- relevel(x, ref = ref_level)
  }
  
  return(x)
}

coef_to_table <- function(cv_fit, s_value, model_name, lambda_label) {
  coef_matrix <- as.matrix(coef(cv_fit, s = s_value))
  
  tibble(
    term = rownames(coef_matrix),
    coefficient = as.numeric(coef_matrix[, 1])
  ) %>%
    mutate(
      model = model_name,
      lambda_type = lambda_label,
      abs_coefficient = abs(coefficient),
      nonzero = coefficient != 0
    ) %>%
    arrange(desc(abs_coefficient))
}

get_original_variable <- function(term) {
  numeric_terms <- c(
    "lap_number",
    "tyre_life",
    "normalized_tyre_life",
    "race_progress",
    "stint",
    "position",
    "position_change"
  )
  
  case_when(
    term == "(Intercept)" ~ "(Intercept)",
    term %in% numeric_terms ~ term,
    str_starts(term, "compound") ~ "compound",
    str_starts(term, "driver") ~ "driver",
    str_starts(term, "race") ~ "race",
    str_starts(term, "year_fct") ~ "year",
    TRUE ~ "other"
  )
}

# ----------------------------
# 4. Load train and test data
# ----------------------------

train_path <- "data/processed/f1_train.csv"
test_path <- "data/processed/f1_test.csv"

if (!file.exists(train_path)) {
  stop("Training data not found. Run scripts/02_split_data.R first.")
}

if (!file.exists(test_path)) {
  stop("Testing data not found. Run scripts/02_split_data.R first.")
}

f1_train <- read_csv(train_path, show_col_types = FALSE)
f1_test <- read_csv(test_path, show_col_types = FALSE)

cat("Training data dimensions:\n")
print(dim(f1_train))

cat("\nTesting data dimensions:\n")
print(dim(f1_test))

# Lambda is chosen using cross-validation inside the training data.
# The 2025 test set is only used after fitting to compare RMSE.

# ----------------------------
# 5. Prepare modeling data
# ----------------------------

regularization_data <- bind_rows(
  f1_train %>% mutate(split = "train"),
  f1_test %>% mutate(split = "test")
) %>%
  mutate(
    compound = relevel_if_present(compound, "MEDIUM"),
    driver = as.factor(driver),
    race = as.factor(race),
    year_fct = as.factor(year)
  ) %>%
  select(
    split,
    lap_time_s,
    lap_number,
    tyre_life,
    normalized_tyre_life,
    race_progress,
    stint,
    position,
    position_change,
    compound,
    driver,
    race,
    year_fct
  ) %>%
  drop_na()

cat("\nRows after regularization prep:\n")
print(table(regularization_data$split))

# ----------------------------
# 6. Build model matrix
# ----------------------------

# Leakage note:
# lap_time_delta and cumulative_degradation are excluded because they are
# engineered from lap-time behavior and could leak information into a
# lap-time prediction model.
#
# pit_stop and pit_next_lap are also excluded from this regression model.

matrix_formula <- ~
  lap_number +
  tyre_life +
  normalized_tyre_life +
  race_progress +
  stint +
  position +
  position_change +
  compound +
  driver +
  race +
  year_fct

x_all <- model.matrix(
  matrix_formula,
  data = regularization_data
)

# Remove intercept because glmnet fits its own intercept.
x_all <- x_all[, colnames(x_all) != "(Intercept)", drop = FALSE]

train_index <- regularization_data$split == "train"
test_index <- regularization_data$split == "test"

x_train <- x_all[train_index, , drop = FALSE]
x_test <- x_all[test_index, , drop = FALSE]

y_train <- regularization_data$lap_time_s[train_index]
y_test <- regularization_data$lap_time_s[test_index]

# Remove columns with zero variance in training.
# This handles factor levels that only appear in 2025.

zero_variance_cols <- apply(x_train, 2, sd, na.rm = TRUE) == 0

removed_predictors <- tibble(
  term = colnames(x_train)[zero_variance_cols],
  reason = "Zero variance in training data"
)

if (sum(zero_variance_cols) > 0) {
  x_train <- x_train[, !zero_variance_cols, drop = FALSE]
  x_test <- x_test[, !zero_variance_cols, drop = FALSE]
}

write_csv(
  removed_predictors,
  file.path(table_dir, "ch07_removed_zero_variance_predictors.csv")
)

model_matrix_summary <- tibble(
  item = c(
    "Training rows",
    "Testing rows",
    "Predictors before zero-variance removal",
    "Predictors removed",
    "Predictors used"
  ),
  value = c(
    nrow(x_all[train_index, , drop = FALSE]),
    nrow(x_all[test_index, , drop = FALSE]),
    ncol(x_all),
    sum(zero_variance_cols),
    ncol(x_train)
  )
)

write_csv(
  model_matrix_summary,
  file.path(table_dir, "ch07_model_matrix_summary.csv")
)

cat("\nModel matrix summary:\n")
print(model_matrix_summary)

# ----------------------------
# 7. Fit ridge and lasso using CV
# ----------------------------

set.seed(4230)

ridge_cv <- cv.glmnet(
  x = x_train,
  y = y_train,
  alpha = 0,
  nfolds = 5,
  type.measure = "mse",
  standardize = TRUE
)

set.seed(4230)

lasso_cv <- cv.glmnet(
  x = x_train,
  y = y_train,
  alpha = 1,
  nfolds = 5,
  type.measure = "mse",
  standardize = TRUE
)

cat("\nRidge and lasso CV finished.\n")

# ----------------------------
# 8. Lambda summary
# ----------------------------

lambda_summary <- tibble(
  model = c("Ridge", "Ridge", "Lasso", "Lasso"),
  lambda_type = c("lambda.min", "lambda.1se", "lambda.min", "lambda.1se"),
  lambda = c(
    ridge_cv$lambda.min,
    ridge_cv$lambda.1se,
    lasso_cv$lambda.min,
    lasso_cv$lambda.1se
  )
) %>%
  mutate(lambda = round(lambda, 8))

write_csv(
  lambda_summary,
  file.path(table_dir, "ch07_lambda_summary.csv")
)

cat("\nLambda summary:\n")
print(lambda_summary)

# ----------------------------
# 9. Test RMSE comparison
# ----------------------------

ridge_pred_min <- as.numeric(
  predict(ridge_cv, newx = x_test, s = ridge_cv$lambda.min)
)

ridge_pred_1se <- as.numeric(
  predict(ridge_cv, newx = x_test, s = ridge_cv$lambda.1se)
)

lasso_pred_min <- as.numeric(
  predict(lasso_cv, newx = x_test, s = lasso_cv$lambda.min)
)

lasso_pred_1se <- as.numeric(
  predict(lasso_cv, newx = x_test, s = lasso_cv$lambda.1se)
)

mean_baseline_pred <- rep(mean(y_train, na.rm = TRUE), length(y_test))

test_rmse_comparison <- tibble(
  model = c("Lasso", "Ridge", "Ridge", "Lasso", "Mean Baseline"),
  lambda_type = c("lambda.min", "lambda.min", "lambda.1se", "lambda.1se", "none"),
  test_rmse = c(
    rmse(y_test, lasso_pred_min),
    rmse(y_test, ridge_pred_min),
    rmse(y_test, ridge_pred_1se),
    rmse(y_test, lasso_pred_1se),
    rmse(y_test, mean_baseline_pred)
  ),
  test_mae = c(
    mae(y_test, lasso_pred_min),
    mae(y_test, ridge_pred_min),
    mae(y_test, ridge_pred_1se),
    mae(y_test, lasso_pred_1se),
    mae(y_test, mean_baseline_pred)
  ),
  test_r_squared = c(
    test_r2(y_test, lasso_pred_min),
    test_r2(y_test, ridge_pred_min),
    test_r2(y_test, ridge_pred_1se),
    test_r2(y_test, lasso_pred_1se),
    test_r2(y_test, mean_baseline_pred)
  )
) %>%
  mutate(
    across(c(test_rmse, test_mae, test_r_squared), ~ round(.x, 4))
  ) %>%
  arrange(test_rmse)

write_csv(
  test_rmse_comparison,
  file.path(table_dir, "ch07_test_rmse_comparison.csv")
)

cat("\nTest RMSE comparison:\n")
print(test_rmse_comparison)

best_test_model <- test_rmse_comparison %>%
  slice(1)

# ----------------------------
# 10. Save predictions
# ----------------------------

test_predictions <- tibble(
  actual_lap_time_s = y_test,
  ridge_lambda_min = ridge_pred_min,
  ridge_lambda_1se = ridge_pred_1se,
  lasso_lambda_min = lasso_pred_min,
  lasso_lambda_1se = lasso_pred_1se,
  mean_baseline = mean_baseline_pred
)

write_csv(
  test_predictions,
  file.path(table_dir, "ch07_test_predictions.csv")
)

# ----------------------------
# 11. Coefficient tables
# ----------------------------

ridge_coef_min <- coef_to_table(
  cv_fit = ridge_cv,
  s_value = ridge_cv$lambda.min,
  model_name = "Ridge",
  lambda_label = "lambda.min"
)

ridge_coef_1se <- coef_to_table(
  cv_fit = ridge_cv,
  s_value = ridge_cv$lambda.1se,
  model_name = "Ridge",
  lambda_label = "lambda.1se"
)

lasso_coef_min <- coef_to_table(
  cv_fit = lasso_cv,
  s_value = lasso_cv$lambda.min,
  model_name = "Lasso",
  lambda_label = "lambda.min"
)

lasso_coef_1se <- coef_to_table(
  cv_fit = lasso_cv,
  s_value = lasso_cv$lambda.1se,
  model_name = "Lasso",
  lambda_label = "lambda.1se"
)

all_coefficients <- bind_rows(
  ridge_coef_min,
  ridge_coef_1se,
  lasso_coef_min,
  lasso_coef_1se
) %>%
  mutate(
    original_variable = get_original_variable(term),
    coefficient = round(coefficient, 6),
    abs_coefficient = round(abs_coefficient, 6)
  )

write_csv(
  all_coefficients,
  file.path(table_dir, "ch07_all_regularized_coefficients.csv")
)

write_csv(
  ridge_coef_min %>%
    mutate(
      original_variable = get_original_variable(term),
      coefficient = round(coefficient, 6),
      abs_coefficient = round(abs_coefficient, 6)
    ),
  file.path(table_dir, "ch07_ridge_coefficients_lambda_min.csv")
)

write_csv(
  ridge_coef_1se %>%
    mutate(
      original_variable = get_original_variable(term),
      coefficient = round(coefficient, 6),
      abs_coefficient = round(abs_coefficient, 6)
    ),
  file.path(table_dir, "ch07_ridge_coefficients_lambda_1se.csv")
)

write_csv(
  lasso_coef_min %>%
    mutate(
      original_variable = get_original_variable(term),
      coefficient = round(coefficient, 6),
      abs_coefficient = round(abs_coefficient, 6)
    ),
  file.path(table_dir, "ch07_lasso_coefficients_lambda_min.csv")
)

write_csv(
  lasso_coef_1se %>%
    mutate(
      original_variable = get_original_variable(term),
      coefficient = round(coefficient, 6),
      abs_coefficient = round(abs_coefficient, 6)
    ),
  file.path(table_dir, "ch07_lasso_coefficients_lambda_1se.csv")
)

# ----------------------------
# 12. Variables lasso kept
# ----------------------------

lasso_kept_min <- lasso_coef_min %>%
  filter(
    term != "(Intercept)",
    coefficient != 0
  ) %>%
  mutate(
    original_variable = get_original_variable(term),
    coefficient = round(coefficient, 6),
    abs_coefficient = round(abs_coefficient, 6)
  ) %>%
  arrange(desc(abs_coefficient))

lasso_kept_1se <- lasso_coef_1se %>%
  filter(
    term != "(Intercept)",
    coefficient != 0
  ) %>%
  mutate(
    original_variable = get_original_variable(term),
    coefficient = round(coefficient, 6),
    abs_coefficient = round(abs_coefficient, 6)
  ) %>%
  arrange(desc(abs_coefficient))

lasso_kept_summary <- bind_rows(
  lasso_kept_min %>%
    mutate(lambda_type = "lambda.min"),
  lasso_kept_1se %>%
    mutate(lambda_type = "lambda.1se")
) %>%
  group_by(lambda_type, original_variable) %>%
  summarise(
    kept_terms = n(),
    largest_abs_coefficient = max(abs_coefficient, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(lambda_type, desc(kept_terms), original_variable)

write_csv(
  lasso_kept_min,
  file.path(table_dir, "ch07_lasso_kept_variables_lambda_min.csv")
)

write_csv(
  lasso_kept_1se,
  file.path(table_dir, "ch07_lasso_kept_variables_lambda_1se.csv")
)

write_csv(
  lasso_kept_summary,
  file.path(table_dir, "ch07_lasso_kept_variable_summary.csv")
)

cat("\nNumber of variables kept by lasso:\n")
cat("lambda.min:", nrow(lasso_kept_min), "\n")
cat("lambda.1se:", nrow(lasso_kept_1se), "\n")

cat("\nTop lasso-kept variables using lambda.min:\n")
print(head(lasso_kept_min, 20))

# ----------------------------
# 13. Lambda CV plot
# ----------------------------

ridge_cv_curve <- tibble(
  model = "Ridge",
  lambda = ridge_cv$lambda,
  log_lambda = log(ridge_cv$lambda),
  cv_mse = ridge_cv$cvm,
  cv_sd = ridge_cv$cvsd,
  cv_rmse = sqrt(ridge_cv$cvm)
)

lasso_cv_curve <- tibble(
  model = "Lasso",
  lambda = lasso_cv$lambda,
  log_lambda = log(lasso_cv$lambda),
  cv_mse = lasso_cv$cvm,
  cv_sd = lasso_cv$cvsd,
  cv_rmse = sqrt(lasso_cv$cvm)
)

cv_curve <- bind_rows(
  ridge_cv_curve,
  lasso_cv_curve
)

write_csv(
  cv_curve,
  file.path(table_dir, "ch07_lambda_cv_curve_data.csv")
)

lambda_lines <- tibble(
  model = c("Ridge", "Ridge", "Lasso", "Lasso"),
  lambda_type = c("lambda.min", "lambda.1se", "lambda.min", "lambda.1se"),
  lambda = c(
    ridge_cv$lambda.min,
    ridge_cv$lambda.1se,
    lasso_cv$lambda.min,
    lasso_cv$lambda.1se
  ),
  log_lambda = log(lambda)
)

p_lambda_cv <- ggplot(cv_curve, aes(x = log_lambda, y = cv_rmse)) +
  geom_line(color = f1_red, linewidth = 1) +
  geom_point(color = f1_red, size = 1.2, alpha = 0.70) +
  geom_vline(
    data = lambda_lines,
    aes(xintercept = log_lambda, linetype = lambda_type),
    color = "gray35",
    linewidth = 0.8
  ) +
  facet_wrap(~ model, scales = "free_x") +
  labs(
    title = "Cross-Validation Curves for Ridge and Lasso",
    subtitle = "Lambda was chosen using 5-fold cross-validation on the training data.",
    x = "log(lambda)",
    y = "Cross-Validated RMSE",
    linetype = "Lambda Type"
  )

safe_ggsave(
  plot_object = p_lambda_cv,
  filename = "ch07_fig01_lambda_cv_plot.png",
  width = 9,
  height = 5
)

# ----------------------------
# 14. Test RMSE plot
# ----------------------------

plot_rmse_comparison <- test_rmse_comparison %>%
  filter(model != "Mean Baseline")

p_test_rmse <- ggplot(
  plot_rmse_comparison,
  aes(x = reorder(paste(model, lambda_type), test_rmse), y = test_rmse, fill = model)
) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = round(test_rmse, 2)),
    hjust = -0.10,
    size = 3.8
  ) +
  coord_flip() +
  labs(
    title = "Test RMSE Comparison for Regularized Regression",
    subtitle = "Lambda was selected by training CV, then models were evaluated on the 2025 test set.",
    x = "Model",
    y = "Test RMSE",
    fill = "Model"
  ) +
  scale_fill_manual(
    values = c(
      "Ridge" = f1_gray,
      "Lasso" = f1_red
    )
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)))

safe_ggsave(
  plot_object = p_test_rmse,
  filename = "ch07_fig02_test_rmse_comparison.png",
  width = 8,
  height = 5
)

# ----------------------------
# 15. Lasso kept coefficients plot
# ----------------------------

# Since lambda.1se kept 0 predictors, lambda.min is used for the
# lasso variable-selection plot.

if (nrow(lasso_kept_min) > 0) {
  
  lasso_plot_data <- lasso_kept_min %>%
    slice_max(abs_coefficient, n = min(20, nrow(lasso_kept_min))) %>%
    mutate(term = fct_reorder(term, coefficient))
  
  p_lasso_coef <- ggplot(
    lasso_plot_data,
    aes(x = term, y = coefficient)
  ) +
    geom_col(fill = f1_red) +
    coord_flip() +
    labs(
      title = "Top Lasso Coefficients Kept",
      subtitle = "Shown for lambda.min because lambda.1se shrank all predictors to zero.",
      x = "Term",
      y = "Coefficient"
    )
  
  safe_ggsave(
    plot_object = p_lasso_coef,
    filename = "ch07_fig03_lasso_kept_coefficients.png",
    width = 9,
    height = 6
  )
}

# ----------------------------
# 16. Save lightweight model record
# ----------------------------

regularization_record <- list(
  chapter_name = "Chapter 7 - Ridge and Lasso Regression",
  matrix_formula = deparse(matrix_formula),
  model_matrix_summary = model_matrix_summary,
  lambda_summary = lambda_summary,
  test_rmse_comparison = test_rmse_comparison,
  best_test_model = best_test_model,
  lasso_kept_summary = lasso_kept_summary,
  lasso_kept_lambda_min = lasso_kept_min,
  lasso_kept_lambda_1se = lasso_kept_1se,
  note = "Lambda was chosen by training cross-validation. Test RMSE uses the 2025 test set only after model fitting."
)

saveRDS(
  regularization_record,
  file.path(model_dir, "ch07_regularization_record.rds")
)

# ----------------------------
# 17. Interpretation notes
# ----------------------------

best_lasso_row <- test_rmse_comparison %>%
  filter(model == "Lasso") %>%
  arrange(test_rmse) %>%
  slice(1)

best_ridge_row <- test_rmse_comparison %>%
  filter(model == "Ridge") %>%
  arrange(test_rmse) %>%
  slice(1)

baseline_row <- test_rmse_comparison %>%
  filter(model == "Mean Baseline") %>%
  slice(1)

lasso_min_count <- nrow(lasso_kept_min)
lasso_1se_count <- nrow(lasso_kept_1se)

top_lasso_terms_min <- lasso_kept_min %>%
  slice_head(n = min(8, nrow(lasso_kept_min))) %>%
  pull(term) %>%
  paste(collapse = ", ")

if (top_lasso_terms_min == "") {
  top_lasso_terms_min <- "No nonzero terms besides the intercept"
}

lasso_group_summary_text <- lasso_kept_summary %>%
  filter(lambda_type == "lambda.min") %>%
  arrange(desc(kept_terms), original_variable) %>%
  mutate(text = paste0(original_variable, " (", kept_terms, ")")) %>%
  pull(text) %>%
  paste(collapse = ", ")

if (lasso_group_summary_text == "") {
  lasso_group_summary_text <- "No predictor groups kept"
}

interpretation <- tibble(
  item = c(
    "Main method",
    "Model matrix",
    "Leakage decision",
    "Ridge lambda choice",
    "Lasso lambda choice",
    "Best ridge test RMSE",
    "Best lasso test RMSE",
    "Best overall test RMSE",
    "Mean baseline comparison",
    "Lasso lambda.1se result",
    "Lasso variables kept at lambda.min",
    "Top lasso lambda.min terms",
    "Main takeaway",
    "Test set note"
  ),
  note = c(
    "Ridge and lasso regression were used as regularized versions of linear regression for lap_time_s.",
    paste0(
      "The final model matrix used ",
      ncol(x_train),
      " predictors after removing ",
      sum(zero_variance_cols),
      " zero-variance training columns."
    ),
    "lap_time_delta, cumulative_degradation, pit_stop, and pit_next_lap were excluded to avoid leakage or target-related information.",
    paste0(
      "Ridge lambda.min was ",
      round(ridge_cv$lambda.min, 8),
      " and ridge lambda.1se was ",
      round(ridge_cv$lambda.1se, 8),
      "."
    ),
    paste0(
      "Lasso lambda.min was ",
      round(lasso_cv$lambda.min, 8),
      " and lasso lambda.1se was ",
      round(lasso_cv$lambda.1se, 8),
      "."
    ),
    paste0(
      "The best ridge version used ",
      best_ridge_row$lambda_type,
      " with test RMSE = ",
      best_ridge_row$test_rmse,
      "."
    ),
    paste0(
      "The best lasso version used ",
      best_lasso_row$lambda_type,
      " with test RMSE = ",
      best_lasso_row$test_rmse,
      "."
    ),
    paste0(
      "The best overall regularized model was ",
      best_test_model$model,
      " using ",
      best_test_model$lambda_type,
      " with test RMSE = ",
      best_test_model$test_rmse,
      ", test MAE = ",
      best_test_model$test_mae,
      ", and test R-squared = ",
      best_test_model$test_r_squared,
      "."
    ),
    paste0(
      "The mean baseline test RMSE was ",
      baseline_row$test_rmse,
      ", so the lambda.min regularized models improved over predicting the training mean."
    ),
    paste0(
      "Lasso lambda.1se kept ",
      lasso_1se_count,
      " nonzero predictors, so it was too heavily regularized for reporting variable importance."
    ),
    paste0(
      "Using lambda.min, lasso kept ",
      lasso_min_count,
      " nonzero predictors. By group, it kept: ",
      lasso_group_summary_text,
      "."
    ),
    top_lasso_terms_min,
    "Regularization helped the linear model handle many dummy variables and correlated predictors. The lambda.min versions predicted better than the lambda.1se versions, while lasso also gave a variable-selection view.",
    "The 2025 test set was used only after lambda selection to compare test RMSE. It was not used to choose lambda."
  )
)

write_csv(
  interpretation,
  file.path(table_dir, "ch07_regularization_interpretation_notes.csv")
)

report_notes <- c(
  "Chapter 7 Report Notes",
  "",
  "This chapter used ridge and lasso regression for lap-time prediction. Both methods are regularized versions of linear regression.",
  "",
  paste0(
    "The model matrix included numeric lap predictors and dummy variables for compound, driver, race, and year. After removing zero-variance training columns, the final model matrix used ",
    ncol(x_train),
    " predictors."
  ),
  "",
  "The variables lap_time_delta, cumulative_degradation, pit_stop, and pit_next_lap were not used. These were excluded to avoid leakage or target-related information.",
  "",
  paste0(
    "Ridge used cross-validation to choose lambda. The ridge lambda.min was ",
    round(ridge_cv$lambda.min, 8),
    ", and ridge lambda.1se was ",
    round(ridge_cv$lambda.1se, 8),
    "."
  ),
  "",
  paste0(
    "Lasso used cross-validation to choose lambda. The lasso lambda.min was ",
    round(lasso_cv$lambda.min, 8),
    ", and lasso lambda.1se was ",
    round(lasso_cv$lambda.1se, 8),
    "."
  ),
  "",
  paste0(
    "The best ridge version used ",
    best_ridge_row$lambda_type,
    " and had test RMSE = ",
    best_ridge_row$test_rmse,
    ". The best lasso version used ",
    best_lasso_row$lambda_type,
    " and had test RMSE = ",
    best_lasso_row$test_rmse,
    "."
  ),
  "",
  paste0(
    "The best overall regularized model was ",
    best_test_model$model,
    " using ",
    best_test_model$lambda_type,
    ", with test RMSE = ",
    best_test_model$test_rmse,
    ", test MAE = ",
    best_test_model$test_mae,
    ", and test R-squared = ",
    best_test_model$test_r_squared,
    "."
  ),
  "",
  paste0(
    "The mean baseline test RMSE was ",
    baseline_row$test_rmse,
    ". This means the lambda.min regularized models improved over simply predicting the training mean."
  ),
  "",
  paste0(
    "Using lambda.1se, lasso kept ",
    lasso_1se_count,
    " nonzero predictors, so lambda.1se was too aggressive for the variable-selection discussion. Using lambda.min, lasso kept ",
    lasso_min_count,
    " nonzero predictors. The strongest kept terms included: ",
    top_lasso_terms_min,
    "."
  ),
  "",
  paste0(
    "By original variable group, lasso lambda.min kept: ",
    lasso_group_summary_text,
    "."
  ),
  "",
  "Real-world decision: Ridge is useful when the goal is stable prediction with many related predictors. Lasso is useful when the goal is a smaller model because it can shrink some coefficients exactly to zero. In this run, lasso lambda.min gave the best test RMSE, but ridge lambda.min was very close. Both are still linear models, so they may not capture all nonlinear race behavior."
)

writeLines(
  report_notes,
  file.path(table_dir, "ch07_report_notes.txt")
)

cat("\nInterpretation notes:\n")
print(interpretation)

# ----------------------------
# 18. Final confirmation
# ----------------------------

cat("\n08_regularization_ridge_lasso.R ran successfully.\n")
cat("Chapter 7 tables saved to: ", table_dir, "\n", sep = "")
cat("Chapter 7 figures saved to: ", figure_dir, "\n", sep = "")
cat("Chapter 7 model record saved to: ", model_dir, "\n", sep = "")
cat("Report notes saved to: ", file.path(table_dir, "ch07_report_notes.txt"), "\n", sep = "")
cat("Lambda was selected using training CV.\n")
cat("The 2025 test set was used only after lambda selection for RMSE comparison.\n")