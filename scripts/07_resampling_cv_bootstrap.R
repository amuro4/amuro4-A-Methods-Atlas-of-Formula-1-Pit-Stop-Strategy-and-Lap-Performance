# ============================================================
# MATH 4230 Capstone Project
# Script: scripts/07_resampling_cv_bootstrap.R
# Purpose: Chapter 6 - Cross-Validation and Bootstrap
# ============================================================

# ----------------------------
# 1. Run setup
# ----------------------------

source("scripts/00_setup.R")

# ----------------------------
# 2. Organized output folders
# ----------------------------

chapter_name <- "ch06_resampling"

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

validation_r2 <- function(actual, predicted) {
  1 - sum((actual - predicted)^2, na.rm = TRUE) /
    sum((actual - mean(actual, na.rm = TRUE))^2, na.rm = TRUE)
}

safe_divide <- function(numerator, denominator) {
  ifelse(denominator == 0, NA_real_, numerator / denominator)
}

make_binary_target <- function(x) {
  x_chr <- tolower(as.character(x))
  
  case_when(
    x_chr %in% c("1", "yes", "y", "true", "pit", "pitted") ~ 1,
    x_chr %in% c("0", "no", "n", "false", "not pit", "not_pit") ~ 0,
    TRUE ~ NA_real_
  )
}

evaluate_classification <- function(actual, predicted_class, predicted_probability) {
  actual <- factor(actual, levels = c("No", "Yes"))
  predicted_class <- factor(predicted_class, levels = c("No", "Yes"))
  
  cm <- table(Actual = actual, Predicted = predicted_class)
  
  tn <- as.numeric(cm["No", "No"])
  fp <- as.numeric(cm["No", "Yes"])
  fn <- as.numeric(cm["Yes", "No"])
  tp <- as.numeric(cm["Yes", "Yes"])
  
  auc_value <- tryCatch(
    {
      as.numeric(
        pROC::auc(
          response = if_else(actual == "Yes", 1, 0),
          predictor = predicted_probability,
          quiet = TRUE
        )
      )
    },
    error = function(e) NA_real_
  )
  
  tibble(
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn),
    sensitivity = safe_divide(tp, tp + fn),
    specificity = safe_divide(tn, tn + fp),
    precision = safe_divide(tp, tp + fp),
    auc = auc_value,
    true_negative = tn,
    false_positive = fp,
    false_negative = fn,
    true_positive = tp
  )
}

relevel_if_present <- function(x, ref_level) {
  x <- as.factor(x)
  
  if (ref_level %in% levels(x)) {
    x <- relevel(x, ref = ref_level)
  }
  
  return(x)
}

# ----------------------------
# 4. Load training data only
# ----------------------------

train_path <- "data/processed/f1_train.csv"

if (!file.exists(train_path)) {
  stop("Training data not found. Run scripts/02_split_data.R first.")
}

f1_train <- read_csv(train_path, show_col_types = FALSE)

# Important:
# We do NOT use f1_test here.
# The 2025 test set should stay untouched until final model evaluation.

# ----------------------------
# 5. Prepare modeling data
# ----------------------------

resampling_data <- f1_train %>%
  mutate(
    pit_next_lap_num = make_binary_target(pit_next_lap),
    pit_next_lap_factor = factor(
      if_else(pit_next_lap_num == 1, "Yes", "No"),
      levels = c("No", "Yes")
    ),
    compound = relevel_if_present(compound, "MEDIUM"),
    year_fct = as.factor(year),
    race_id = as.factor(race_id)
  ) %>%
  select(
    lap_time_s,
    pit_next_lap_num,
    pit_next_lap_factor,
    lap_number,
    tyre_life,
    normalized_tyre_life,
    race_progress,
    stint,
    position,
    position_change,
    compound,
    year_fct,
    race_id
  ) %>%
  drop_na()

if (length(unique(resampling_data$pit_next_lap_num)) < 2) {
  stop("pit_next_lap does not contain both classes after cleaning.")
}

cat("Training rows used for Chapter 6 resampling:\n")
print(nrow(resampling_data))

cat("\nNumber of race-year IDs used for grouped CV:\n")
print(length(unique(resampling_data$race_id)))

cat("\nClass balance for pit_next_lap:\n")
print(table(resampling_data$pit_next_lap_factor))

# ----------------------------
# 6. Create grouped 5-fold CV split
# ----------------------------

# Since laps within the same race are connected, folds are assigned by race_id.
# This keeps all laps from the same race-year together in the same fold.

set.seed(4230)

k <- 5

race_folds <- resampling_data %>%
  distinct(race_id) %>%
  mutate(
    fold = sample(rep(1:k, length.out = n()))
  )

cv_data <- resampling_data %>%
  left_join(race_folds, by = "race_id")

fold_summary <- cv_data %>%
  group_by(fold) %>%
  summarise(
    rows = n(),
    race_year_ids = n_distinct(race_id),
    pit_next_lap_yes = sum(pit_next_lap_factor == "Yes"),
    pit_next_lap_no = sum(pit_next_lap_factor == "No"),
    pit_next_lap_yes_rate = mean(pit_next_lap_factor == "Yes"),
    .groups = "drop"
  ) %>%
  mutate(
    pit_next_lap_yes_rate = round(pit_next_lap_yes_rate, 4)
  )

write_csv(
  fold_summary,
  file.path(table_dir, "ch06_cv_fold_summary.csv")
)

cat("\nCV fold summary:\n")
print(fold_summary)

# ----------------------------
# 7. Define CV model formulas
# ----------------------------

# Regression model:
# This is a simple MLR model for lap_time_s.
# It uses the same general style as Chapter 4, but not the huge driver/race model.

reg_formula <- lap_time_s ~
  lap_number +
  tyre_life +
  normalized_tyre_life +
  race_progress +
  stint +
  position +
  position_change +
  compound +
  year_fct

# Classification model:
# This matches the Chapter 5 logistic setup.
# pit_stop is excluded to avoid leakage.

class_formula <- pit_next_lap_num ~
  tyre_life +
  race_progress +
  stint +
  position +
  position_change +
  compound +
  year_fct

# ----------------------------
# 8. K-fold CV for regression
# ----------------------------

regression_cv_by_fold <- tibble()

for (fold_id in 1:k) {
  
  train_fold <- cv_data %>%
    filter(fold != fold_id)
  
  valid_fold <- cv_data %>%
    filter(fold == fold_id)
  
  reg_fit <- lm(reg_formula, data = train_fold)
  
  reg_pred <- predict(reg_fit, newdata = valid_fold)
  
  fold_results <- tibble(
    fold = fold_id,
    model_type = "Regression",
    model = "MLR for lap_time_s",
    rmse = rmse(valid_fold$lap_time_s, reg_pred),
    mae = mae(valid_fold$lap_time_s, reg_pred),
    validation_r_squared = validation_r2(valid_fold$lap_time_s, reg_pred),
    validation_rows = nrow(valid_fold)
  )
  
  regression_cv_by_fold <- bind_rows(regression_cv_by_fold, fold_results)
}

regression_cv_by_fold <- regression_cv_by_fold %>%
  mutate(
    rmse_flag = if_else(
      rmse == max(rmse, na.rm = TRUE),
      "Highest RMSE fold",
      "Typical fold"
    ),
    across(c(rmse, mae, validation_r_squared), ~ round(.x, 4))
  )

write_csv(
  regression_cv_by_fold,
  file.path(table_dir, "ch06_regression_cv_by_fold.csv")
)

cat("\nRegression CV results by fold:\n")
print(regression_cv_by_fold)

# ----------------------------
# 9. K-fold CV for classification
# ----------------------------

classification_cv_by_fold <- tibble()
classification_cv_predictions <- tibble()

for (fold_id in 1:k) {
  
  train_fold <- cv_data %>%
    filter(fold != fold_id)
  
  valid_fold <- cv_data %>%
    filter(fold == fold_id)
  
  class_fit <- glm(
    formula = class_formula,
    data = train_fold,
    family = binomial
  )
  
  pred_prob <- predict(
    class_fit,
    newdata = valid_fold,
    type = "response"
  )
  
  pred_class <- factor(
    if_else(pred_prob >= 0.50, "Yes", "No"),
    levels = c("No", "Yes")
  )
  
  fold_metrics <- evaluate_classification(
    actual = valid_fold$pit_next_lap_factor,
    predicted_class = pred_class,
    predicted_probability = pred_prob
  ) %>%
    mutate(
      fold = fold_id,
      model_type = "Classification",
      model = "Logistic regression for pit_next_lap",
      validation_rows = nrow(valid_fold)
    ) %>%
    select(
      fold,
      model_type,
      model,
      validation_rows,
      accuracy,
      sensitivity,
      specificity,
      precision,
      auc,
      true_negative,
      false_positive,
      false_negative,
      true_positive
    )
  
  fold_predictions <- tibble(
    fold = fold_id,
    actual = valid_fold$pit_next_lap_factor,
    predicted_probability = pred_prob,
    predicted_class = pred_class
  )
  
  classification_cv_by_fold <- bind_rows(
    classification_cv_by_fold,
    fold_metrics
  )
  
  classification_cv_predictions <- bind_rows(
    classification_cv_predictions,
    fold_predictions
  )
}

classification_cv_by_fold <- classification_cv_by_fold %>%
  mutate(
    across(
      c(accuracy, sensitivity, specificity, precision, auc),
      ~ round(.x, 4)
    )
  )

write_csv(
  classification_cv_by_fold,
  file.path(table_dir, "ch06_classification_cv_by_fold.csv")
)

write_csv(
  classification_cv_predictions,
  file.path(table_dir, "ch06_classification_cv_predictions.csv")
)

cat("\nClassification CV results by fold:\n")
print(classification_cv_by_fold)

# ----------------------------
# 10. CV summary tables
# ----------------------------

regression_cv_summary <- tibble(
  model_type = "Regression",
  model = "MLR for lap_time_s",
  metric = c("RMSE", "MAE", "Validation R-squared"),
  unit = c("seconds", "seconds", "proportion"),
  mean_value = c(
    mean(regression_cv_by_fold$rmse, na.rm = TRUE),
    mean(regression_cv_by_fold$mae, na.rm = TRUE),
    mean(regression_cv_by_fold$validation_r_squared, na.rm = TRUE)
  ),
  sd_value = c(
    sd(regression_cv_by_fold$rmse, na.rm = TRUE),
    sd(regression_cv_by_fold$mae, na.rm = TRUE),
    sd(regression_cv_by_fold$validation_r_squared, na.rm = TRUE)
  ),
  median_value = c(
    median(regression_cv_by_fold$rmse, na.rm = TRUE),
    median(regression_cv_by_fold$mae, na.rm = TRUE),
    median(regression_cv_by_fold$validation_r_squared, na.rm = TRUE)
  ),
  min_value = c(
    min(regression_cv_by_fold$rmse, na.rm = TRUE),
    min(regression_cv_by_fold$mae, na.rm = TRUE),
    min(regression_cv_by_fold$validation_r_squared, na.rm = TRUE)
  ),
  max_value = c(
    max(regression_cv_by_fold$rmse, na.rm = TRUE),
    max(regression_cv_by_fold$mae, na.rm = TRUE),
    max(regression_cv_by_fold$validation_r_squared, na.rm = TRUE)
  )
) %>%
  mutate(
    across(
      c(mean_value, sd_value, median_value, min_value, max_value),
      ~ round(.x, 4)
    )
  )

classification_cv_summary <- tibble(
  model_type = "Classification",
  model = "Logistic regression for pit_next_lap",
  metric = c("Accuracy", "Sensitivity", "Specificity", "Precision", "AUC"),
  unit = "proportion",
  mean_value = c(
    mean(classification_cv_by_fold$accuracy, na.rm = TRUE),
    mean(classification_cv_by_fold$sensitivity, na.rm = TRUE),
    mean(classification_cv_by_fold$specificity, na.rm = TRUE),
    mean(classification_cv_by_fold$precision, na.rm = TRUE),
    mean(classification_cv_by_fold$auc, na.rm = TRUE)
  ),
  sd_value = c(
    sd(classification_cv_by_fold$accuracy, na.rm = TRUE),
    sd(classification_cv_by_fold$sensitivity, na.rm = TRUE),
    sd(classification_cv_by_fold$specificity, na.rm = TRUE),
    sd(classification_cv_by_fold$precision, na.rm = TRUE),
    sd(classification_cv_by_fold$auc, na.rm = TRUE)
  ),
  median_value = c(
    median(classification_cv_by_fold$accuracy, na.rm = TRUE),
    median(classification_cv_by_fold$sensitivity, na.rm = TRUE),
    median(classification_cv_by_fold$specificity, na.rm = TRUE),
    median(classification_cv_by_fold$precision, na.rm = TRUE),
    median(classification_cv_by_fold$auc, na.rm = TRUE)
  ),
  min_value = c(
    min(classification_cv_by_fold$accuracy, na.rm = TRUE),
    min(classification_cv_by_fold$sensitivity, na.rm = TRUE),
    min(classification_cv_by_fold$specificity, na.rm = TRUE),
    min(classification_cv_by_fold$precision, na.rm = TRUE),
    min(classification_cv_by_fold$auc, na.rm = TRUE)
  ),
  max_value = c(
    max(classification_cv_by_fold$accuracy, na.rm = TRUE),
    max(classification_cv_by_fold$sensitivity, na.rm = TRUE),
    max(classification_cv_by_fold$specificity, na.rm = TRUE),
    max(classification_cv_by_fold$precision, na.rm = TRUE),
    max(classification_cv_by_fold$auc, na.rm = TRUE)
  )
) %>%
  mutate(
    across(
      c(mean_value, sd_value, median_value, min_value, max_value),
      ~ round(.x, 4)
    )
  )

cv_summary <- bind_rows(
  regression_cv_summary,
  classification_cv_summary
)

write_csv(
  regression_cv_summary,
  file.path(table_dir, "ch06_regression_cv_summary.csv")
)

write_csv(
  classification_cv_summary,
  file.path(table_dir, "ch06_classification_cv_summary.csv")
)

write_csv(
  cv_summary,
  file.path(table_dir, "ch06_cv_summary.csv")
)

cat("\nRegression CV summary:\n")
print(regression_cv_summary)

cat("\nClassification CV summary:\n")
print(classification_cv_summary)

cat("\nCombined CV summary:\n")
print(cv_summary)

# ----------------------------
# 11. CV figures
# ----------------------------

# The old combined plot mixed seconds and proportions on one axis.
# These separate plots are clearer for the report.

regression_plot_data <- regression_cv_by_fold %>%
  select(fold, rmse, mae, validation_r_squared) %>%
  pivot_longer(
    cols = c(rmse, mae, validation_r_squared),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = case_when(
      metric == "rmse" ~ "RMSE",
      metric == "mae" ~ "MAE",
      metric == "validation_r_squared" ~ "Validation R-squared",
      TRUE ~ metric
    )
  )

p_regression_cv <- ggplot(
  regression_plot_data,
  aes(x = factor(fold), y = value)
) +
  geom_col(fill = f1_red, width = 0.7) +
  facet_wrap(~ metric, scales = "free_y") +
  labs(
    title = "Regression Cross-Validation by Fold",
    subtitle = "Fold 2 has a much larger RMSE, showing that RMSE is sensitive to very slow or unusual laps.",
    x = "CV Fold",
    y = "Value"
  )

safe_ggsave(
  plot_object = p_regression_cv,
  filename = "ch06_fig01_regression_cv_by_fold.png",
  width = 9,
  height = 5
)

classification_plot_data <- classification_cv_by_fold %>%
  select(fold, accuracy, sensitivity, specificity, precision, auc) %>%
  pivot_longer(
    cols = c(accuracy, sensitivity, specificity, precision, auc),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = case_when(
      metric == "accuracy" ~ "Accuracy",
      metric == "sensitivity" ~ "Sensitivity",
      metric == "specificity" ~ "Specificity",
      metric == "precision" ~ "Precision",
      metric == "auc" ~ "AUC",
      TRUE ~ metric
    )
  )

p_classification_cv <- ggplot(
  classification_plot_data,
  aes(x = factor(fold), y = value, group = metric, color = metric)
) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    limits = c(0, 1)
  ) +
  labs(
    title = "Classification Cross-Validation by Fold",
    subtitle = "AUC and specificity are strong, but sensitivity is much lower at the 0.50 threshold.",
    x = "CV Fold",
    y = "Metric Value",
    color = "Metric"
  )

safe_ggsave(
  plot_object = p_classification_cv,
  filename = "ch06_fig02_classification_cv_by_fold.png",
  width = 9,
  height = 5
)

# ----------------------------
# 12. Bootstrap slope of tyre_life in SLR
# ----------------------------

# Bootstrap model:
# lap_time_s = beta_0 + beta_1 * tyre_life + error
#
# The statistic being bootstrapped is beta_1, the slope for tyre_life.

set.seed(4230)

bootstrap_data <- resampling_data %>%
  select(lap_time_s, tyre_life) %>%
  drop_na()

original_slr <- lm(lap_time_s ~ tyre_life, data = bootstrap_data)

original_slope <- unname(coef(original_slr)["tyre_life"])

B <- 1000
n_boot <- nrow(bootstrap_data)

boot_slopes <- map_dfr(
  1:B,
  function(b) {
    boot_index <- sample.int(
      n = n_boot,
      size = n_boot,
      replace = TRUE
    )
    
    boot_sample <- bootstrap_data[boot_index, ]
    
    boot_fit <- lm(lap_time_s ~ tyre_life, data = boot_sample)
    
    tibble(
      bootstrap_sample = b,
      tyre_life_slope = unname(coef(boot_fit)["tyre_life"])
    )
  }
)

bootstrap_ci <- tibble(
  estimate = "Slope of tyre_life in SLR",
  original_slope = original_slope,
  bootstrap_mean = mean(boot_slopes$tyre_life_slope, na.rm = TRUE),
  bootstrap_se = sd(boot_slopes$tyre_life_slope, na.rm = TRUE),
  ci_lower_95 = quantile(boot_slopes$tyre_life_slope, 0.025, na.rm = TRUE),
  ci_upper_95 = quantile(boot_slopes$tyre_life_slope, 0.975, na.rm = TRUE),
  bootstrap_samples = B
) %>%
  mutate(
    across(
      c(original_slope, bootstrap_mean, bootstrap_se, ci_lower_95, ci_upper_95),
      ~ round(.x, 4)
    )
  )

write_csv(
  boot_slopes,
  file.path(table_dir, "ch06_bootstrap_tyre_life_slopes.csv")
)

write_csv(
  bootstrap_ci,
  file.path(table_dir, "ch06_bootstrap_tyre_life_slope_ci.csv")
)

cat("\nBootstrap confidence interval:\n")
print(bootstrap_ci)

# ----------------------------
# 13. Bootstrap histogram
# ----------------------------

p_bootstrap <- ggplot(boot_slopes, aes(x = tyre_life_slope)) +
  geom_histogram(bins = 40, fill = f1_red, color = "white", linewidth = 0.25) +
  geom_vline(
    xintercept = bootstrap_ci$original_slope,
    linewidth = 1,
    linetype = "dashed",
    color = "black"
  ) +
  geom_vline(
    xintercept = bootstrap_ci$ci_lower_95,
    linewidth = 0.8,
    linetype = "dotted",
    color = f1_dark
  ) +
  geom_vline(
    xintercept = bootstrap_ci$ci_upper_95,
    linewidth = 0.8,
    linetype = "dotted",
    color = f1_dark
  ) +
  labs(
    title = "Bootstrap Distribution of Tire Life Slope",
    subtitle = paste0(
      "Original slope = ",
      bootstrap_ci$original_slope,
      "; 95% bootstrap CI = [",
      bootstrap_ci$ci_lower_95,
      ", ",
      bootstrap_ci$ci_upper_95,
      "]"
    ),
    x = "Bootstrap Slope Estimate for Tire Life",
    y = "Count"
  ) +
  scale_y_continuous(labels = scales::comma)

safe_ggsave(
  plot_object = p_bootstrap,
  filename = "ch06_fig03_bootstrap_tyre_life_slope.png",
  width = 8,
  height = 5
)

# ----------------------------
# 14. Extra summary values for interpretation
# ----------------------------

reg_rmse_mean <- regression_cv_summary %>%
  filter(metric == "RMSE") %>%
  pull(mean_value)

reg_rmse_sd <- regression_cv_summary %>%
  filter(metric == "RMSE") %>%
  pull(sd_value)

reg_rmse_median <- regression_cv_summary %>%
  filter(metric == "RMSE") %>%
  pull(median_value)

reg_rmse_max <- regression_cv_summary %>%
  filter(metric == "RMSE") %>%
  pull(max_value)

reg_mae_mean <- regression_cv_summary %>%
  filter(metric == "MAE") %>%
  pull(mean_value)

reg_r2_mean <- regression_cv_summary %>%
  filter(metric == "Validation R-squared") %>%
  pull(mean_value)

class_accuracy <- classification_cv_summary %>%
  filter(metric == "Accuracy") %>%
  pull(mean_value)

class_sensitivity <- classification_cv_summary %>%
  filter(metric == "Sensitivity") %>%
  pull(mean_value)

class_specificity <- classification_cv_summary %>%
  filter(metric == "Specificity") %>%
  pull(mean_value)

class_precision <- classification_cv_summary %>%
  filter(metric == "Precision") %>%
  pull(mean_value)

class_auc <- classification_cv_summary %>%
  filter(metric == "AUC") %>%
  pull(mean_value)

highest_rmse_fold <- regression_cv_by_fold %>%
  arrange(desc(rmse)) %>%
  slice(1)

# ----------------------------
# 15. Save lightweight model record
# ----------------------------

resampling_record <- list(
  chapter_name = "Chapter 6 - Cross-Validation and Bootstrap",
  cv_type = "Grouped 5-fold cross-validation by race_id",
  regression_formula = deparse(reg_formula),
  classification_formula = deparse(class_formula),
  cv_summary = cv_summary,
  regression_cv_by_fold = regression_cv_by_fold,
  classification_cv_by_fold = classification_cv_by_fold,
  bootstrap_model = "lap_time_s ~ tyre_life",
  bootstrap_ci = bootstrap_ci,
  note = "This script uses training data only. The 2025 test set is untouched."
)

saveRDS(
  resampling_record,
  file.path(model_dir, "ch06_resampling_record.rds")
)

# ----------------------------
# 16. Updated interpretation notes
# ----------------------------

interpretation <- tibble(
  item = c(
    "Main resampling method",
    "Why grouped folds were used",
    "Regression CV model",
    "Regression RMSE result",
    "Regression MAE result",
    "Regression R-squared result",
    "Regression warning",
    "Classification CV model",
    "Classification accuracy",
    "Classification sensitivity",
    "Classification specificity",
    "Classification AUC",
    "Threshold note",
    "Bootstrap estimate",
    "Bootstrap confidence interval",
    "Bootstrap interpretation",
    "Chapter 6 takeaway",
    "Test set note"
  ),
  note = c(
    "This chapter used grouped 5-fold cross-validation and a bootstrap confidence interval.",
    "Folds were assigned by race_id because laps from the same race are connected. This avoids putting laps from the same race-year in both the training and validation fold.",
    "The regression CV model predicted lap_time_s using a multiple linear regression model with numeric lap predictors, compound, and year.",
    paste0(
      "The mean CV RMSE was ",
      reg_rmse_mean,
      " seconds, but the median RMSE was only ",
      reg_rmse_median,
      " seconds. This difference happened because fold ",
      highest_rmse_fold$fold,
      " had RMSE = ",
      highest_rmse_fold$rmse,
      "."
    ),
    paste0(
      "The mean CV MAE was ",
      reg_mae_mean,
      " seconds. This is more stable than RMSE here because MAE is less affected by a few very large errors."
    ),
    paste0(
      "The mean validation R-squared was ",
      reg_r2_mean,
      ", so the regression model only explained a small amount of held-out lap-time variation."
    ),
    "The regression CV results should be interpreted carefully because one fold had a much higher RMSE. This likely reflects unusual slow laps or race-specific conditions that are not fully captured by the model.",
    "The classification CV model predicted pit_next_lap using the same logistic regression setup from Chapter 5, excluding pit_stop to avoid leakage.",
    paste0(
      "The mean CV accuracy was ",
      class_accuracy,
      "."
    ),
    paste0(
      "The mean CV sensitivity was ",
      class_sensitivity,
      ", so the 0.50 threshold missed many actual pit-next-lap cases."
    ),
    paste0(
      "The mean CV specificity was ",
      class_specificity,
      ", so the model was much better at identifying no-pit-next-lap cases."
    ),
    paste0(
      "The mean CV AUC was ",
      class_auc,
      ", which means the model ranked pit-next-lap cases fairly well even though the 0.50 classification threshold was conservative."
    ),
    "The classification CV results use a 0.50 threshold. For a pit-stop warning tool, a lower threshold may be more useful because sensitivity matters.",
    "The bootstrap estimate was the slope of tyre_life in the simple linear regression model lap_time_s ~ tyre_life.",
    paste0(
      "The original slope was ",
      bootstrap_ci$original_slope,
      ". The 95% bootstrap confidence interval was [",
      bootstrap_ci$ci_lower_95,
      ", ",
      bootstrap_ci$ci_upper_95,
      "]."
    ),
    "The entire bootstrap interval was negative, so the negative slope from the SLR model was stable in this sample. However, this should not be treated as causal because tire life is also connected to race progress, fuel load, stint timing, and other race context.",
    "Cross-validation gave a more honest estimate of held-out model performance, and the bootstrap showed that one important slope estimate was stable. The main lesson is that the models are useful baselines, but they still miss important race-specific structure.",
    "The 2025 test set was not used in this script. Final test evaluation should be saved for later model comparison."
  )
)

write_csv(
  interpretation,
  file.path(table_dir, "ch06_resampling_interpretation_notes.csv")
)

report_notes <- c(
  "Chapter 6 Report Notes",
  "",
  "This chapter focuses on resampling. Cross-validation was used to estimate held-out model performance, and the bootstrap was used to estimate uncertainty for one slope estimate.",
  "",
  "The cross-validation step used 5 folds. Instead of randomly assigning individual laps to folds, the script assigned whole race-year IDs to folds. This matters because laps from the same race are connected, so putting laps from the same race in both training and validation could make the validation error look too optimistic.",
  "",
  paste0(
    "For the regression example, the model predicted lap_time_s using MLR. The mean CV RMSE was ",
    reg_rmse_mean,
    " seconds, but the median RMSE was ",
    reg_rmse_median,
    " seconds. The mean is much higher because fold ",
    highest_rmse_fold$fold,
    " had RMSE = ",
    highest_rmse_fold$rmse,
    ". This shows that RMSE is sensitive to a few very large errors or unusual race contexts."
  ),
  "",
  paste0(
    "The mean CV MAE was ",
    reg_mae_mean,
    " seconds, which gives a more typical sense of prediction error for this model. The mean validation R-squared was only ",
    reg_r2_mean,
    ", so the regression model still explains only a small amount of held-out lap-time variation."
  ),
  "",
  paste0(
    "For the classification example, the model predicted pit_next_lap using logistic regression. The mean CV accuracy was ",
    class_accuracy,
    ", mean sensitivity was ",
    class_sensitivity,
    ", mean specificity was ",
    class_specificity,
    ", mean precision was ",
    class_precision,
    ", and mean AUC was ",
    class_auc,
    "."
  ),
  "",
  "The classification results show the same pattern as Chapter 5. The model has strong specificity and a strong AUC, but the sensitivity is low at the 0.50 threshold. In plain language, the model is better at recognizing when a driver will not pit next lap than it is at catching the actual pit-next-lap cases. For a strategy warning tool, a lower threshold may be more useful.",
  "",
  paste0(
    "For the bootstrap example, the slope of tyre_life in the SLR model lap_time_s ~ tyre_life was bootstrapped using ",
    B,
    " resamples. The original slope was ",
    bootstrap_ci$original_slope,
    ", and the 95% bootstrap confidence interval was [",
    bootstrap_ci$ci_lower_95,
    ", ",
    bootstrap_ci$ci_upper_95,
    "]."
  ),
  "",
  "The bootstrap interval stayed negative, so the negative tire-life slope was stable in this sample. However, this does not mean older tires directly cause faster laps. Tire life is mixed with race progress, fuel load, stint timing, and race context, so the SLR slope should be interpreted as an association, not a causal effect.",
  "",
  "Real-world decision: Cross-validation gives a more honest estimate of model performance than training error alone. The bootstrap gives a practical way to see how stable an estimate is. For this project, both methods show that the models are useful baselines, but they still need later comparison against more flexible methods."
)

writeLines(
  report_notes,
  file.path(table_dir, "ch06_report_notes.txt")
)

cat("\nInterpretation notes:\n")
print(interpretation)

# ----------------------------
# 17. Final confirmation
# ----------------------------

cat("\n07_resampling_cv_bootstrap.R ran successfully.\n")
cat("Chapter 6 tables saved to: ", table_dir, "\n", sep = "")
cat("Chapter 6 figures saved to: ", figure_dir, "\n", sep = "")
cat("Chapter 6 model record saved to: ", model_dir, "\n", sep = "")
cat("Report notes saved to: ", file.path(table_dir, "ch06_report_notes.txt"), "\n", sep = "")
cat("The 2025 test set was not used in this script.\n")