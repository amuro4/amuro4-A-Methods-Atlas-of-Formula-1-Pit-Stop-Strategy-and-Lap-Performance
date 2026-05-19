# ============================================================
# MATH 4230 Capstone Project
# Script: scripts/12_knn.R
# Purpose: Chapter 11 - K-Nearest Neighbors
# ============================================================

# ----------------------------
# 1. Run setup
# ----------------------------

source("scripts/00_setup.R")

# ----------------------------
# 2. Extra packages for KNN
# ----------------------------

packages <- c(
  "tidyverse",
  "class",
  "pROC"
)

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# ----------------------------
# 3. Organized output folders
# ----------------------------

chapter_name <- "ch11_knn"

table_dir <- file.path("results/tables", chapter_name)
model_dir <- file.path("results/models", chapter_name)
figure_dir <- file.path("figures", chapter_name)

dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# 4. Helper functions
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

relevel_if_present <- function(x, ref_level) {
  x <- as.factor(x)
  
  if (ref_level %in% levels(x)) {
    x <- relevel(x, ref = ref_level)
  }
  
  return(x)
}

stratified_sample_indices <- function(y, n_per_class, seed = 4230) {
  set.seed(seed)
  
  y <- as.factor(y)
  class_levels <- levels(y)
  
  sampled_indices <- map(
    class_levels,
    function(class_value) {
      ids <- which(y == class_value)
      sample(ids, size = min(n_per_class, length(ids)), replace = FALSE)
    }
  ) %>%
    unlist()
  
  sample(sampled_indices, length(sampled_indices), replace = FALSE)
}

create_stratified_folds <- function(y, k = 3, seed = 4230) {
  set.seed(seed)
  
  y <- as.factor(y)
  folds <- rep(NA_integer_, length(y))
  
  for (class_value in levels(y)) {
    ids <- which(y == class_value)
    folds[ids] <- sample(rep(1:k, length.out = length(ids)))
  }
  
  folds
}

scale_train_test <- function(train_x, test_x) {
  train_center <- apply(train_x, 2, mean, na.rm = TRUE)
  train_scale <- apply(train_x, 2, sd, na.rm = TRUE)
  
  train_scale[train_scale == 0 | is.na(train_scale)] <- 1
  
  train_scaled <- scale(
    train_x,
    center = train_center,
    scale = train_scale
  )
  
  test_scaled <- scale(
    test_x,
    center = train_center,
    scale = train_scale
  )
  
  list(
    train = as.matrix(train_scaled),
    test = as.matrix(test_scaled),
    center = train_center,
    scale = train_scale
  )
}

get_knn_yes_probability <- function(knn_pred) {
  winning_prob <- attr(knn_pred, "prob")
  
  ifelse(
    as.character(knn_pred) == "Yes",
    winning_prob,
    1 - winning_prob
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

# ----------------------------
# 5. Load train and test data
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

# ----------------------------
# 6. Prepare KNN data
# ----------------------------

knn_data <- bind_rows(
  f1_train %>% mutate(split = "train"),
  f1_test %>% mutate(split = "test")
) %>%
  mutate(
    pit_next_lap_num = make_binary_target(pit_next_lap),
    pit_next_lap_factor = factor(
      if_else(pit_next_lap_num == 1, "Yes", "No"),
      levels = c("No", "Yes")
    ),
    compound = relevel_if_present(compound, "MEDIUM"),
    year = as.numeric(year)
  ) %>%
  select(
    split,
    pit_next_lap_factor,
    tyre_life,
    normalized_tyre_life,
    race_progress,
    stint,
    position,
    position_change,
    compound,
    year
  ) %>%
  drop_na()

knn_formula <- ~
  tyre_life +
  normalized_tyre_life +
  race_progress +
  stint +
  position +
  position_change +
  compound +
  year

x_all <- model.matrix(
  knn_formula,
  data = knn_data
)

x_all <- x_all[, colnames(x_all) != "(Intercept)", drop = FALSE]

train_index <- knn_data$split == "train"
test_index <- knn_data$split == "test"

x_train_full <- x_all[train_index, , drop = FALSE]
x_test_full <- x_all[test_index, , drop = FALSE]

y_train_full <- knn_data$pit_next_lap_factor[train_index]
y_test <- knn_data$pit_next_lap_factor[test_index]

zero_variance_cols <- apply(x_train_full, 2, sd, na.rm = TRUE) == 0

removed_predictors <- tibble(
  term = colnames(x_train_full)[zero_variance_cols],
  reason = "Zero variance in training data"
)

if (sum(zero_variance_cols) > 0) {
  x_train_full <- x_train_full[, !zero_variance_cols, drop = FALSE]
  x_test_full <- x_test_full[, !zero_variance_cols, drop = FALSE]
}

write_csv(
  removed_predictors,
  file.path(table_dir, "ch11_knn_removed_zero_variance_predictors.csv")
)

# ----------------------------
# 7. Balanced sample for KNN
# ----------------------------

# KNN can be slow because prediction compares test rows to training rows.
# This chapter uses a balanced training sample for speed, then evaluates
# the final model on the full 2025 test set.

final_train_per_class <- 2500
tune_per_class <- 1000
cv_folds <- 3

train_sample_idx <- stratified_sample_indices(
  y = y_train_full,
  n_per_class = final_train_per_class,
  seed = 4230
)

x_train_knn <- x_train_full[train_sample_idx, , drop = FALSE]
y_train_knn <- y_train_full[train_sample_idx]

tune_sample_idx_local <- stratified_sample_indices(
  y = y_train_knn,
  n_per_class = tune_per_class,
  seed = 4230
)

x_tune <- x_train_knn[tune_sample_idx_local, , drop = FALSE]
y_tune <- y_train_knn[tune_sample_idx_local]

data_summary <- tibble(
  item = c(
    "Full training rows",
    "Full testing rows",
    "Predictors used",
    "Zero-variance predictors removed",
    "KNN training sample rows",
    "KNN tuning sample rows",
    "Training sample No",
    "Training sample Yes",
    "Tuning sample No",
    "Tuning sample Yes",
    "CV folds for tuning"
  ),
  value = c(
    nrow(x_train_full),
    nrow(x_test_full),
    ncol(x_train_full),
    sum(zero_variance_cols),
    nrow(x_train_knn),
    nrow(x_tune),
    sum(y_train_knn == "No"),
    sum(y_train_knn == "Yes"),
    sum(y_tune == "No"),
    sum(y_tune == "Yes"),
    cv_folds
  )
)

write_csv(
  data_summary,
  file.path(table_dir, "ch11_knn_data_summary.csv")
)

cat("\nKNN data summary:\n")
print(data_summary)

# ----------------------------
# 8. Tune k using cross-validation
# ----------------------------

k_values <- c(3, 5, 7, 11, 15, 21, 31, 41, 51)

folds <- create_stratified_folds(
  y = y_tune,
  k = cv_folds,
  seed = 4230
)

cv_results <- tibble()

for (this_k in k_values) {
  
  fold_results <- tibble()
  
  for (fold_id in 1:cv_folds) {
    
    train_idx <- which(folds != fold_id)
    valid_idx <- which(folds == fold_id)
    
    scaled_fold <- scale_train_test(
      train_x = x_tune[train_idx, , drop = FALSE],
      test_x = x_tune[valid_idx, , drop = FALSE]
    )
    
    knn_pred <- class::knn(
      train = scaled_fold$train,
      test = scaled_fold$test,
      cl = y_tune[train_idx],
      k = this_k,
      prob = TRUE
    )
    
    yes_prob <- get_knn_yes_probability(knn_pred)
    
    fold_metrics <- evaluate_classification(
      actual = y_tune[valid_idx],
      predicted_class = knn_pred,
      predicted_probability = yes_prob
    ) %>%
      mutate(
        fold = fold_id,
        k = this_k,
        error_rate = 1 - accuracy,
        balanced_accuracy = mean(c(sensitivity, specificity), na.rm = TRUE)
      )
    
    fold_results <- bind_rows(fold_results, fold_metrics)
  }
  
  cv_results <- bind_rows(
    cv_results,
    fold_results %>%
      summarise(
        k = this_k,
        mean_cv_accuracy = mean(accuracy, na.rm = TRUE),
        mean_cv_error = mean(error_rate, na.rm = TRUE),
        mean_cv_sensitivity = mean(sensitivity, na.rm = TRUE),
        mean_cv_specificity = mean(specificity, na.rm = TRUE),
        mean_cv_precision = mean(precision, na.rm = TRUE),
        mean_cv_auc = mean(auc, na.rm = TRUE),
        mean_cv_balanced_accuracy = mean(balanced_accuracy, na.rm = TRUE),
        .groups = "drop"
      )
  )
  
  cat("Finished CV for k =", this_k, "\n")
}

cv_results <- cv_results %>%
  mutate(
    across(
      c(
        mean_cv_accuracy,
        mean_cv_error,
        mean_cv_sensitivity,
        mean_cv_specificity,
        mean_cv_precision,
        mean_cv_auc,
        mean_cv_balanced_accuracy
      ),
      ~ round(.x, 4)
    )
  ) %>%
  arrange(desc(mean_cv_balanced_accuracy), mean_cv_error, k)

write_csv(
  cv_results,
  file.path(table_dir, "ch11_knn_cv_results.csv")
)

cat("\nKNN CV results:\n")
print(cv_results)

best_k <- cv_results %>%
  slice(1) %>%
  pull(k)

cat("\nBest k selected:", best_k, "\n")

# ----------------------------
# 9. Fit final KNN on balanced sample and evaluate full test set
# ----------------------------

scaled_final <- scale_train_test(
  train_x = x_train_knn,
  test_x = x_test_full
)

knn_test_pred <- class::knn(
  train = scaled_final$train,
  test = scaled_final$test,
  cl = y_train_knn,
  k = best_k,
  prob = TRUE
)

knn_test_prob <- get_knn_yes_probability(knn_test_pred)

majority_prob <- rep(
  mean(y_train_full == "Yes", na.rm = TRUE),
  length(y_test)
)

majority_class <- factor(
  rep("No", length(y_test)),
  levels = c("No", "Yes")
)

performance_table <- bind_rows(
  evaluate_classification(
    actual = y_test,
    predicted_class = knn_test_pred,
    predicted_probability = knn_test_prob
  ) %>%
    mutate(
      model = "KNN",
      k = best_k
    ),
  evaluate_classification(
    actual = y_test,
    predicted_class = majority_class,
    predicted_probability = majority_prob
  ) %>%
    mutate(
      model = "Majority baseline",
      k = NA_real_
    )
) %>%
  select(
    model,
    k,
    accuracy,
    sensitivity,
    specificity,
    precision,
    auc,
    true_negative,
    false_positive,
    false_negative,
    true_positive
  ) %>%
  mutate(
    across(c(accuracy, sensitivity, specificity, precision, auc), ~ round(.x, 4))
  ) %>%
  arrange(desc(auc))

write_csv(
  performance_table,
  file.path(table_dir, "ch11_knn_performance.csv")
)

cat("\nKNN test performance:\n")
print(performance_table)

best_knn <- performance_table %>%
  filter(model == "KNN") %>%
  slice(1)

# ----------------------------
# 10. Minimal figures
# ----------------------------

p_cv_error <- cv_results %>%
  arrange(k) %>%
  ggplot(aes(x = k, y = mean_cv_error)) +
  geom_line(color = f1_red, linewidth = 1) +
  geom_point(color = f1_red, size = 2) +
  geom_vline(
    xintercept = best_k,
    linetype = "dashed",
    color = "gray35"
  ) +
  labs(
    title = "KNN Cross-Validation Error Curve",
    subtitle = paste0("Best k selected by balanced accuracy: k = ", best_k),
    x = "Number of Neighbors (k)",
    y = "Mean CV Error"
  )

safe_ggsave(
  plot_object = p_cv_error,
  filename = "ch11_fig01_knn_cv_error_curve.png",
  width = 8,
  height = 5
)

p_knn_metrics <- performance_table %>%
  filter(model == "KNN") %>%
  select(model, accuracy, sensitivity, specificity, precision, auc) %>%
  pivot_longer(
    cols = c(accuracy, sensitivity, specificity, precision, auc),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(metric = str_to_title(metric)) %>%
  ggplot(aes(x = metric, y = value)) +
  geom_col(fill = f1_red, width = 0.7) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    limits = c(0, 1)
  ) +
  labs(
    title = "KNN Test Performance",
    subtitle = paste0("Final KNN classifier used k = ", best_k, "."),
    x = "Metric",
    y = "Metric Value"
  )

safe_ggsave(
  plot_object = p_knn_metrics,
  filename = "ch11_fig02_knn_test_metrics.png",
  width = 8,
  height = 5
)

# ----------------------------
# 11. Lightweight model record
# ----------------------------

knn_record <- list(
  chapter_name = "Chapter 11 - K-Nearest Neighbors",
  formula = deparse(knn_formula),
  data_summary = data_summary,
  cv_results = cv_results,
  performance_table = performance_table,
  best_k = best_k,
  note = "KNN was trained on a balanced sample to keep runtime manageable. Evaluation used the full 2025 test set."
)

saveRDS(
  knn_record,
  file.path(model_dir, "ch11_knn_record.rds")
)

# ----------------------------
# 12. Report notes
# ----------------------------

report_notes <- c(
  "Chapter 11 Report Notes",
  "",
  "This chapter used K-nearest neighbors to predict pit_next_lap.",
  "",
  "Numeric predictors and dummy variables were scaled before fitting KNN. Scaling matters because KNN uses distance, so variables on larger scales can dominate the model if they are not standardized.",
  "",
  "Because KNN can be slow on a large dataset, the model was tuned and fit using a balanced sample from the training data. The final evaluation still used the full 2025 test set.",
  "",
  paste0(
    "The balanced KNN training sample had ",
    nrow(x_train_knn),
    " rows, with ",
    sum(y_train_knn == "No"),
    " No cases and ",
    sum(y_train_knn == "Yes"),
    " Yes cases."
  ),
  "",
  paste0(
    "The tuning sample had ",
    nrow(x_tune),
    " rows and used ",
    cv_folds,
    "-fold cross-validation."
  ),
  "",
  paste0(
    "The selected k was ",
    best_k,
    ". It was chosen using cross-validation and ranked by balanced accuracy."
  ),
  "",
  paste0(
    "On the full 2025 test set, KNN had accuracy = ",
    best_knn$accuracy,
    ", sensitivity = ",
    best_knn$sensitivity,
    ", specificity = ",
    best_knn$specificity,
    ", precision = ",
    best_knn$precision,
    ", and AUC = ",
    best_knn$auc,
    "."
  ),
  "",
  "Real-world decision: KNN is simple and flexible, but it can be computationally expensive because prediction requires comparing new observations to the training observations. It is useful as a comparison method, but may not be the best final model for this project."
)

writeLines(
  report_notes,
  file.path(table_dir, "ch11_report_notes.txt")
)

cat("\nReport notes:\n")
cat(paste(report_notes, collapse = "\n"))

# ----------------------------
# 13. Final confirmation
# ----------------------------

cat("\n\n12_knn.R ran successfully.\n")
cat("Chapter 11 tables saved to: ", table_dir, "\n", sep = "")
cat("Chapter 11 figures saved to: ", figure_dir, "\n", sep = "")
cat("Chapter 11 model record saved to: ", model_dir, "\n", sep = "")
cat("Report notes saved to: ", file.path(table_dir, "ch11_report_notes.txt"), "\n", sep = "")
cat("KNN was trained on a balanced sample and evaluated on the full 2025 test set.\n")