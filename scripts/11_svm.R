# ============================================================
# MATH 4230 Capstone Project
# Script: scripts/11_svm.R
# Purpose: Chapter 10 - Support Vector Machines
# ============================================================

# ----------------------------
# 1. Run setup
# ----------------------------

source("scripts/00_setup.R")

# ----------------------------
# 2. Extra packages for SVM
# ----------------------------

packages <- c(
  "tidyverse",
  "e1071",
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

chapter_name <- "ch10_svm"

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

# Fixed version.
# This avoids slice_sample(n = min(n_per_class, n())), which caused the error.
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

create_stratified_folds <- function(y, k = 2, seed = 4230) {
  set.seed(seed)
  
  y <- as.factor(y)
  folds <- rep(NA_integer_, length(y))
  
  for (class_value in levels(y)) {
    ids <- which(y == class_value)
    folds[ids] <- sample(rep(1:k, length.out = length(ids)))
  }
  
  folds
}

fit_svm_model <- function(x, y, kernel_type, cost_value, gamma_value = NULL, probability_flag = FALSE) {
  
  if (kernel_type == "linear") {
    fit <- e1071::svm(
      x = x,
      y = y,
      kernel = "linear",
      cost = cost_value,
      scale = TRUE,
      probability = probability_flag,
      cachesize = 500
    )
  } else {
    fit <- e1071::svm(
      x = x,
      y = y,
      kernel = "radial",
      cost = cost_value,
      gamma = gamma_value,
      scale = TRUE,
      probability = probability_flag,
      cachesize = 500
    )
  }
  
  fit
}

get_svm_yes_probability <- function(fit, new_x) {
  pred <- predict(fit, newdata = new_x, probability = TRUE)
  probs <- attr(pred, "probabilities")
  
  if (is.null(probs)) {
    return(if_else(as.character(pred) == "Yes", 1, 0))
  }
  
  if ("Yes" %in% colnames(probs)) {
    return(as.numeric(probs[, "Yes"]))
  }
  
  return(if_else(as.character(pred) == "Yes", 1, 0))
}

tune_svm_cv <- function(x, y, kernel_type, cost_values, gamma_values = NA_real_, k = 2) {
  
  folds <- create_stratified_folds(y, k = k, seed = 4230)
  
  tuning_grid <- expand_grid(
    cost = cost_values,
    gamma = gamma_values
  )
  
  tuning_results <- tibble()
  
  for (grid_row in 1:nrow(tuning_grid)) {
    
    this_cost <- tuning_grid$cost[grid_row]
    this_gamma <- tuning_grid$gamma[grid_row]
    
    fold_results <- tibble()
    
    for (fold_id in 1:k) {
      
      train_idx <- which(folds != fold_id)
      valid_idx <- which(folds == fold_id)
      
      svm_fit <- fit_svm_model(
        x = x[train_idx, , drop = FALSE],
        y = y[train_idx],
        kernel_type = kernel_type,
        cost_value = this_cost,
        gamma_value = this_gamma,
        probability_flag = FALSE
      )
      
      pred_class <- predict(
        svm_fit,
        newdata = x[valid_idx, , drop = FALSE]
      )
      
      actual <- factor(y[valid_idx], levels = c("No", "Yes"))
      pred_class <- factor(pred_class, levels = c("No", "Yes"))
      
      cm <- table(Actual = actual, Predicted = pred_class)
      
      tn <- as.numeric(cm["No", "No"])
      fp <- as.numeric(cm["No", "Yes"])
      fn <- as.numeric(cm["Yes", "No"])
      tp <- as.numeric(cm["Yes", "Yes"])
      
      sensitivity <- safe_divide(tp, tp + fn)
      specificity <- safe_divide(tn, tn + fp)
      
      fold_results <- bind_rows(
        fold_results,
        tibble(
          fold = fold_id,
          accuracy = safe_divide(tp + tn, tp + tn + fp + fn),
          sensitivity = sensitivity,
          specificity = specificity,
          balanced_accuracy = mean(c(sensitivity, specificity), na.rm = TRUE)
        )
      )
    }
    
    tuning_results <- bind_rows(
      tuning_results,
      fold_results %>%
        summarise(
          kernel = kernel_type,
          cost = this_cost,
          gamma = this_gamma,
          mean_accuracy = mean(accuracy, na.rm = TRUE),
          mean_sensitivity = mean(sensitivity, na.rm = TRUE),
          mean_specificity = mean(specificity, na.rm = TRUE),
          mean_balanced_accuracy = mean(balanced_accuracy, na.rm = TRUE),
          .groups = "drop"
        )
    )
    
    cat(
      "Finished tuning:",
      kernel_type,
      "cost =", this_cost,
      "gamma =", this_gamma,
      "\n"
    )
  }
  
  tuning_results %>%
    mutate(
      across(
        c(mean_accuracy, mean_sensitivity, mean_specificity, mean_balanced_accuracy),
        ~ round(.x, 4)
      )
    ) %>%
    arrange(desc(mean_balanced_accuracy), desc(mean_accuracy), cost, gamma)
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
# 6. Prepare SVM data
# ----------------------------

svm_data <- bind_rows(
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

svm_formula <- ~
  tyre_life +
  normalized_tyre_life +
  race_progress +
  stint +
  position +
  position_change +
  compound +
  year

x_all <- model.matrix(
  svm_formula,
  data = svm_data
)

x_all <- x_all[, colnames(x_all) != "(Intercept)", drop = FALSE]

train_index <- svm_data$split == "train"
test_index <- svm_data$split == "test"

x_train_full <- x_all[train_index, , drop = FALSE]
x_test <- x_all[test_index, , drop = FALSE]

y_train_full <- svm_data$pit_next_lap_factor[train_index]
y_test <- svm_data$pit_next_lap_factor[test_index]

zero_variance_cols <- apply(x_train_full, 2, sd, na.rm = TRUE) == 0

removed_predictors <- tibble(
  term = colnames(x_train_full)[zero_variance_cols],
  reason = "Zero variance in training data"
)

if (sum(zero_variance_cols) > 0) {
  x_train_full <- x_train_full[, !zero_variance_cols, drop = FALSE]
  x_test <- x_test[, !zero_variance_cols, drop = FALSE]
}

write_csv(
  removed_predictors,
  file.path(table_dir, "ch10_svm_removed_zero_variance_predictors.csv")
)

# ----------------------------
# 7. Use smaller balanced sample for SVM
# ----------------------------

# SVMs are expensive on large datasets.
# This uses a small balanced training sample for speed, then evaluates
# both final models on the full 2025 test set.

final_train_per_class <- 1500
tune_per_class <- 700
cv_folds <- 2

train_sample_idx <- stratified_sample_indices(
  y = y_train_full,
  n_per_class = final_train_per_class,
  seed = 4230
)

x_train_svm <- x_train_full[train_sample_idx, , drop = FALSE]
y_train_svm <- y_train_full[train_sample_idx]

tune_sample_idx_local <- stratified_sample_indices(
  y = y_train_svm,
  n_per_class = tune_per_class,
  seed = 4230
)

x_tune <- x_train_svm[tune_sample_idx_local, , drop = FALSE]
y_tune <- y_train_svm[tune_sample_idx_local]

data_summary <- tibble(
  item = c(
    "Full training rows",
    "Full testing rows",
    "Predictors used",
    "Zero-variance predictors removed",
    "SVM training sample rows",
    "SVM tuning sample rows",
    "Training sample No",
    "Training sample Yes",
    "Tuning sample No",
    "Tuning sample Yes",
    "CV folds for tuning"
  ),
  value = c(
    nrow(x_train_full),
    nrow(x_test),
    ncol(x_train_full),
    sum(zero_variance_cols),
    nrow(x_train_svm),
    nrow(x_tune),
    sum(y_train_svm == "No"),
    sum(y_train_svm == "Yes"),
    sum(y_tune == "No"),
    sum(y_tune == "Yes"),
    cv_folds
  )
)

write_csv(
  data_summary,
  file.path(table_dir, "ch10_svm_data_summary.csv")
)

cat("\nSVM data summary:\n")
print(data_summary)

# ----------------------------
# 8. Tune linear and RBF SVM
# ----------------------------

# Small grids keep runtime reasonable.
linear_cost_grid <- c(0.1, 1, 10)

rbf_cost_grid <- c(0.1, 1)
rbf_gamma_grid <- c(0.01, 0.05)

cat("\nTuning linear SVM...\n")

linear_tuning <- tune_svm_cv(
  x = x_tune,
  y = y_tune,
  kernel_type = "linear",
  cost_values = linear_cost_grid,
  gamma_values = NA_real_,
  k = cv_folds
)

cat("\nTuning RBF SVM...\n")

rbf_tuning <- tune_svm_cv(
  x = x_tune,
  y = y_tune,
  kernel_type = "radial",
  cost_values = rbf_cost_grid,
  gamma_values = rbf_gamma_grid,
  k = cv_folds
)

tuning_results <- bind_rows(
  linear_tuning,
  rbf_tuning
) %>%
  arrange(kernel, desc(mean_balanced_accuracy), desc(mean_accuracy), cost, gamma)

write_csv(
  tuning_results,
  file.path(table_dir, "ch10_svm_tuning_results.csv")
)

best_linear <- linear_tuning %>%
  slice(1)

best_rbf <- rbf_tuning %>%
  slice(1)

tuning_summary <- tibble(
  model = c("Linear SVM", "RBF SVM"),
  kernel = c("linear", "radial"),
  best_cost = c(best_linear$cost, best_rbf$cost),
  best_gamma = c(NA_real_, best_rbf$gamma),
  cv_accuracy = c(best_linear$mean_accuracy, best_rbf$mean_accuracy),
  cv_sensitivity = c(best_linear$mean_sensitivity, best_rbf$mean_sensitivity),
  cv_specificity = c(best_linear$mean_specificity, best_rbf$mean_specificity),
  cv_balanced_accuracy = c(
    best_linear$mean_balanced_accuracy,
    best_rbf$mean_balanced_accuracy
  )
)

write_csv(
  tuning_summary,
  file.path(table_dir, "ch10_svm_tuning_summary.csv")
)

cat("\nSVM tuning summary:\n")
print(tuning_summary)

# ----------------------------
# 9. Fit final SVM models on balanced sample
# ----------------------------

set.seed(4230)

linear_svm <- fit_svm_model(
  x = x_train_svm,
  y = y_train_svm,
  kernel_type = "linear",
  cost_value = best_linear$cost,
  gamma_value = NA_real_,
  probability_flag = TRUE
)

cat("\nFinal linear SVM fitted.\n")

set.seed(4230)

rbf_svm <- fit_svm_model(
  x = x_train_svm,
  y = y_train_svm,
  kernel_type = "radial",
  cost_value = best_rbf$cost,
  gamma_value = best_rbf$gamma,
  probability_flag = TRUE
)

cat("\nFinal RBF SVM fitted.\n")

# ----------------------------
# 10. Test-set performance
# ----------------------------

linear_prob <- get_svm_yes_probability(linear_svm, x_test)
rbf_prob <- get_svm_yes_probability(rbf_svm, x_test)

linear_class <- factor(
  if_else(linear_prob >= 0.50, "Yes", "No"),
  levels = c("No", "Yes")
)

rbf_class <- factor(
  if_else(rbf_prob >= 0.50, "Yes", "No"),
  levels = c("No", "Yes")
)

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
    predicted_class = linear_class,
    predicted_probability = linear_prob
  ) %>%
    mutate(
      model = "Linear SVM",
      kernel = "linear",
      cost = best_linear$cost,
      gamma = NA_real_
    ),
  evaluate_classification(
    actual = y_test,
    predicted_class = rbf_class,
    predicted_probability = rbf_prob
  ) %>%
    mutate(
      model = "RBF SVM",
      kernel = "radial",
      cost = best_rbf$cost,
      gamma = best_rbf$gamma
    ),
  evaluate_classification(
    actual = y_test,
    predicted_class = majority_class,
    predicted_probability = majority_prob
  ) %>%
    mutate(
      model = "Majority baseline",
      kernel = "none",
      cost = NA_real_,
      gamma = NA_real_
    )
) %>%
  select(
    model,
    kernel,
    cost,
    gamma,
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
  file.path(table_dir, "ch10_svm_performance.csv")
)

cat("\nSVM test performance:\n")
print(performance_table)

best_svm <- performance_table %>%
  filter(model != "Majority baseline") %>%
  arrange(desc(auc)) %>%
  slice(1)

# ----------------------------
# 11. Minimal figures
# ----------------------------

performance_plot_data <- performance_table %>%
  filter(model != "Majority baseline") %>%
  select(model, accuracy, sensitivity, specificity, precision, auc) %>%
  pivot_longer(
    cols = c(accuracy, sensitivity, specificity, precision, auc),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(metric = str_to_title(metric))

p_svm_metrics <- ggplot(
  performance_plot_data,
  aes(x = model, y = value, fill = metric)
) +
  geom_col(position = "dodge") +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    limits = c(0, 1)
  ) +
  labs(
    title = "SVM Test Performance",
    subtitle = "Models were trained on a balanced sample and evaluated on the full 2025 test set.",
    x = "Model",
    y = "Metric Value",
    fill = "Metric"
  )

safe_ggsave(
  plot_object = p_svm_metrics,
  filename = "ch10_fig01_svm_metrics.png",
  width = 8,
  height = 5
)

roc_linear <- pROC::roc(
  response = if_else(y_test == "Yes", 1, 0),
  predictor = linear_prob,
  quiet = TRUE
)

roc_rbf <- pROC::roc(
  response = if_else(y_test == "Yes", 1, 0),
  predictor = rbf_prob,
  quiet = TRUE
)

roc_plot_data <- bind_rows(
  tibble(
    model = "Linear SVM",
    specificity = roc_linear$specificities,
    sensitivity = roc_linear$sensitivities
  ),
  tibble(
    model = "RBF SVM",
    specificity = roc_rbf$specificities,
    sensitivity = roc_rbf$sensitivities
  )
) %>%
  mutate(false_positive_rate = 1 - specificity)

p_roc <- ggplot(
  roc_plot_data,
  aes(x = false_positive_rate, y = sensitivity, color = model)
) +
  geom_line(linewidth = 1) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    color = "gray50"
  ) +
  coord_equal() +
  labs(
    title = "SVM ROC Curves",
    subtitle = "Higher curves indicate better class ranking.",
    x = "False Positive Rate",
    y = "Sensitivity",
    color = "Model"
  )

safe_ggsave(
  plot_object = p_roc,
  filename = "ch10_fig02_svm_roc_curves.png",
  width = 7,
  height = 5
)

# ----------------------------
# 12. Lightweight model record
# ----------------------------

svm_record <- list(
  chapter_name = "Chapter 10 - Support Vector Machines",
  formula = deparse(svm_formula),
  data_summary = data_summary,
  tuning_summary = tuning_summary,
  performance_table = performance_table,
  best_svm = best_svm,
  note = "SVMs were trained on a smaller balanced sample to keep runtime manageable on a laptop. Evaluation used the full 2025 test set."
)

saveRDS(
  svm_record,
  file.path(model_dir, "ch10_svm_record.rds")
)

# ----------------------------
# 13. Report notes
# ----------------------------

report_notes <- c(
  "Chapter 10 Report Notes",
  "",
  "This chapter used support vector machines to predict pit_next_lap.",
  "",
  "A linear SVM and an RBF SVM were fit. Cost was tuned for both models, and gamma was tuned for the RBF model.",
  "",
  "Because SVMs can be slow on a large dataset, the models were tuned and fit using a smaller balanced sample from the training data. The final evaluation still used the full 2025 test set.",
  "",
  paste0(
    "The balanced SVM training sample had ",
    nrow(x_train_svm),
    " rows, with ",
    sum(y_train_svm == "No"),
    " No cases and ",
    sum(y_train_svm == "Yes"),
    " Yes cases."
  ),
  "",
  paste0(
    "The linear SVM used cost = ",
    best_linear$cost,
    ". Its tuning balanced accuracy was ",
    best_linear$mean_balanced_accuracy,
    "."
  ),
  "",
  paste0(
    "The RBF SVM used cost = ",
    best_rbf$cost,
    " and gamma = ",
    best_rbf$gamma,
    ". Its tuning balanced accuracy was ",
    best_rbf$mean_balanced_accuracy,
    "."
  ),
  "",
  paste0(
    "On the full 2025 test set, the best SVM by AUC was ",
    best_svm$model,
    ". It had accuracy = ",
    best_svm$accuracy,
    ", sensitivity = ",
    best_svm$sensitivity,
    ", specificity = ",
    best_svm$specificity,
    ", precision = ",
    best_svm$precision,
    ", and AUC = ",
    best_svm$auc,
    "."
  ),
  "",
  "Real-world decision: SVMs are useful for classification, especially when nonlinear boundaries may matter. However, they can be computationally expensive, so this chapter used a smaller balanced training sample as a practical compromise."
)

writeLines(
  report_notes,
  file.path(table_dir, "ch10_report_notes.txt")
)

cat("\nReport notes:\n")
cat(paste(report_notes, collapse = "\n"))

# ----------------------------
# 14. Final confirmation
# ----------------------------

cat("\n\n11_svm.R ran successfully.\n")
cat("Chapter 10 tables saved to: ", table_dir, "\n", sep = "")
cat("Chapter 10 figures saved to: ", figure_dir, "\n", sep = "")
cat("Chapter 10 model record saved to: ", model_dir, "\n", sep = "")
cat("Report notes saved to: ", file.path(table_dir, "ch10_report_notes.txt"), "\n", sep = "")
cat("SVMs were trained on a balanced sample and evaluated on the full 2025 test set.\n")