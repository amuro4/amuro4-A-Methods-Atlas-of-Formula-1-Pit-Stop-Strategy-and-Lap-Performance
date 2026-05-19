# ============================================================
# MATH 4230 Capstone Project
# Script: scripts/15_method_comparison.R
# Purpose: Chapter 14 - Method Comparison and Final Recommendation
# ============================================================

# ----------------------------
# 1. Run setup
# ----------------------------

source("scripts/00_setup.R")

# ----------------------------
# 2. Packages
# ----------------------------

packages <- c(
  "tidyverse"
)

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# ----------------------------
# 3. Output folders
# ----------------------------

table_dir <- "results/tables"

dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# 4. Helper functions
# ----------------------------

read_if_exists <- function(path) {
  if (file.exists(path)) {
    read_csv(path, show_col_types = FALSE)
  } else {
    NULL
  }
}

clean_method_name <- function(x) {
  x %>%
    str_replace_all("_", " ") %>%
    str_replace_all("-", " ") %>%
    str_squish() %>%
    str_to_title()
}

make_method_label <- function(data, source_name) {
  
  method <- if ("model" %in% names(data)) {
    as.character(data$model)
  } else {
    clean_method_name(source_name)
  }
  
  if ("lambda_type" %in% names(data)) {
    method <- paste(method, data$lambda_type)
  }
  
  if ("tree_version" %in% names(data)) {
    method <- paste(method, data$tree_version)
  }
  
  if ("kernel" %in% names(data)) {
    method <- if_else(
      !is.na(data$kernel) & data$kernel != "none",
      paste(method, paste0("(", data$kernel, ")")),
      method
    )
  }
  
  if ("k" %in% names(data)) {
    method <- if_else(
      !is.na(data$k),
      paste(method, paste0("k=", data$k)),
      method
    )
  }
  
  method %>%
    str_squish()
}

get_tuning_method <- function(method) {
  method_lower <- str_to_lower(method)
  
  case_when(
    str_detect(method_lower, "mean baseline|majority baseline") ~ "No tuning",
    str_detect(method_lower, "simple linear|slr") ~ "No tuning",
    str_detect(method_lower, "multiple linear|mlr") ~ "No tuning",
    str_detect(method_lower, "logistic") ~ "Fixed 0.50 threshold; ROC/AUC checked",
    str_detect(method_lower, "ridge|lasso") ~ "5-fold CV selected lambda",
    str_detect(method_lower, "tree") ~ "CP pruning / tree complexity",
    str_detect(method_lower, "bagging") ~ "OOB error; mtry fixed to all predictors",
    str_detect(method_lower, "random forest") ~ "OOB error; mtry fixed",
    str_detect(method_lower, "boosting") ~ "CV selected number of trees",
    str_detect(method_lower, "svm") ~ "CV selected cost/gamma",
    str_detect(method_lower, "knn") ~ "CV selected k",
    str_detect(method_lower, "neural") ~ "Validation tuning for size/decay",
    TRUE ~ "See chapter notes"
  )
}

get_interpretability <- function(method) {
  method_lower <- str_to_lower(method)
  
  case_when(
    str_detect(method_lower, "mean baseline|majority baseline") ~ "High",
    str_detect(method_lower, "simple linear|slr|multiple linear|mlr|logistic") ~ "High",
    str_detect(method_lower, "lasso|ridge") ~ "Medium",
    str_detect(method_lower, "tree") & !str_detect(method_lower, "random forest|boosting|bagging") ~ "High",
    str_detect(method_lower, "knn|linear svm") ~ "Medium",
    str_detect(method_lower, "random forest|bagging|boosting|rbf svm|neural") ~ "Low",
    TRUE ~ "Medium"
  )
}

get_notes <- function(method, task) {
  method_lower <- str_to_lower(method)
  
  case_when(
    str_detect(method_lower, "mean baseline|majority baseline") ~
      "Baseline comparison only.",
    str_detect(method_lower, "lasso") ~
      "Best regularized regression model; useful because it shrinks and can remove predictors.",
    str_detect(method_lower, "ridge") ~
      "Stable regularized regression model; kept all predictors.",
    str_detect(method_lower, "regression tree") ~
      "Interpretable, but did not generalize well for lap-time prediction.",
    str_detect(method_lower, "boosting") & task == "Classification" ~
      "Strongest classification model by AUC among the main ensemble models.",
    str_detect(method_lower, "boosting") & task == "Regression" ~
      "Best regression ensemble, but still worse than the mean baseline.",
    str_detect(method_lower, "random forest") ~
      "Strong nonlinear benchmark; less interpretable than a single tree.",
    str_detect(method_lower, "bagging") ~
      "Averaged many trees; useful comparison to random forest.",
    str_detect(method_lower, "linear svm") ~
      "More usable threshold behavior than RBF SVM.",
    str_detect(method_lower, "rbf svm") ~
      "Better AUC than linear SVM, but default threshold created too many false positives.",
    str_detect(method_lower, "knn") ~
      "High sensitivity but many false positives.",
    str_detect(method_lower, "neural") ~
      "Flexible nonlinear model; improved over logistic benchmark but did not beat random forest.",
    str_detect(method_lower, "logistic") ~
      "Interpretable classification baseline with odds/probability interpretation.",
    TRUE ~
      "Included for method comparison."
  )
}

build_regression_rows <- function(data, source_file) {
  
  if (!("test_rmse" %in% names(data))) {
    return(tibble())
  }
  
  rows <- data %>%
    filter(!is.na(test_rmse)) %>%
    mutate(
      source_file = basename(source_file),
      method = make_method_label(cur_data_all(), basename(source_file)),
      task = "Regression",
      primary_metric = "Test RMSE",
      test_metric = as.numeric(test_rmse),
      accuracy = NA_real_,
      auc = NA_real_,
      sensitivity = NA_real_,
      specificity = NA_real_,
      precision = NA_real_,
      test_rmse = as.numeric(test_rmse),
      test_mae = if ("test_mae" %in% names(.)) as.numeric(test_mae) else NA_real_,
      test_r_squared = if ("test_r_squared" %in% names(.)) as.numeric(test_r_squared) else NA_real_
    ) %>%
    select(
      method,
      task,
      primary_metric,
      test_metric,
      accuracy,
      auc,
      sensitivity,
      specificity,
      precision,
      test_rmse,
      test_mae,
      test_r_squared,
      source_file
    )
  
  rows
}

build_classification_rows <- function(data, source_file) {
  
  class_cols <- c("accuracy", "auc", "sensitivity", "specificity", "precision")
  
  if (!any(class_cols %in% names(data))) {
    return(tibble())
  }
  
  if (!("auc" %in% names(data))) {
    return(tibble())
  }
  
  rows <- data %>%
    filter(!is.na(auc)) %>%
    mutate(
      source_file = basename(source_file),
      method = make_method_label(cur_data_all(), basename(source_file)),
      task = "Classification",
      primary_metric = "AUC",
      test_metric = as.numeric(auc),
      accuracy = if ("accuracy" %in% names(.)) as.numeric(accuracy) else NA_real_,
      auc = as.numeric(auc),
      sensitivity = if ("sensitivity" %in% names(.)) as.numeric(sensitivity) else NA_real_,
      specificity = if ("specificity" %in% names(.)) as.numeric(specificity) else NA_real_,
      precision = if ("precision" %in% names(.)) as.numeric(precision) else NA_real_,
      test_rmse = NA_real_,
      test_mae = NA_real_,
      test_r_squared = NA_real_
    ) %>%
    select(
      method,
      task,
      primary_metric,
      test_metric,
      accuracy,
      auc,
      sensitivity,
      specificity,
      precision,
      test_rmse,
      test_mae,
      test_r_squared,
      source_file
    )
  
  rows
}

# ----------------------------
# 5. Find saved performance files
# ----------------------------

all_csv_files <- list.files(
  path = "results/tables",
  pattern = "\\.csv$",
  recursive = TRUE,
  full.names = TRUE
)

performance_files <- all_csv_files[
  str_detect(
    basename(all_csv_files),
    regex("performance|test_performance", ignore_case = TRUE)
  )
]

performance_files <- performance_files[
  !str_detect(
    basename(performance_files),
    regex("prediction|predictions|importance|tuning|cv_results|summary", ignore_case = TRUE)
  )
]

cat("Performance files found:\n")
print(performance_files)

# ----------------------------
# 6. Build automatic comparison table
# ----------------------------

auto_rows <- tibble()

for (file_path in performance_files) {
  
  this_data <- read_csv(file_path, show_col_types = FALSE)
  
  reg_rows <- build_regression_rows(this_data, file_path)
  class_rows <- build_classification_rows(this_data, file_path)
  
  auto_rows <- bind_rows(
    auto_rows,
    reg_rows,
    class_rows
  )
}

# ----------------------------
# 7. Manual safety rows for key known chapter results
# ----------------------------

# These rows make the final comparison stable even if a previous
# chapter used a slightly different file name. If the same method is
# found automatically, the automatic row is kept.

manual_rows <- tribble(
  ~method, ~task, ~primary_metric, ~test_metric, ~accuracy, ~auc, ~sensitivity, ~specificity, ~precision, ~test_rmse, ~test_mae, ~test_r_squared, ~source_file,
  "Lasso lambda.min", "Regression", "Test RMSE", 10.4757, NA, NA, NA, NA, NA, 10.4757, 7.3175, 0.4075, "manual_from_ch07",
  "Ridge lambda.min", "Regression", "Test RMSE", 10.5020, NA, NA, NA, NA, NA, 10.5020, 7.3859, 0.4046, "manual_from_ch07",
  "Mean baseline", "Regression", "Test RMSE", 13.7817, NA, NA, NA, NA, NA, 13.7817, 10.8053, -0.0254, "manual_baseline",
  "Regression tree Pruned cp.min", "Regression", "Test RMSE", 17.3523, NA, NA, NA, NA, NA, 17.3523, 11.1760, -0.6260, "manual_from_ch08",
  "Boosting", "Regression", "Test RMSE", 18.5304, NA, NA, NA, NA, NA, 18.5304, 11.2395, -0.8538, "manual_from_ch09",
  "Random forest", "Regression", "Test RMSE", 34.1639, NA, NA, NA, NA, NA, 34.1639, 11.7592, -5.3025, "manual_from_ch09",
  "Bagging", "Regression", "Test RMSE", 52.2524, NA, NA, NA, NA, NA, 52.2524, 13.0792, -13.7219, "manual_from_ch09",
  "Boosting", "Classification", "AUC", 0.8608, 0.7909, 0.8608, 0.6384, 0.8649, 0.6963, NA, NA, NA, "manual_from_ch09",
  "Random forest", "Classification", "AUC", 0.8496, 0.7868, 0.8496, 0.6514, 0.8529, 0.6832, NA, NA, NA, "manual_from_ch09",
  "Bagging", "Classification", "AUC", 0.8433, 0.7785, 0.8433, 0.6696, 0.8310, 0.6569, NA, NA, NA, "manual_from_ch09",
  "Classification tree Full tree", "Classification", "AUC", 0.7818, 0.7607, 0.7818, 0.6679, 0.8056, 0.6251, NA, NA, NA, "manual_from_ch08",
  "RBF SVM", "Classification", "AUC", 0.7859, 0.3535, 0.7859, 0.9973, 0.0411, 0.3354, NA, NA, NA, "manual_from_ch10",
  "Linear SVM", "Classification", "AUC", 0.7328, 0.6882, 0.7328, 0.5971, 0.7324, 0.5198, NA, NA, NA, "manual_from_ch10",
  "KNN k=3", "Classification", "AUC", 0.7100, 0.6198, 0.7100, 0.7981, 0.5333, 0.4534, NA, NA, NA, "manual_from_ch11",
  "Neural network", "Classification", "AUC", 0.7948, 0.6876, 0.7948, 0.8672, 0.6005, 0.5130, NA, NA, NA, "manual_from_ch13",
  "Logistic regression benchmark", "Classification", "AUC", 0.7243, 0.6684, 0.7243, 0.6861, 0.6598, 0.4945, NA, NA, NA, "manual_from_ch13",
  "Random forest benchmark", "Classification", "AUC", 0.8271, 0.7070, 0.8271, 0.8636, 0.6310, 0.5317, NA, NA, NA, "manual_from_ch13",
  "Majority baseline", "Classification", "AUC", 0.5000, 0.6733, 0.5000, 0.0000, 1.0000, NA, NA, NA, NA, "manual_baseline"
)

combined_rows <- bind_rows(auto_rows, manual_rows) %>%
  mutate(
    method_key = str_to_lower(method) %>%
      str_replace_all("[^a-z0-9]+", " ") %>%
      str_squish(),
    task_key = str_to_lower(task),
    is_manual = str_detect(source_file, "^manual")
  ) %>%
  arrange(task_key, method_key, is_manual) %>%
  group_by(method_key, task_key) %>%
  slice(1) %>%
  ungroup() %>%
  select(-method_key, -task_key, -is_manual)

# ----------------------------
# 8. Add interpretation fields
# ----------------------------

method_comparison <- combined_rows %>%
  mutate(
    tuning_method = map_chr(method, get_tuning_method),
    interpretability = map_chr(method, get_interpretability),
    notes = map2_chr(method, task, get_notes)
  ) %>%
  mutate(
    across(
      c(
        test_metric,
        accuracy,
        auc,
        sensitivity,
        specificity,
        precision,
        test_rmse,
        test_mae,
        test_r_squared
      ),
      ~ round(.x, 4)
    )
  ) %>%
  arrange(
    task,
    if_else(task == "Regression", test_metric, -test_metric)
  ) %>%
  select(
    method,
    task,
    primary_metric,
    test_metric,
    accuracy,
    auc,
    sensitivity,
    specificity,
    precision,
    test_rmse,
    test_mae,
    test_r_squared,
    tuning_method,
    interpretability,
    notes,
    source_file
  )

# ----------------------------
# 9. Select best models and recommendations
# ----------------------------

best_regression <- method_comparison %>%
  filter(
    task == "Regression",
    !str_detect(str_to_lower(method), "baseline")
  ) %>%
  arrange(test_rmse) %>%
  slice(1)

best_classification_auc <- method_comparison %>%
  filter(
    task == "Classification",
    !str_detect(str_to_lower(method), "baseline")
  ) %>%
  arrange(desc(auc)) %>%
  slice(1)

best_interpretable_classification <- method_comparison %>%
  filter(
    task == "Classification",
    interpretability == "High",
    !str_detect(str_to_lower(method), "baseline")
  ) %>%
  arrange(desc(auc)) %>%
  slice(1)

recommendations <- tibble(
  recommendation_type = c(
    "Best regression method",
    "Best classification method by AUC",
    "Best interpretable classification option",
    "Best overall practical recommendation"
  ),
  selected_method = c(
    best_regression$method,
    best_classification_auc$method,
    best_interpretable_classification$method,
    "Use lasso for lap-time regression and boosting for pit-next-lap classification. Use logistic regression or a decision tree when explanation matters more than raw performance."
  ),
  reason = c(
    paste0(
      best_regression$method,
      " had the lowest test RMSE among the regression models, with RMSE = ",
      best_regression$test_rmse,
      "."
    ),
    paste0(
      best_classification_auc$method,
      " had the highest AUC among the classification models, with AUC = ",
      best_classification_auc$auc,
      "."
    ),
    paste0(
      best_interpretable_classification$method,
      " was the strongest high-interpretability classification option available in the comparison table."
    ),
    "The practical recommendation separates prediction from explanation. Flexible models help performance, but simpler models are easier to justify and explain."
  )
)

# ----------------------------
# 10. Save outputs
# ----------------------------

write_csv(
  method_comparison,
  file.path(table_dir, "method_comparison.csv")
)

write_csv(
  recommendations,
  file.path(table_dir, "method_recommendations.csv")
)

report_notes <- c(
  "Chapter 14 Report Notes",
  "",
  "This chapter compares the main methods used in the project.",
  "",
  paste0(
    "The best regression method was ",
    best_regression$method,
    ", with test RMSE = ",
    best_regression$test_rmse,
    "."
  ),
  "",
  paste0(
    "The best classification method by AUC was ",
    best_classification_auc$method,
    ", with AUC = ",
    best_classification_auc$auc,
    "."
  ),
  "",
  paste0(
    "The best high-interpretability classification option was ",
    best_interpretable_classification$method,
    ". This matters because a model that is easier to explain can be more useful in a report or real strategy discussion, even when it is not the highest-AUC model."
  ),
  "",
  "Final practical recommendation:",
  "Use lasso for lap-time regression because it had the strongest regression test RMSE while still being more interpretable than black-box models.",
  "Use boosting for pit-next-lap classification because it had the strongest classification AUC among the main models.",
  "Use logistic regression or a decision tree when interpretability matters more than raw prediction strength.",
  "",
  "Important interpretation point:",
  "Interpretability sometimes beat raw accuracy in this project. For example, decision trees and logistic regression were easier to explain than boosting, SVMs, or neural networks. Even when flexible models had stronger AUC, simpler models were more useful for explaining why a prediction was made.",
  "",
  "Important limitation:",
  "The dataset does not include race-status and circuit-condition variables such as Safety Car, Virtual Safety Car, yellow flags, red flags, track temperature, air temperature, circuit length, and pit lane time loss. This limits every method in the comparison."
)

writeLines(
  report_notes,
  file.path(table_dir, "ch14_method_comparison_report_notes.txt")
)

# ----------------------------
# 11. Print summary
# ----------------------------

cat("\nMethod comparison saved to:\n")
cat(file.path(table_dir, "method_comparison.csv"), "\n")

cat("\nRecommendations saved to:\n")
cat(file.path(table_dir, "method_recommendations.csv"), "\n")

cat("\nReport notes saved to:\n")
cat(file.path(table_dir, "ch14_method_comparison_report_notes.txt"), "\n")

cat("\nBest regression method:\n")
print(best_regression)

cat("\nBest classification method by AUC:\n")
print(best_classification_auc)

cat("\nBest interpretable classification option:\n")
print(best_interpretable_classification)

cat("\n15_method_comparison.R ran successfully.\n")