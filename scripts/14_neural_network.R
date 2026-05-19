# ============================================================
# MATH 4230 Capstone Project
# Script: scripts/14_neural_network.R
# Purpose: Chapter 13 - Neural Network
# ============================================================

# ----------------------------
# 1. Run setup
# ----------------------------

source("scripts/00_setup.R")

# ----------------------------
# 2. Extra packages
# ----------------------------

packages <- c(
  "tidyverse",
  "nnet",
  "randomForest",
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

chapter_name <- "ch13_neural_network"

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

stratified_train_valid_split <- function(y, valid_prop = 0.30, seed = 4230) {
  set.seed(seed)
  
  y <- as.factor(y)
  
  valid_idx <- map(
    levels(y),
    function(class_value) {
      ids <- which(y == class_value)
      n_valid <- max(1, floor(length(ids) * valid_prop))
      sample(ids, size = n_valid, replace = FALSE)
    }
  ) %>%
    unlist()
  
  train_idx <- setdiff(seq_along(y), valid_idx)
  
  list(
    train = train_idx,
    valid = valid_idx
  )
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

get_nn_yes_probability <- function(nn_fit, new_x) {
  pred <- predict(nn_fit, newdata = new_x, type = "raw")
  
  if (is.vector(pred)) {
    return(as.numeric(pred))
  }
  
  pred <- as.matrix(pred)
  
  if ("Yes" %in% colnames(pred)) {
    return(as.numeric(pred[, "Yes"]))
  }
  
  as.numeric(pred[, ncol(pred)])
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
# 6. Prepare neural network data
# ----------------------------

nn_data <- bind_rows(
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

nn_formula <- ~
  tyre_life +
  normalized_tyre_life +
  race_progress +
  stint +
  position +
  position_change +
  compound +
  year

x_all <- model.matrix(
  nn_formula,
  data = nn_data
)

x_all <- x_all[, colnames(x_all) != "(Intercept)", drop = FALSE]

train_index <- nn_data$split == "train"
test_index <- nn_data$split == "test"

x_train_full <- x_all[train_index, , drop = FALSE]
x_test_full <- x_all[test_index, , drop = FALSE]

y_train_full <- nn_data$pit_next_lap_factor[train_index]
y_test <- nn_data$pit_next_lap_factor[test_index]

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
  file.path(table_dir, "ch13_nn_removed_zero_variance_predictors.csv")
)

# ----------------------------
# 7. Balanced sample for runtime
# ----------------------------

# Neural networks can take time to tune, especially on a large dataset.
# This chapter uses a balanced training sample for fitting and tuning.
# The final evaluation still uses the full 2025 test set.

final_train_per_class <- 2000
tune_per_class <- 1000
valid_prop <- 0.30

train_sample_idx <- stratified_sample_indices(
  y = y_train_full,
  n_per_class = final_train_per_class,
  seed = 4230
)

x_train_nn_raw <- x_train_full[train_sample_idx, , drop = FALSE]
y_train_nn <- y_train_full[train_sample_idx]

tune_sample_idx_local <- stratified_sample_indices(
  y = y_train_nn,
  n_per_class = tune_per_class,
  seed = 4230
)

x_tune_raw <- x_train_nn_raw[tune_sample_idx_local, , drop = FALSE]
y_tune <- y_train_nn[tune_sample_idx_local]

tune_split <- stratified_train_valid_split(
  y = y_tune,
  valid_prop = valid_prop,
  seed = 4230
)

x_tune_train_raw <- x_tune_raw[tune_split$train, , drop = FALSE]
x_tune_valid_raw <- x_tune_raw[tune_split$valid, , drop = FALSE]

y_tune_train <- y_tune[tune_split$train]
y_tune_valid <- y_tune[tune_split$valid]

scaled_tune <- scale_train_test(
  train_x = x_tune_train_raw,
  test_x = x_tune_valid_raw
)

x_tune_train <- scaled_tune$train
x_tune_valid <- scaled_tune$test

data_summary <- tibble(
  item = c(
    "Full training rows",
    "Full testing rows",
    "Predictors used",
    "Zero-variance predictors removed",
    "NN training sample rows",
    "NN tuning sample rows",
    "NN tuning train rows",
    "NN tuning validation rows",
    "Training sample No",
    "Training sample Yes",
    "Tuning sample No",
    "Tuning sample Yes"
  ),
  value = c(
    nrow(x_train_full),
    nrow(x_test_full),
    ncol(x_train_full),
    sum(zero_variance_cols),
    nrow(x_train_nn_raw),
    nrow(x_tune_raw),
    nrow(x_tune_train),
    nrow(x_tune_valid),
    sum(y_train_nn == "No"),
    sum(y_train_nn == "Yes"),
    sum(y_tune == "No"),
    sum(y_tune == "Yes")
  )
)

write_csv(
  data_summary,
  file.path(table_dir, "ch13_nn_data_summary.csv")
)

cat("\nNeural network data summary:\n")
print(data_summary)

# ----------------------------
# 8. Light tuning for size and decay
# ----------------------------

# size = number of hidden nodes
# decay = weight decay regularization

size_grid <- c(1, 3, 5)
decay_grid <- c(0.001, 0.01)

tuning_grid <- expand_grid(
  size = size_grid,
  decay = decay_grid
)

nn_tuning_results <- tibble()

y_tune_train_matrix <- nnet::class.ind(y_tune_train)

for (grid_row in 1:nrow(tuning_grid)) {
  
  this_size <- tuning_grid$size[grid_row]
  this_decay <- tuning_grid$decay[grid_row]
  
  set.seed(4230 + grid_row)
  
  nn_fit <- nnet::nnet(
    x = x_tune_train,
    y = y_tune_train_matrix,
    size = this_size,
    decay = this_decay,
    softmax = TRUE,
    maxit = 250,
    trace = FALSE,
    MaxNWts = 5000
  )
  
  valid_prob <- get_nn_yes_probability(nn_fit, x_tune_valid)
  
  valid_class <- factor(
    if_else(valid_prob >= 0.50, "Yes", "No"),
    levels = c("No", "Yes")
  )
  
  valid_metrics <- evaluate_classification(
    actual = y_tune_valid,
    predicted_class = valid_class,
    predicted_probability = valid_prob
  ) %>%
    mutate(
      size = this_size,
      decay = this_decay,
      balanced_accuracy = mean(c(sensitivity, specificity), na.rm = TRUE)
    )
  
  nn_tuning_results <- bind_rows(
    nn_tuning_results,
    valid_metrics
  )
  
  cat(
    "Finished NN tuning: size =",
    this_size,
    "decay =",
    this_decay,
    "\n"
  )
}

nn_tuning_results <- nn_tuning_results %>%
  select(
    size,
    decay,
    accuracy,
    sensitivity,
    specificity,
    precision,
    auc,
    balanced_accuracy,
    true_negative,
    false_positive,
    false_negative,
    true_positive
  ) %>%
  mutate(
    across(
      c(accuracy, sensitivity, specificity, precision, auc, balanced_accuracy),
      ~ round(.x, 4)
    )
  ) %>%
  arrange(desc(auc), desc(balanced_accuracy), desc(accuracy), size, decay)

write_csv(
  nn_tuning_results,
  file.path(table_dir, "ch13_nn_tuning_results.csv")
)

cat("\nNeural network tuning results:\n")
print(nn_tuning_results)

best_nn_settings <- nn_tuning_results %>%
  slice(1)

best_size <- best_nn_settings$size
best_decay <- best_nn_settings$decay

cat("\nBest NN settings:\n")
cat("size =", best_size, "\n")
cat("decay =", best_decay, "\n")

# ----------------------------
# 9. Fit final models on balanced sample
# ----------------------------

scaled_final <- scale_train_test(
  train_x = x_train_nn_raw,
  test_x = x_test_full
)

x_train_nn <- scaled_final$train
x_test <- scaled_final$test

# Clean column names for data-frame models.
clean_names <- make.names(colnames(x_train_nn), unique = TRUE)

x_train_df <- as.data.frame(x_train_nn)
x_test_df <- as.data.frame(x_test)

names(x_train_df) <- clean_names
names(x_test_df) <- clean_names

# Neural network
set.seed(4230)

final_nn <- nnet::nnet(
  x = x_train_nn,
  y = nnet::class.ind(y_train_nn),
  size = best_size,
  decay = best_decay,
  softmax = TRUE,
  maxit = 250,
  trace = FALSE,
  MaxNWts = 5000
)

cat("\nFinal neural network fitted.\n")

# Logistic regression benchmark
glm_train_df <- x_train_df %>%
  mutate(
    y_bin = if_else(y_train_nn == "Yes", 1, 0)
  )

logistic_fit <- glm(
  y_bin ~ .,
  data = glm_train_df,
  family = binomial()
)

cat("Logistic regression benchmark fitted.\n")

# Small random forest benchmark
set.seed(4230)

rf_fit <- randomForest::randomForest(
  x = x_train_df,
  y = y_train_nn,
  ntree = 100,
  mtry = max(1, floor(sqrt(ncol(x_train_df)))),
  importance = FALSE
)

cat("Random forest benchmark fitted.\n")

# ----------------------------
# 10. Test-set performance
# ----------------------------

nn_prob <- get_nn_yes_probability(final_nn, x_test)

nn_class <- factor(
  if_else(nn_prob >= 0.50, "Yes", "No"),
  levels = c("No", "Yes")
)

logistic_prob <- as.numeric(
  predict(
    logistic_fit,
    newdata = x_test_df,
    type = "response"
  )
)

logistic_class <- factor(
  if_else(logistic_prob >= 0.50, "Yes", "No"),
  levels = c("No", "Yes")
)

rf_prob_matrix <- predict(
  rf_fit,
  newdata = x_test_df,
  type = "prob"
)

rf_prob <- as.numeric(rf_prob_matrix[, "Yes"])

rf_class <- factor(
  if_else(rf_prob >= 0.50, "Yes", "No"),
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
    predicted_class = nn_class,
    predicted_probability = nn_prob
  ) %>%
    mutate(
      model = "Neural network",
      details = paste0("size=", best_size, ", decay=", best_decay)
    ),
  evaluate_classification(
    actual = y_test,
    predicted_class = logistic_class,
    predicted_probability = logistic_prob
  ) %>%
    mutate(
      model = "Logistic regression",
      details = "balanced sample benchmark"
    ),
  evaluate_classification(
    actual = y_test,
    predicted_class = rf_class,
    predicted_probability = rf_prob
  ) %>%
    mutate(
      model = "Random forest",
      details = "100 trees, balanced sample benchmark"
    ),
  evaluate_classification(
    actual = y_test,
    predicted_class = majority_class,
    predicted_probability = majority_prob
  ) %>%
    mutate(
      model = "Majority baseline",
      details = "predict No"
    )
) %>%
  select(
    model,
    details,
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
  file.path(table_dir, "ch13_neural_network_performance.csv")
)

cat("\nNeural network chapter test performance:\n")
print(performance_table)

best_model <- performance_table %>%
  filter(model != "Majority baseline") %>%
  arrange(desc(auc)) %>%
  slice(1)

nn_row <- performance_table %>%
  filter(model == "Neural network") %>%
  slice(1)

# ----------------------------
# 11. Minimal figure
# ----------------------------

p_metrics <- performance_table %>%
  filter(model != "Majority baseline") %>%
  select(model, accuracy, sensitivity, specificity, precision, auc) %>%
  pivot_longer(
    cols = c(accuracy, sensitivity, specificity, precision, auc),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(metric = str_to_title(metric)) %>%
  ggplot(
    aes(x = model, y = value, fill = metric)
  ) +
  geom_col(position = "dodge") +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    limits = c(0, 1)
  ) +
  labs(
    title = "Neural Network Chapter Test Performance",
    subtitle = "Models were trained on the same balanced sample and evaluated on the full 2025 test set.",
    x = "Model",
    y = "Metric Value",
    fill = "Metric"
  )

safe_ggsave(
  plot_object = p_metrics,
  filename = "ch13_fig01_neural_network_metrics.png",
  width = 9,
  height = 5
)

# ----------------------------
# 12. Lightweight model record
# ----------------------------

nn_record <- list(
  chapter_name = "Chapter 13 - Neural Network",
  formula = deparse(nn_formula),
  data_summary = data_summary,
  tuning_results = nn_tuning_results,
  performance_table = performance_table,
  best_nn_settings = best_nn_settings,
  best_model = best_model,
  note = "The neural network, logistic regression benchmark, and random forest benchmark were trained on the same balanced sample. Evaluation used the full 2025 test set."
)

saveRDS(
  nn_record,
  file.path(model_dir, "ch13_neural_network_record.rds")
)

# ----------------------------
# 13. Report notes
# ----------------------------

report_notes <- c(
  "Chapter 13 Report Notes",
  "",
  "This chapter used a small neural network to predict pit_next_lap.",
  "",
  "The predictors were scaled before fitting the neural network. Scaling matters because neural networks train better when variables are on similar scales.",
  "",
  "The neural network was trained on a balanced sample from the training data to keep runtime manageable on a basic laptop. The final evaluation still used the full 2025 test set.",
  "",
  paste0(
    "The balanced training sample had ",
    nrow(x_train_nn),
    " rows, with ",
    sum(y_train_nn == "No"),
    " No cases and ",
    sum(y_train_nn == "Yes"),
    " Yes cases."
  ),
  "",
  paste0(
    "The neural network tuning grid tested size values ",
    paste(size_grid, collapse = ", "),
    " and decay values ",
    paste(decay_grid, collapse = ", "),
    "."
  ),
  "",
  paste0(
    "The selected neural network used size = ",
    best_size,
    " and decay = ",
    best_decay,
    "."
  ),
  "",
  paste0(
    "On the full 2025 test set, the neural network had accuracy = ",
    nn_row$accuracy,
    ", sensitivity = ",
    nn_row$sensitivity,
    ", specificity = ",
    nn_row$specificity,
    ", precision = ",
    nn_row$precision,
    ", and AUC = ",
    nn_row$auc,
    "."
  ),
  "",
  paste0(
    "The best model in this chapter by AUC was ",
    best_model$model,
    ", with AUC = ",
    best_model$auc,
    "."
  ),
  "",
  "Real-world decision: The neural network is useful as a flexible nonlinear comparison model. However, because it is less interpretable and was trained on a smaller balanced sample, it should be compared carefully against simpler models like logistic regression and stronger models like random forest or boosting."
)

writeLines(
  report_notes,
  file.path(table_dir, "ch13_report_notes.txt")
)

cat("\nReport notes:\n")
cat(paste(report_notes, collapse = "\n"))

# ----------------------------
# 14. Final confirmation
# ----------------------------

cat("\n\n14_neural_network.R ran successfully.\n")
cat("Chapter 13 tables saved to: ", table_dir, "\n", sep = "")
cat("Chapter 13 figures saved to: ", figure_dir, "\n", sep = "")
cat("Chapter 13 model record saved to: ", model_dir, "\n", sep = "")
cat("Report notes saved to: ", file.path(table_dir, "ch13_report_notes.txt"), "\n", sep = "")
cat("Neural network was trained on a balanced sample and evaluated on the full 2025 test set.\n")