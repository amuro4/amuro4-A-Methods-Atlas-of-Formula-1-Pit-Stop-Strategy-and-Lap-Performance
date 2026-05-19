# ============================================================
# MATH 4230 Capstone Project
# Script: scripts/10_tree_ensembles.R
# Purpose: Chapter 9 - Tree Ensembles
# ============================================================

# ----------------------------
# 1. Run setup
# ----------------------------

source("scripts/00_setup.R")

# ----------------------------
# 2. Extra packages for tree ensembles
# ----------------------------

packages <- c(
  "tidyverse",
  "randomForest",
  "gbm",
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

chapter_name <- "ch09_tree_ensembles"

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

get_class_probs_rf <- function(fit, new_data) {
  probs <- predict(fit, newdata = new_data, type = "prob")
  
  if (!("Yes" %in% colnames(probs))) {
    return(rep(0, nrow(new_data)))
  }
  
  as.numeric(probs[, "Yes"])
}

rf_importance_to_long <- function(fit, task_name, model_name) {
  randomForest::importance(fit) %>%
    as.data.frame() %>%
    rownames_to_column("variable") %>%
    as_tibble() %>%
    pivot_longer(
      cols = -variable,
      names_to = "importance_type",
      values_to = "importance"
    ) %>%
    mutate(
      task = task_name,
      model = model_name
    ) %>%
    select(task, model, variable, importance_type, importance) %>%
    arrange(task, model, desc(importance))
}

paste_top_vars <- function(data, task_name, n_terms = 5) {
  text <- data %>%
    filter(task == task_name) %>%
    slice_head(n = n_terms) %>%
    mutate(label = paste0(variable, " (", round(importance, 2), ")")) %>%
    pull(label) %>%
    paste(collapse = ", ")
  
  if (text == "") {
    text <- "No variable importance available"
  }
  
  text
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
# 6. Prepare modeling data
# ----------------------------

ensemble_data <- bind_rows(
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
    year
  ) %>%
  drop_na()

train_ensemble <- ensemble_data %>%
  filter(split == "train") %>%
  select(-split)

test_ensemble <- ensemble_data %>%
  filter(split == "test") %>%
  select(-split)

cat("\nRows used for tree ensembles:\n")
print(table(ensemble_data$split))

cat("\nTraining class balance:\n")
print(table(train_ensemble$pit_next_lap_factor))

cat("\nTesting class balance:\n")
print(table(test_ensemble$pit_next_lap_factor))

data_summary <- tibble(
  item = c(
    "Training rows",
    "Testing rows",
    "Training No pit next lap",
    "Training Yes pit next lap",
    "Testing No pit next lap",
    "Testing Yes pit next lap"
  ),
  value = c(
    nrow(train_ensemble),
    nrow(test_ensemble),
    sum(train_ensemble$pit_next_lap_factor == "No"),
    sum(train_ensemble$pit_next_lap_factor == "Yes"),
    sum(test_ensemble$pit_next_lap_factor == "No"),
    sum(test_ensemble$pit_next_lap_factor == "Yes")
  )
)

write_csv(
  data_summary,
  file.path(table_dir, "ch09_tree_ensemble_data_summary.csv")
)

# ----------------------------
# 7. Define model formulas
# ----------------------------

reg_formula <- lap_time_s ~
  lap_number +
  tyre_life +
  normalized_tyre_life +
  race_progress +
  stint +
  position +
  position_change +
  compound +
  year

class_formula <- pit_next_lap_factor ~
  tyre_life +
  normalized_tyre_life +
  race_progress +
  stint +
  position +
  position_change +
  compound +
  year

class_gbm_formula <- pit_next_lap_num ~
  tyre_life +
  normalized_tyre_life +
  race_progress +
  stint +
  position +
  position_change +
  compound +
  year

p_reg <- length(attr(terms(reg_formula), "term.labels"))
p_class <- length(attr(terms(class_formula), "term.labels"))

# Bagging uses all predictors at each split.
bagging_mtry_reg <- p_reg
bagging_mtry_class <- p_class

# Random forest uses a smaller random subset of predictors.
rf_mtry_reg <- max(1, floor(p_reg / 3))
rf_mtry_class <- max(1, floor(sqrt(p_class)))

ntree_ensemble <- 200
gbm_trees <- 600

model_settings <- tibble(
  item = c(
    "Regression predictors",
    "Classification predictors",
    "Bagging mtry regression",
    "Bagging mtry classification",
    "Random forest mtry regression",
    "Random forest mtry classification",
    "Random forest / bagging trees",
    "Boosting max trees"
  ),
  value = c(
    p_reg,
    p_class,
    bagging_mtry_reg,
    bagging_mtry_class,
    rf_mtry_reg,
    rf_mtry_class,
    ntree_ensemble,
    gbm_trees
  )
)

write_csv(
  model_settings,
  file.path(table_dir, "ch09_tree_ensemble_model_settings.csv")
)

cat("\nModel settings:\n")
print(model_settings)

# ----------------------------
# 8. Fit bagging models
# ----------------------------

set.seed(4230)

bagging_reg <- randomForest(
  formula = reg_formula,
  data = train_ensemble,
  ntree = ntree_ensemble,
  mtry = bagging_mtry_reg,
  importance = TRUE
)

set.seed(4230)

bagging_class <- randomForest(
  formula = class_formula,
  data = train_ensemble,
  ntree = ntree_ensemble,
  mtry = bagging_mtry_class,
  importance = TRUE
)

cat("\nBagging models finished.\n")

# ----------------------------
# 9. Fit random forest models
# ----------------------------

set.seed(4230)

rf_reg <- randomForest(
  formula = reg_formula,
  data = train_ensemble,
  ntree = ntree_ensemble,
  mtry = rf_mtry_reg,
  importance = TRUE
)

set.seed(4230)

rf_class <- randomForest(
  formula = class_formula,
  data = train_ensemble,
  ntree = ntree_ensemble,
  mtry = rf_mtry_class,
  importance = TRUE
)

cat("\nRandom forest models finished.\n")

# ----------------------------
# 10. Fit boosting models
# ----------------------------

set.seed(4230)

boost_reg <- gbm(
  formula = reg_formula,
  data = train_ensemble,
  distribution = "gaussian",
  n.trees = gbm_trees,
  interaction.depth = 3,
  shrinkage = 0.05,
  n.minobsinnode = 250,
  bag.fraction = 0.70,
  train.fraction = 1,
  cv.folds = 5,
  verbose = FALSE
)

reg_boost_best_iter <- gbm.perf(
  boost_reg,
  method = "cv",
  plot.it = FALSE
)

if (is.na(reg_boost_best_iter) || length(reg_boost_best_iter) == 0) {
  reg_boost_best_iter <- gbm_trees
}

set.seed(4230)

boost_class <- gbm(
  formula = class_gbm_formula,
  data = train_ensemble,
  distribution = "bernoulli",
  n.trees = gbm_trees,
  interaction.depth = 3,
  shrinkage = 0.05,
  n.minobsinnode = 250,
  bag.fraction = 0.70,
  train.fraction = 1,
  cv.folds = 5,
  verbose = FALSE
)

class_boost_best_iter <- gbm.perf(
  boost_class,
  method = "cv",
  plot.it = FALSE
)

if (is.na(class_boost_best_iter) || length(class_boost_best_iter) == 0) {
  class_boost_best_iter <- gbm_trees
}

cat("\nBoosting models finished.\n")
cat("Best boosting trees for regression:", reg_boost_best_iter, "\n")
cat("Best boosting trees for classification:", class_boost_best_iter, "\n")

boosting_summary <- tibble(
  task = c("Regression", "Classification"),
  model = c("Boosting", "Boosting"),
  best_trees = c(reg_boost_best_iter, class_boost_best_iter),
  max_trees = gbm_trees,
  interaction_depth = 3,
  shrinkage = 0.05,
  min_obs_node = 250,
  bag_fraction = 0.70,
  cv_folds = 5
)

write_csv(
  boosting_summary,
  file.path(table_dir, "ch09_boosting_summary.csv")
)

# ----------------------------
# 11. OOB error summaries
# ----------------------------

bagging_reg_oob_rmse <- sqrt(tail(bagging_reg$mse, 1))
bagging_reg_oob_rsq <- tail(bagging_reg$rsq, 1)

rf_reg_oob_rmse <- sqrt(tail(rf_reg$mse, 1))
rf_reg_oob_rsq <- tail(rf_reg$rsq, 1)

bagging_class_oob_error <- tail(bagging_class$err.rate[, "OOB"], 1)
rf_class_oob_error <- tail(rf_class$err.rate[, "OOB"], 1)

oob_summary <- tibble(
  task = c("Regression", "Regression", "Classification", "Classification"),
  model = c("Bagging", "Random forest", "Bagging", "Random forest"),
  oob_rmse = c(
    bagging_reg_oob_rmse,
    rf_reg_oob_rmse,
    NA_real_,
    NA_real_
  ),
  oob_r_squared = c(
    bagging_reg_oob_rsq,
    rf_reg_oob_rsq,
    NA_real_,
    NA_real_
  ),
  oob_error_rate = c(
    NA_real_,
    NA_real_,
    bagging_class_oob_error,
    rf_class_oob_error
  ),
  oob_accuracy = c(
    NA_real_,
    NA_real_,
    1 - bagging_class_oob_error,
    1 - rf_class_oob_error
  )
) %>%
  mutate(
    across(
      c(oob_rmse, oob_r_squared, oob_error_rate, oob_accuracy),
      ~ round(.x, 4)
    )
  )

write_csv(
  oob_summary,
  file.path(table_dir, "ch09_oob_error_summary.csv")
)

cat("\nOOB error summary:\n")
print(oob_summary)

# ----------------------------
# 12. Test-set predictions
# ----------------------------

# Regression predictions
pred_bag_reg <- predict(bagging_reg, newdata = test_ensemble)
pred_rf_reg <- predict(rf_reg, newdata = test_ensemble)
pred_boost_reg <- predict(
  boost_reg,
  newdata = test_ensemble,
  n.trees = reg_boost_best_iter
)

mean_baseline_pred <- rep(
  mean(train_ensemble$lap_time_s, na.rm = TRUE),
  nrow(test_ensemble)
)

# Classification probabilities
prob_bag_class <- get_class_probs_rf(bagging_class, test_ensemble)
prob_rf_class <- get_class_probs_rf(rf_class, test_ensemble)
prob_boost_class <- predict(
  boost_class,
  newdata = test_ensemble,
  n.trees = class_boost_best_iter,
  type = "response"
)

train_yes_rate <- mean(train_ensemble$pit_next_lap_num == 1, na.rm = TRUE)
prob_majority_baseline <- rep(train_yes_rate, nrow(test_ensemble))

class_bag <- factor(
  if_else(prob_bag_class >= 0.50, "Yes", "No"),
  levels = c("No", "Yes")
)

class_rf <- factor(
  if_else(prob_rf_class >= 0.50, "Yes", "No"),
  levels = c("No", "Yes")
)

class_boost <- factor(
  if_else(prob_boost_class >= 0.50, "Yes", "No"),
  levels = c("No", "Yes")
)

class_majority_baseline <- factor(
  rep("No", nrow(test_ensemble)),
  levels = c("No", "Yes")
)

# ----------------------------
# 13. Test performance tables
# ----------------------------

regression_performance <- tibble(
  task = "Regression",
  model = c("Bagging", "Random forest", "Boosting", "Mean baseline"),
  test_rmse = c(
    rmse(test_ensemble$lap_time_s, pred_bag_reg),
    rmse(test_ensemble$lap_time_s, pred_rf_reg),
    rmse(test_ensemble$lap_time_s, pred_boost_reg),
    rmse(test_ensemble$lap_time_s, mean_baseline_pred)
  ),
  test_mae = c(
    mae(test_ensemble$lap_time_s, pred_bag_reg),
    mae(test_ensemble$lap_time_s, pred_rf_reg),
    mae(test_ensemble$lap_time_s, pred_boost_reg),
    mae(test_ensemble$lap_time_s, mean_baseline_pred)
  ),
  test_r_squared = c(
    test_r2(test_ensemble$lap_time_s, pred_bag_reg),
    test_r2(test_ensemble$lap_time_s, pred_rf_reg),
    test_r2(test_ensemble$lap_time_s, pred_boost_reg),
    test_r2(test_ensemble$lap_time_s, mean_baseline_pred)
  )
) %>%
  mutate(
    across(c(test_rmse, test_mae, test_r_squared), ~ round(.x, 4))
  ) %>%
  arrange(test_rmse)

classification_performance <- bind_rows(
  evaluate_classification(
    actual = test_ensemble$pit_next_lap_factor,
    predicted_class = class_bag,
    predicted_probability = prob_bag_class
  ) %>%
    mutate(
      task = "Classification",
      model = "Bagging"
    ),
  evaluate_classification(
    actual = test_ensemble$pit_next_lap_factor,
    predicted_class = class_rf,
    predicted_probability = prob_rf_class
  ) %>%
    mutate(
      task = "Classification",
      model = "Random forest"
    ),
  evaluate_classification(
    actual = test_ensemble$pit_next_lap_factor,
    predicted_class = class_boost,
    predicted_probability = prob_boost_class
  ) %>%
    mutate(
      task = "Classification",
      model = "Boosting"
    ),
  evaluate_classification(
    actual = test_ensemble$pit_next_lap_factor,
    predicted_class = class_majority_baseline,
    predicted_probability = prob_majority_baseline
  ) %>%
    mutate(
      task = "Classification",
      model = "Majority baseline"
    )
) %>%
  select(
    task,
    model,
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

combined_performance <- bind_rows(
  regression_performance %>%
    mutate(
      accuracy = NA_real_,
      sensitivity = NA_real_,
      specificity = NA_real_,
      precision = NA_real_,
      auc = NA_real_,
      true_negative = NA_real_,
      false_positive = NA_real_,
      false_negative = NA_real_,
      true_positive = NA_real_
    ),
  classification_performance %>%
    mutate(
      test_rmse = NA_real_,
      test_mae = NA_real_,
      test_r_squared = NA_real_
    )
)

write_csv(
  regression_performance,
  file.path(table_dir, "ch09_regression_ensemble_performance.csv")
)

write_csv(
  classification_performance,
  file.path(table_dir, "ch09_classification_ensemble_performance.csv")
)

write_csv(
  combined_performance,
  file.path(table_dir, "ch09_tree_ensemble_performance.csv")
)

cat("\nRegression ensemble performance:\n")
print(regression_performance)

cat("\nClassification ensemble performance:\n")
print(classification_performance)

# ----------------------------
# 14. Save predictions
# ----------------------------

regression_predictions <- tibble(
  actual_lap_time_s = test_ensemble$lap_time_s,
  bagging = pred_bag_reg,
  random_forest = pred_rf_reg,
  boosting = pred_boost_reg,
  mean_baseline = mean_baseline_pred
)

classification_predictions <- tibble(
  actual = test_ensemble$pit_next_lap_factor,
  bagging_probability = prob_bag_class,
  bagging_class = class_bag,
  random_forest_probability = prob_rf_class,
  random_forest_class = class_rf,
  boosting_probability = prob_boost_class,
  boosting_class = class_boost,
  majority_baseline_probability = prob_majority_baseline,
  majority_baseline_class = class_majority_baseline
)

write_csv(
  regression_predictions,
  file.path(table_dir, "ch09_regression_ensemble_predictions.csv")
)

write_csv(
  classification_predictions,
  file.path(table_dir, "ch09_classification_ensemble_predictions.csv")
)

# ----------------------------
# 15. Variable importance
# ----------------------------

rf_importance <- bind_rows(
  rf_importance_to_long(rf_reg, "Regression", "Random forest"),
  rf_importance_to_long(rf_class, "Classification", "Random forest")
)

write_csv(
  rf_importance,
  file.path(table_dir, "ch09_random_forest_variable_importance.csv")
)

reg_boost_importance <- summary(
  boost_reg,
  n.trees = reg_boost_best_iter,
  plotit = FALSE
) %>%
  as_tibble() %>%
  mutate(
    task = "Regression",
    model = "Boosting"
  ) %>%
  rename(
    variable = var,
    importance = rel.inf
  ) %>%
  select(task, model, variable, importance)

class_boost_importance <- summary(
  boost_class,
  n.trees = class_boost_best_iter,
  plotit = FALSE
) %>%
  as_tibble() %>%
  mutate(
    task = "Classification",
    model = "Boosting"
  ) %>%
  rename(
    variable = var,
    importance = rel.inf
  ) %>%
  select(task, model, variable, importance)

boost_importance <- bind_rows(
  reg_boost_importance,
  class_boost_importance
) %>%
  arrange(task, desc(importance))

write_csv(
  boost_importance,
  file.path(table_dir, "ch09_boosting_variable_importance.csv")
)

cat("\nRandom forest variable importance:\n")
print(rf_importance)

cat("\nBoosting variable importance:\n")
print(boost_importance)

# ----------------------------
# 16. Importance plots
# ----------------------------

rf_primary_importance <- rf_importance %>%
  filter(
    (task == "Regression" & importance_type == "IncNodePurity") |
      (task == "Classification" & importance_type == "MeanDecreaseGini")
  ) %>%
  group_by(task) %>%
  slice_max(importance, n = 10, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(variable = fct_reorder(variable, importance))

p_rf_importance <- ggplot(
  rf_primary_importance,
  aes(x = variable, y = importance)
) +
  geom_col(fill = f1_red) +
  coord_flip() +
  facet_wrap(~ task, scales = "free") +
  labs(
    title = "Random Forest Variable Importance",
    subtitle = "Regression uses IncNodePurity; classification uses MeanDecreaseGini.",
    x = "Variable",
    y = "Importance"
  )

safe_ggsave(
  plot_object = p_rf_importance,
  filename = "ch09_fig01_random_forest_importance.png",
  width = 9,
  height = 5
)

boost_plot_data <- boost_importance %>%
  group_by(task) %>%
  slice_max(importance, n = 10, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(variable = fct_reorder(variable, importance))

p_boost_importance <- ggplot(
  boost_plot_data,
  aes(x = variable, y = importance)
) +
  geom_col(fill = f1_red) +
  coord_flip() +
  facet_wrap(~ task, scales = "free") +
  labs(
    title = "Boosting Variable Importance",
    subtitle = "Relative influence from GBM models.",
    x = "Variable",
    y = "Relative Influence"
  )

safe_ggsave(
  plot_object = p_boost_importance,
  filename = "ch09_fig02_boosting_importance.png",
  width = 9,
  height = 5
)

# ----------------------------
# 17. Performance plots
# ----------------------------

p_reg_perf <- regression_performance %>%
  ggplot(
    aes(x = reorder(model, test_rmse), y = test_rmse)
  ) +
  geom_col(fill = f1_red, width = 0.7) +
  geom_text(
    aes(label = round(test_rmse, 2)),
    hjust = -0.10,
    size = 3.8
  ) +
  coord_flip() +
  labs(
    title = "Regression Ensemble Test RMSE",
    subtitle = "Models were evaluated on the 2025 test set.",
    x = "Model",
    y = "Test RMSE"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)))

safe_ggsave(
  plot_object = p_reg_perf,
  filename = "ch09_fig03_regression_ensemble_rmse.png",
  width = 8,
  height = 5
)

p_class_perf <- classification_performance %>%
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
    title = "Classification Ensemble Test Performance",
    subtitle = "Classification metrics use a 0.50 probability threshold.",
    x = "Model",
    y = "Metric Value",
    fill = "Metric"
  )

safe_ggsave(
  plot_object = p_class_perf,
  filename = "ch09_fig04_classification_ensemble_metrics.png",
  width = 9,
  height = 5
)

# ----------------------------
# 18. Save model record
# ----------------------------

tree_ensemble_record <- list(
  chapter_name = "Chapter 9 - Tree Ensembles",
  regression_formula = deparse(reg_formula),
  classification_formula = deparse(class_formula),
  data_summary = data_summary,
  model_settings = model_settings,
  oob_summary = oob_summary,
  boosting_summary = boosting_summary,
  regression_performance = regression_performance,
  classification_performance = classification_performance,
  rf_importance = rf_importance,
  boost_importance = boost_importance,
  bagging_reg = bagging_reg,
  bagging_class = bagging_class,
  rf_reg = rf_reg,
  rf_class = rf_class,
  boost_reg = boost_reg,
  boost_class = boost_class,
  note = "Models were fit on training data and evaluated on the 2025 test set. pit_stop was excluded from classification models."
)

saveRDS(
  tree_ensemble_record,
  file.path(model_dir, "ch09_tree_ensemble_record.rds")
)

# ----------------------------
# 19. Interpretation notes
# ----------------------------

best_reg_model <- regression_performance %>%
  filter(model != "Mean baseline") %>%
  arrange(test_rmse) %>%
  slice(1)

best_class_model <- classification_performance %>%
  filter(model != "Majority baseline") %>%
  arrange(desc(auc)) %>%
  slice(1)

best_bag_reg_oob <- oob_summary %>%
  filter(task == "Regression", model == "Bagging") %>%
  slice(1)

best_rf_reg_oob <- oob_summary %>%
  filter(task == "Regression", model == "Random forest") %>%
  slice(1)

bag_class_oob <- oob_summary %>%
  filter(task == "Classification", model == "Bagging") %>%
  slice(1)

rf_class_oob <- oob_summary %>%
  filter(task == "Classification", model == "Random forest") %>%
  slice(1)

top_rf_reg_vars <- paste_top_vars(rf_primary_importance, "Regression", 5)
top_rf_class_vars <- paste_top_vars(rf_primary_importance, "Classification", 5)

top_boost_reg_vars <- paste_top_vars(boost_plot_data, "Regression", 5)
top_boost_class_vars <- paste_top_vars(boost_plot_data, "Classification", 5)

interpretation <- tibble(
  item = c(
    "Main method",
    "Bagging OOB result",
    "Random forest OOB result",
    "Boosting setup",
    "Best regression ensemble",
    "Best classification ensemble",
    "Random forest regression importance",
    "Random forest classification importance",
    "Boosting regression importance",
    "Boosting classification importance",
    "Leakage decision",
    "Main takeaway",
    "Test set note"
  ),
  note = c(
    "This chapter fit bagging, random forest, and boosting models for both lap_time_s and pit_next_lap.",
    paste0(
      "Bagging OOB RMSE for regression was ",
      best_bag_reg_oob$oob_rmse,
      ". Bagging OOB error rate for classification was ",
      bag_class_oob$oob_error_rate,
      "."
    ),
    paste0(
      "Random forest OOB RMSE for regression was ",
      best_rf_reg_oob$oob_rmse,
      ". Random forest OOB error rate for classification was ",
      rf_class_oob$oob_error_rate,
      "."
    ),
    paste0(
      "Boosting used up to ",
      gbm_trees,
      " trees. The CV-selected number of trees was ",
      reg_boost_best_iter,
      " for regression and ",
      class_boost_best_iter,
      " for classification."
    ),
    paste0(
      "The best regression ensemble was ",
      best_reg_model$model,
      " with test RMSE = ",
      best_reg_model$test_rmse,
      ", test MAE = ",
      best_reg_model$test_mae,
      ", and test R-squared = ",
      best_reg_model$test_r_squared,
      "."
    ),
    paste0(
      "The best classification ensemble by AUC was ",
      best_class_model$model,
      " with accuracy = ",
      best_class_model$accuracy,
      ", sensitivity = ",
      best_class_model$sensitivity,
      ", specificity = ",
      best_class_model$specificity,
      ", precision = ",
      best_class_model$precision,
      ", and AUC = ",
      best_class_model$auc,
      "."
    ),
    paste0(
      "The top random forest regression variables were: ",
      top_rf_reg_vars,
      "."
    ),
    paste0(
      "The top random forest classification variables were: ",
      top_rf_class_vars,
      "."
    ),
    paste0(
      "The top boosting regression variables were: ",
      top_boost_reg_vars,
      "."
    ),
    paste0(
      "The top boosting classification variables were: ",
      top_boost_class_vars,
      "."
    ),
    "pit_stop was excluded from the classification models to avoid leakage.",
    "Tree ensembles are usually more stable than a single tree. The main goal is to check whether averaging or boosting improves prediction compared with the single-tree results.",
    "The 2025 test set was used only after model fitting to compare ensemble performance."
  )
)

write_csv(
  interpretation,
  file.path(table_dir, "ch09_tree_ensemble_interpretation_notes.csv")
)

report_notes <- c(
  "Chapter 9 Report Notes",
  "",
  "This chapter used tree ensembles for both project tasks. Bagging, random forest, and boosting were fit for lap_time_s regression and pit_next_lap classification.",
  "",
  "The models used lap-level predictors, compound, and year. The variable pit_stop was not used in the classification models because it could create leakage.",
  "",
  paste0(
    "For bagging, the regression OOB RMSE was ",
    best_bag_reg_oob$oob_rmse,
    ", and the classification OOB error rate was ",
    bag_class_oob$oob_error_rate,
    "."
  ),
  "",
  paste0(
    "For random forest, the regression OOB RMSE was ",
    best_rf_reg_oob$oob_rmse,
    ", and the classification OOB error rate was ",
    rf_class_oob$oob_error_rate,
    "."
  ),
  "",
  paste0(
    "For boosting, the CV-selected number of trees was ",
    reg_boost_best_iter,
    " for regression and ",
    class_boost_best_iter,
    " for classification."
  ),
  "",
  paste0(
    "For the regression task, the best ensemble model was ",
    best_reg_model$model,
    ". It had test RMSE = ",
    best_reg_model$test_rmse,
    ", test MAE = ",
    best_reg_model$test_mae,
    ", and test R-squared = ",
    best_reg_model$test_r_squared,
    "."
  ),
  "",
  paste0(
    "For the classification task, the best ensemble model by AUC was ",
    best_class_model$model,
    ". It had accuracy = ",
    best_class_model$accuracy,
    ", sensitivity = ",
    best_class_model$sensitivity,
    ", specificity = ",
    best_class_model$specificity,
    ", precision = ",
    best_class_model$precision,
    ", and AUC = ",
    best_class_model$auc,
    "."
  ),
  "",
  paste0(
    "The top random forest regression variables were: ",
    top_rf_reg_vars,
    "."
  ),
  "",
  paste0(
    "The top random forest classification variables were: ",
    top_rf_class_vars,
    "."
  ),
  "",
  "Real-world decision: Tree ensembles are less interpretable than a single tree, but they are usually more stable. This chapter checks whether bagging, random forests, and boosting improve performance after the single decision tree struggled for lap-time prediction."
)

writeLines(
  report_notes,
  file.path(table_dir, "ch09_report_notes.txt")
)

cat("\nInterpretation notes:\n")
print(interpretation)

# ----------------------------
# 20. Final confirmation
# ----------------------------

cat("\n10_tree_ensembles.R ran successfully.\n")
cat("Chapter 9 tables saved to: ", table_dir, "\n", sep = "")
cat("Chapter 9 figures saved to: ", figure_dir, "\n", sep = "")
cat("Chapter 9 model record saved to: ", model_dir, "\n", sep = "")
cat("Report notes saved to: ", file.path(table_dir, "ch09_report_notes.txt"), "\n", sep = "")
cat("The 2025 test set was used only after model fitting.\n")