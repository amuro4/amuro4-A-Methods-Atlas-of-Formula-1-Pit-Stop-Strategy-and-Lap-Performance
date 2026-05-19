# ============================================================
# MATH 4230 Capstone Project
# Script: scripts/09_decision_trees.R
# Purpose: Chapter 8 - Decision Trees
# ============================================================

# ----------------------------
# 1. Run setup
# ----------------------------

source("scripts/00_setup.R")

# ----------------------------
# 2. Extra packages for trees
# ----------------------------

packages <- c(
  "tidyverse",
  "rpart",
  "rpart.plot",
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

chapter_name <- "ch08_decision_trees"

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

safe_rpart_plot <- function(tree_object, filename, title_text, extra_value = 101) {
  main_path <- file.path(figure_dir, filename)
  
  if (file.exists(main_path)) {
    try(unlink(main_path, force = TRUE), silent = TRUE)
  }
  
  png(
    filename = main_path,
    width = 2200,
    height = 1400,
    res = 200
  )
  
  rpart.plot::rpart.plot(
    tree_object,
    type = 2,
    extra = extra_value,
    under = TRUE,
    fallen.leaves = TRUE,
    faclen = 0,
    cex = 0.65,
    main = title_text
  )
  
  dev.off()
  
  cat("Saved tree plot:", main_path, "\n")
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

choose_cp_values <- function(tree_fit) {
  cp_table <- as_tibble(tree_fit$cptable)
  
  min_row <- cp_table %>%
    slice_min(xerror, n = 1, with_ties = FALSE)
  
  one_se_threshold <- min_row$xerror + min_row$xstd
  
  one_se_row <- cp_table %>%
    filter(xerror <= one_se_threshold) %>%
    slice(1)
  
  list(
    cp_min = min_row$CP,
    cp_1se = one_se_row$CP,
    min_xerror = min_row$xerror,
    one_se_threshold = one_se_threshold
  )
}

get_tree_size <- function(tree_fit, tree_name, tree_version, cp_value) {
  frame <- tree_fit$frame
  
  tibble(
    tree = tree_name,
    tree_version = tree_version,
    cp = cp_value,
    internal_nodes = sum(frame$var != "<leaf>"),
    terminal_nodes = sum(frame$var == "<leaf>"),
    total_nodes = nrow(frame),
    has_splits = sum(frame$var != "<leaf>") > 0
  )
}

get_split_summary <- function(tree_fit, tree_name, tree_version) {
  frame <- tree_fit$frame
  
  split_rows <- frame %>%
    as_tibble(rownames = "node") %>%
    filter(var != "<leaf>") %>%
    mutate(
      node = as.integer(node),
      depth = floor(log2(node))
    )
  
  if (nrow(split_rows) == 0) {
    return(
      tibble(
        tree = tree_name,
        tree_version = tree_version,
        variable = "No splits",
        split_count = 0,
        first_depth = NA_real_,
        mean_depth = NA_real_
      )
    )
  }
  
  split_rows %>%
    group_by(var) %>%
    summarise(
      split_count = n(),
      first_depth = min(depth, na.rm = TRUE),
      mean_depth = mean(depth, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    rename(variable = var) %>%
    mutate(
      tree = tree_name,
      tree_version = tree_version
    ) %>%
    select(tree, tree_version, variable, split_count, first_depth, mean_depth) %>%
    arrange(first_depth, desc(split_count), variable)
}

get_variable_importance <- function(tree_fit, tree_name, tree_version) {
  importance <- tree_fit$variable.importance
  
  if (is.null(importance)) {
    return(
      tibble(
        tree = tree_name,
        tree_version = tree_version,
        variable = "No variable importance",
        importance = NA_real_
      )
    )
  }
  
  tibble(
    variable = names(importance),
    importance = as.numeric(importance)
  ) %>%
    mutate(
      tree = tree_name,
      tree_version = tree_version
    ) %>%
    select(tree, tree_version, variable, importance) %>%
    arrange(desc(importance))
}

paste_top_variables <- function(importance_table, tree_name, n_terms = 5) {
  text <- importance_table %>%
    filter(tree == tree_name) %>%
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

tree_data <- bind_rows(
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

train_tree <- tree_data %>%
  filter(split == "train") %>%
  select(-split)

test_tree <- tree_data %>%
  filter(split == "test") %>%
  select(-split)

cat("\nRows used for decision trees:\n")
print(table(tree_data$split))

cat("\nTraining class balance:\n")
print(table(train_tree$pit_next_lap_factor))

cat("\nTesting class balance:\n")
print(table(test_tree$pit_next_lap_factor))

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
    nrow(train_tree),
    nrow(test_tree),
    sum(train_tree$pit_next_lap_factor == "No"),
    sum(train_tree$pit_next_lap_factor == "Yes"),
    sum(test_tree$pit_next_lap_factor == "No"),
    sum(test_tree$pit_next_lap_factor == "Yes")
  )
)

write_csv(
  data_summary,
  file.path(table_dir, "ch08_tree_data_summary.csv")
)

# ----------------------------
# 7. Define tree formulas
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

# ----------------------------
# 8. Fit full trees
# ----------------------------

set.seed(4230)

reg_tree_full <- rpart(
  formula = reg_formula,
  data = train_tree,
  method = "anova",
  control = rpart.control(
    cp = 0.0005,
    xval = 5,
    minbucket = 250,
    maxdepth = 5
  )
)

set.seed(4230)

class_tree_full <- rpart(
  formula = class_formula,
  data = train_tree,
  method = "class",
  parms = list(split = "gini"),
  control = rpart.control(
    cp = 0.0005,
    xval = 5,
    minbucket = 250,
    maxdepth = 5
  )
)

cat("\nFull regression tree CP table:\n")
printcp(reg_tree_full)

cat("\nFull classification tree CP table:\n")
printcp(class_tree_full)

# ----------------------------
# 9. Tune/prune trees using CP
# ----------------------------

reg_cp <- choose_cp_values(reg_tree_full)
class_cp <- choose_cp_values(class_tree_full)

reg_tree_min <- prune(reg_tree_full, cp = reg_cp$cp_min)
reg_tree_1se <- prune(reg_tree_full, cp = reg_cp$cp_1se)

class_tree_min <- prune(class_tree_full, cp = class_cp$cp_min)
class_tree_1se <- prune(class_tree_full, cp = class_cp$cp_1se)

cp_summary <- tibble(
  tree = c("Regression", "Regression", "Classification", "Classification"),
  cp_type = c("cp.min", "cp.1se", "cp.min", "cp.1se"),
  cp = c(
    reg_cp$cp_min,
    reg_cp$cp_1se,
    class_cp$cp_min,
    class_cp$cp_1se
  )
) %>%
  mutate(cp = round(cp, 8))

write_csv(
  cp_summary,
  file.path(table_dir, "ch08_cp_summary.csv")
)

cat("\nCP tuning summary:\n")
print(cp_summary)

tree_size_summary <- bind_rows(
  get_tree_size(reg_tree_full, "Regression tree", "Full tree", NA_real_),
  get_tree_size(reg_tree_min, "Regression tree", "Pruned cp.min", reg_cp$cp_min),
  get_tree_size(reg_tree_1se, "Regression tree", "Pruned cp.1se", reg_cp$cp_1se),
  get_tree_size(class_tree_full, "Classification tree", "Full tree", NA_real_),
  get_tree_size(class_tree_min, "Classification tree", "Pruned cp.min", class_cp$cp_min),
  get_tree_size(class_tree_1se, "Classification tree", "Pruned cp.1se", class_cp$cp_1se)
) %>%
  mutate(cp = round(cp, 8))

write_csv(
  tree_size_summary,
  file.path(table_dir, "ch08_tree_size_summary.csv")
)

cat("\nTree size summary:\n")
print(tree_size_summary)

# ----------------------------
# 10. Regression tree test performance
# ----------------------------

reg_pred_full <- predict(reg_tree_full, newdata = test_tree)
reg_pred_min <- predict(reg_tree_min, newdata = test_tree)
reg_pred_1se <- predict(reg_tree_1se, newdata = test_tree)

mean_baseline_pred <- rep(mean(train_tree$lap_time_s, na.rm = TRUE), nrow(test_tree))

regression_performance <- tibble(
  task = "Regression",
  model = "Regression tree",
  tree_version = c("Full tree", "Pruned cp.min", "Pruned cp.1se", "Mean baseline"),
  test_rmse = c(
    rmse(test_tree$lap_time_s, reg_pred_full),
    rmse(test_tree$lap_time_s, reg_pred_min),
    rmse(test_tree$lap_time_s, reg_pred_1se),
    rmse(test_tree$lap_time_s, mean_baseline_pred)
  ),
  test_mae = c(
    mae(test_tree$lap_time_s, reg_pred_full),
    mae(test_tree$lap_time_s, reg_pred_min),
    mae(test_tree$lap_time_s, reg_pred_1se),
    mae(test_tree$lap_time_s, mean_baseline_pred)
  ),
  test_r_squared = c(
    test_r2(test_tree$lap_time_s, reg_pred_full),
    test_r2(test_tree$lap_time_s, reg_pred_min),
    test_r2(test_tree$lap_time_s, reg_pred_1se),
    test_r2(test_tree$lap_time_s, mean_baseline_pred)
  )
) %>%
  mutate(
    across(c(test_rmse, test_mae, test_r_squared), ~ round(.x, 4))
  ) %>%
  arrange(test_rmse)

write_csv(
  regression_performance,
  file.path(table_dir, "ch08_regression_tree_test_performance.csv")
)

cat("\nRegression tree test performance:\n")
print(regression_performance)

baseline_reg <- regression_performance %>%
  filter(tree_version == "Mean baseline") %>%
  slice(1)

best_reg_overall <- regression_performance %>%
  arrange(test_rmse) %>%
  slice(1)

best_reg_split_tree <- regression_performance %>%
  filter(tree_version %in% c("Full tree", "Pruned cp.min")) %>%
  arrange(test_rmse) %>%
  slice(1)

reg_tree_beats_baseline <- best_reg_split_tree$test_rmse < baseline_reg$test_rmse

# ----------------------------
# 11. Classification tree test performance
# ----------------------------

get_class_probs <- function(tree_fit, new_data) {
  probs <- predict(tree_fit, newdata = new_data, type = "prob")
  
  if (!("Yes" %in% colnames(probs))) {
    return(rep(0, nrow(new_data)))
  }
  
  as.numeric(probs[, "Yes"])
}

class_prob_full <- get_class_probs(class_tree_full, test_tree)
class_prob_min <- get_class_probs(class_tree_min, test_tree)
class_prob_1se <- get_class_probs(class_tree_1se, test_tree)

class_pred_full <- factor(
  if_else(class_prob_full >= 0.50, "Yes", "No"),
  levels = c("No", "Yes")
)

class_pred_min <- factor(
  if_else(class_prob_min >= 0.50, "Yes", "No"),
  levels = c("No", "Yes")
)

class_pred_1se <- factor(
  if_else(class_prob_1se >= 0.50, "Yes", "No"),
  levels = c("No", "Yes")
)

classification_performance <- bind_rows(
  evaluate_classification(
    actual = test_tree$pit_next_lap_factor,
    predicted_class = class_pred_full,
    predicted_probability = class_prob_full
  ) %>%
    mutate(
      task = "Classification",
      model = "Classification tree",
      tree_version = "Full tree"
    ),
  evaluate_classification(
    actual = test_tree$pit_next_lap_factor,
    predicted_class = class_pred_min,
    predicted_probability = class_prob_min
  ) %>%
    mutate(
      task = "Classification",
      model = "Classification tree",
      tree_version = "Pruned cp.min"
    ),
  evaluate_classification(
    actual = test_tree$pit_next_lap_factor,
    predicted_class = class_pred_1se,
    predicted_probability = class_prob_1se
  ) %>%
    mutate(
      task = "Classification",
      model = "Classification tree",
      tree_version = "Pruned cp.1se"
    )
) %>%
  select(
    task,
    model,
    tree_version,
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
  classification_performance,
  file.path(table_dir, "ch08_classification_tree_test_performance.csv")
)

cat("\nClassification tree test performance:\n")
print(classification_performance)

best_class_tree <- classification_performance %>%
  arrange(desc(auc)) %>%
  slice(1)

# ----------------------------
# 12. Save prediction tables
# ----------------------------

regression_predictions <- tibble(
  actual_lap_time_s = test_tree$lap_time_s,
  full_tree = reg_pred_full,
  pruned_cp_min = reg_pred_min,
  pruned_cp_1se = reg_pred_1se,
  mean_baseline = mean_baseline_pred
)

classification_predictions <- tibble(
  actual = test_tree$pit_next_lap_factor,
  full_tree_probability = class_prob_full,
  full_tree_class = class_pred_full,
  pruned_cp_min_probability = class_prob_min,
  pruned_cp_min_class = class_pred_min,
  pruned_cp_1se_probability = class_prob_1se,
  pruned_cp_1se_class = class_pred_1se
)

write_csv(
  regression_predictions,
  file.path(table_dir, "ch08_regression_tree_predictions.csv")
)

write_csv(
  classification_predictions,
  file.path(table_dir, "ch08_classification_tree_predictions.csv")
)

combined_performance <- bind_rows(
  regression_performance %>%
    mutate(
      accuracy = NA_real_,
      sensitivity = NA_real_,
      specificity = NA_real_,
      precision = NA_real_,
      auc = NA_real_
    ),
  classification_performance %>%
    mutate(
      test_rmse = NA_real_,
      test_mae = NA_real_,
      test_r_squared = NA_real_
    )
)

write_csv(
  combined_performance,
  file.path(table_dir, "ch08_tree_test_performance.csv")
)

# ----------------------------
# 13. Interpret top splits
# ----------------------------

# For regression, the cp.min tree is plotted because cp.1se is only a stump.
# For classification, the full tree is used because it had the best AUC.

split_summary <- bind_rows(
  get_split_summary(reg_tree_min, "Regression tree", "Pruned cp.min"),
  get_split_summary(class_tree_full, "Classification tree", "Full tree")
)

variable_importance <- bind_rows(
  get_variable_importance(reg_tree_min, "Regression tree", "Pruned cp.min"),
  get_variable_importance(class_tree_full, "Classification tree", "Full tree")
)

write_csv(
  split_summary,
  file.path(table_dir, "ch08_tree_split_summary.csv")
)

write_csv(
  variable_importance,
  file.path(table_dir, "ch08_tree_variable_importance.csv")
)

cat("\nTree split summary:\n")
print(split_summary)

cat("\nTree variable importance:\n")
print(variable_importance)

reg_root_split <- reg_tree_min$frame$var[1]
class_root_split <- class_tree_full$frame$var[1]

top_reg_vars <- paste_top_variables(variable_importance, "Regression tree", 5)
top_class_vars <- paste_top_variables(variable_importance, "Classification tree", 5)

# ----------------------------
# 14. Figures
# ----------------------------

safe_rpart_plot(
  tree_object = reg_tree_min,
  filename = "ch08_fig01_regression_tree.png",
  title_text = "Regression Tree for Lap Time",
  extra_value = 101
)

safe_rpart_plot(
  tree_object = class_tree_full,
  filename = "ch08_fig02_classification_tree.png",
  title_text = "Classification Tree for Pit Next Lap",
  extra_value = 104
)

reg_cp_table <- as_tibble(reg_tree_full$cptable) %>%
  mutate(tree = "Regression tree")

class_cp_table <- as_tibble(class_tree_full$cptable) %>%
  mutate(tree = "Classification tree")

cp_plot_data <- bind_rows(
  reg_cp_table,
  class_cp_table
)

write_csv(
  cp_plot_data,
  file.path(table_dir, "ch08_cp_curve_data.csv")
)

p_cp <- ggplot(cp_plot_data, aes(x = nsplit, y = xerror)) +
  geom_line(color = f1_red, linewidth = 1) +
  geom_point(color = f1_red, size = 2) +
  geom_errorbar(
    aes(
      ymin = xerror - xstd,
      ymax = xerror + xstd
    ),
    width = 0.15,
    color = "gray35"
  ) +
  facet_wrap(~ tree, scales = "free_y") +
  labs(
    title = "Decision Tree CP Tuning Curves",
    subtitle = "Lower cross-validated error is better. Error bars show one standard error.",
    x = "Number of Splits",
    y = "Cross-Validated Relative Error"
  )

safe_ggsave(
  plot_object = p_cp,
  filename = "ch08_fig03_cp_tuning_curves.png",
  width = 9,
  height = 5
)

p_reg_perf <- regression_performance %>%
  mutate(
    tree_version = factor(
      tree_version,
      levels = c("Mean baseline", "Pruned cp.1se", "Full tree", "Pruned cp.min")
    )
  ) %>%
  ggplot(
    aes(x = reorder(tree_version, test_rmse), y = test_rmse)
  ) +
  geom_col(fill = f1_red, width = 0.7) +
  geom_text(
    aes(label = round(test_rmse, 2)),
    hjust = -0.10,
    size = 3.8
  ) +
  coord_flip() +
  labs(
    title = "Regression Tree Test RMSE",
    subtitle = "The split trees did not beat the mean baseline on the 2025 test set.",
    x = "Tree Version",
    y = "Test RMSE"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)))

safe_ggsave(
  plot_object = p_reg_perf,
  filename = "ch08_fig04_regression_tree_rmse.png",
  width = 8,
  height = 5
)

p_class_perf <- classification_performance %>%
  select(tree_version, accuracy, sensitivity, specificity, precision, auc) %>%
  pivot_longer(
    cols = c(accuracy, sensitivity, specificity, precision, auc),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = str_to_title(metric)
  ) %>%
  ggplot(
    aes(x = tree_version, y = value, fill = metric)
  ) +
  geom_col(position = "dodge") +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    limits = c(0, 1)
  ) +
  labs(
    title = "Classification Tree Test Performance",
    subtitle = "Classification metrics use a 0.50 probability threshold.",
    x = "Tree Version",
    y = "Metric Value",
    fill = "Metric"
  )

safe_ggsave(
  plot_object = p_class_perf,
  filename = "ch08_fig05_classification_tree_metrics.png",
  width = 9,
  height = 5
)

# ----------------------------
# 15. Save model record
# ----------------------------

decision_tree_record <- list(
  chapter_name = "Chapter 8 - Decision Trees",
  regression_formula = deparse(reg_formula),
  classification_formula = deparse(class_formula),
  data_summary = data_summary,
  cp_summary = cp_summary,
  tree_size_summary = tree_size_summary,
  regression_performance = regression_performance,
  classification_performance = classification_performance,
  split_summary = split_summary,
  variable_importance = variable_importance,
  regression_tree_pruned_cp_min = reg_tree_min,
  classification_tree_full = class_tree_full,
  note = "Regression split trees did not beat the mean baseline. The full classification tree had the best AUC. pit_stop was excluded."
)

saveRDS(
  decision_tree_record,
  file.path(model_dir, "ch08_decision_tree_record.rds")
)

# ----------------------------
# 16. Interpretation notes
# ----------------------------

regression_result_note <- if (reg_tree_beats_baseline) {
  paste0(
    "The best split regression tree was ",
    best_reg_split_tree$tree_version,
    " with test RMSE = ",
    best_reg_split_tree$test_rmse,
    ", which beat the mean baseline RMSE = ",
    baseline_reg$test_rmse,
    "."
  )
} else {
  paste0(
    "The split regression trees did not beat the mean baseline. The mean baseline had test RMSE = ",
    baseline_reg$test_rmse,
    ", while the best split tree had test RMSE = ",
    best_reg_split_tree$test_rmse,
    ". The cp.1se tree matched the baseline because it pruned back to a stump."
  )
}

interpretation <- tibble(
  item = c(
    "Main method",
    "Regression tree",
    "Classification tree",
    "Pruning method",
    "Regression test result",
    "Best classification tree",
    "Regression top split",
    "Classification top split",
    "Regression important variables",
    "Classification important variables",
    "Leakage decision",
    "Main takeaway",
    "Test set note"
  ),
  note = c(
    "This chapter fit a regression tree for lap_time_s and a classification tree for pit_next_lap.",
    "The regression tree used lap-level predictors, compound, and year to predict lap_time_s.",
    "The classification tree used lap-level predictors, compound, and year to predict pit_next_lap.",
    "The trees were tuned using the rpart complexity parameter CP. Full, cp.min, and cp.1se versions were saved and compared.",
    regression_result_note,
    paste0(
      "The best classification tree by AUC was ",
      best_class_tree$tree_version,
      " with accuracy = ",
      best_class_tree$accuracy,
      ", sensitivity = ",
      best_class_tree$sensitivity,
      ", specificity = ",
      best_class_tree$specificity,
      ", precision = ",
      best_class_tree$precision,
      ", and AUC = ",
      best_class_tree$auc,
      "."
    ),
    paste0(
      "The root split for the plotted regression tree was ",
      reg_root_split,
      "."
    ),
    paste0(
      "The root split for the best classification tree was ",
      class_root_split,
      "."
    ),
    paste0(
      "The most important regression tree variables were: ",
      top_reg_vars,
      "."
    ),
    paste0(
      "The most important classification tree variables were: ",
      top_class_vars,
      "."
    ),
    "pit_stop was excluded from the classification tree to avoid leakage.",
    "Decision trees are useful because they create readable if-then rules and can capture nonlinear splits. In this run, the classification tree was useful, but the regression tree did not generalize well.",
    "The 2025 test set was used only after tree fitting and pruning decisions."
  )
)

write_csv(
  interpretation,
  file.path(table_dir, "ch08_decision_tree_interpretation_notes.csv")
)

report_notes <- c(
  "Chapter 8 Report Notes",
  "",
  "This chapter used decision trees for both project tasks. A regression tree predicted lap_time_s, and a classification tree predicted pit_next_lap.",
  "",
  "The trees used lap-level predictors, compound, and year. The current-lap variable pit_stop was not used in the classification tree because it could create leakage.",
  "",
  "Tree pruning was handled using the rpart complexity parameter CP. The script saved the full tree, the cp.min pruned tree, and the cp.1se pruned tree.",
  "",
  regression_result_note,
  "",
  paste0(
    "For the classification task, the best tree version by AUC was ",
    best_class_tree$tree_version,
    ". It had accuracy = ",
    best_class_tree$accuracy,
    ", sensitivity = ",
    best_class_tree$sensitivity,
    ", specificity = ",
    best_class_tree$specificity,
    ", precision = ",
    best_class_tree$precision,
    ", and AUC = ",
    best_class_tree$auc,
    "."
  ),
  "",
  paste0(
    "The plotted regression tree's root split was ",
    reg_root_split,
    ". The most important regression variables were: ",
    top_reg_vars,
    "."
  ),
  "",
  paste0(
    "The best classification tree's root split was ",
    class_root_split,
    ". The most important classification variables were: ",
    top_class_vars,
    "."
  ),
  "",
  "Real-world decision: Decision trees are easier to explain than many flexible methods because they give if-then rules. The classification tree was useful because it caught more pit-next-lap cases than a conservative classifier. The regression tree was not useful for lap-time prediction here because the split trees performed worse than the mean baseline. The next chapter should check whether tree ensembles improve prediction."
)

writeLines(
  report_notes,
  file.path(table_dir, "ch08_report_notes.txt")
)

cat("\nInterpretation notes:\n")
print(interpretation)

# ----------------------------
# 17. Final confirmation
# ----------------------------

cat("\n09_decision_trees.R ran successfully.\n")
cat("Chapter 8 tables saved to: ", table_dir, "\n", sep = "")
cat("Chapter 8 figures saved to: ", figure_dir, "\n", sep = "")
cat("Chapter 8 model record saved to: ", model_dir, "\n", sep = "")
cat("Report notes saved to: ", file.path(table_dir, "ch08_report_notes.txt"), "\n", sep = "")
cat("The 2025 test set was used only after tree fitting and pruning decisions.\n")