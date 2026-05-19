# ============================================================
# MATH 4230 Capstone Project
# Script: scripts/05_mlr_lap_time.R
# Purpose: Chapter 4 - Multiple Linear Regression for lap time
# ============================================================

# ----------------------------
# 1. Run setup
# ----------------------------

source("scripts/00_setup.R")

# ----------------------------
# 2. Organized output folders
# ----------------------------

chapter_name <- "ch04_mlr"

table_dir <- file.path("results/tables", chapter_name)
model_dir <- file.path("results/models", chapter_name)
figure_dir <- file.path("figures", chapter_name)

dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# 3. Safe save helpers
# ----------------------------

safe_ggsave <- function(plot_object, filename, width = 8, height = 5, dpi = 300) {
  main_path <- file.path(figure_dir, filename)
  
  if (file.exists(main_path)) {
    try(unlink(main_path, force = TRUE), silent = TRUE)
  }
  
  result <- try(
    ggsave(
      filename = main_path,
      plot = plot_object,
      width = width,
      height = height,
      dpi = dpi,
      device = "png"
    ),
    silent = TRUE
  )
  
  main_ok <- file.exists(main_path) && file.info(main_path)$size > 0
  
  if (inherits(result, "try-error") || !main_ok) {
    alt_name <- paste0(
      tools::file_path_sans_ext(filename),
      "_updated_",
      format(Sys.time(), "%Y%m%d_%H%M%S"),
      ".png"
    )
    
    alt_path <- file.path(figure_dir, alt_name)
    
    ggsave(
      filename = alt_path,
      plot = plot_object,
      width = width,
      height = height,
      dpi = dpi,
      device = "png"
    )
    
    cat("Could not overwrite original figure. Saved alternate figure to:\n")
    cat(alt_path, "\n")
    
    return(alt_path)
  }
  
  cat("Saved figure:", main_path, "\n")
  return(main_path)
}

safe_save_rds <- function(object, filename) {
  main_path <- file.path(model_dir, filename)
  
  if (file.exists(main_path)) {
    try(unlink(main_path, force = TRUE), silent = TRUE)
  }
  
  result <- try(
    saveRDS(object, main_path, compress = "xz"),
    silent = TRUE
  )
  
  main_ok <- file.exists(main_path) && file.info(main_path)$size > 0
  
  if (inherits(result, "try-error") || !main_ok) {
    alt_name <- paste0(
      tools::file_path_sans_ext(filename),
      "_updated_",
      format(Sys.time(), "%Y%m%d_%H%M%S"),
      ".rds"
    )
    
    alt_path <- file.path(model_dir, alt_name)
    
    saveRDS(object, alt_path, compress = "xz")
    
    cat("Could not overwrite original RDS. Saved alternate RDS to:\n")
    cat(alt_path, "\n")
    
    return(alt_path)
  }
  
  cat("Saved RDS:", main_path, "\n")
  return(main_path)
}

make_light_lm_record <- function(fit, model_name, model_description, metrics_row) {
  list(
    model_name = model_name,
    description = model_description,
    formula = deparse(formula(fit)),
    coefficients = coef(fit),
    metrics = metrics_row,
    note = "This is a lightweight model record. Re-run scripts/05_mlr_lap_time.R to recreate the full lm object."
  )
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

relevel_if_present <- function(x, ref_level) {
  x <- as.factor(x)
  
  if (ref_level %in% levels(x)) {
    x <- relevel(x, ref = ref_level)
  }
  
  return(x)
}

mlr_data <- f1_train %>%
  mutate(
    compound = relevel_if_present(compound, "MEDIUM"),
    driver = as.factor(driver),
    race = as.factor(race),
    year_fct = as.factor(year)
  ) %>%
  select(
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

cat("Training rows used for MLR:\n")
print(nrow(mlr_data))

cat("\nVariables used for MLR:\n")
print(names(mlr_data))

# ----------------------------
# 6. Define model formulas
# ----------------------------

numeric_formula <- lap_time_s ~
  lap_number +
  tyre_life +
  normalized_tyre_life +
  race_progress +
  stint +
  position +
  position_change

compound_formula <- lap_time_s ~
  lap_number +
  tyre_life +
  normalized_tyre_life +
  race_progress +
  stint +
  position +
  position_change +
  compound +
  year_fct

full_formula <- lap_time_s ~
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

# ----------------------------
# 7. Fit MLR models
# ----------------------------

mlr_numeric <- lm(numeric_formula, data = mlr_data)

mlr_compound <- lm(compound_formula, data = mlr_data)

mlr_full <- lm(full_formula, data = mlr_data)

# ----------------------------
# 8. Helper functions
# ----------------------------

rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

mae <- function(actual, predicted) {
  mean(abs(actual - predicted), na.rm = TRUE)
}

model_metrics <- function(fit, data, model_name, model_description) {
  preds <- predict(fit, newdata = data)
  fit_summary <- summary(fit)
  
  tibble(
    model_name = model_name,
    description = model_description,
    n = nrow(data),
    parameters = length(coef(fit)),
    train_rmse = rmse(data$lap_time_s, preds),
    train_mae = mae(data$lap_time_s, preds),
    r_squared = fit_summary$r.squared,
    adj_r_squared = fit_summary$adj.r.squared,
    aic = AIC(fit),
    bic = BIC(fit)
  )
}

# ----------------------------
# 9. Model comparison table
# ----------------------------

model_comparison_raw <- bind_rows(
  model_metrics(
    fit = mlr_numeric,
    data = mlr_data,
    model_name = "Numeric MLR",
    model_description = "Uses only numeric lap-level predictors."
  ),
  model_metrics(
    fit = mlr_compound,
    data = mlr_data,
    model_name = "Numeric + Compound + Year MLR",
    model_description = "Adds tire compound dummy variables and year indicators."
  ),
  model_metrics(
    fit = mlr_full,
    data = mlr_data,
    model_name = "Full Context MLR",
    model_description = "Adds compound, driver, race, and year dummy variables."
  )
)

model_comparison <- model_comparison_raw %>%
  mutate(
    across(
      c(train_rmse, train_mae, r_squared, adj_r_squared, aic, bic),
      ~ round(.x, 4)
    )
  )

write_csv(
  model_comparison,
  file.path(table_dir, "ch04_mlr_model_comparison.csv")
)

cat("\nMLR model comparison using training data only:\n")
print(model_comparison)

# ----------------------------
# 10. Save coefficient tables
# ----------------------------

full_coef_table <- tidy(mlr_full, conf.int = TRUE) %>%
  mutate(
    estimate = round(estimate, 4),
    std.error = round(std.error, 4),
    statistic = round(statistic, 4),
    p_value = p.value,
    p_value_label = case_when(
      p.value < 0.001 ~ "<0.001",
      TRUE ~ as.character(round(p.value, 4))
    ),
    conf.low = round(conf.low, 4),
    conf.high = round(conf.high, 4)
  ) %>%
  select(
    term,
    estimate,
    std.error,
    statistic,
    p_value,
    p_value_label,
    conf.low,
    conf.high
  )

write_csv(
  full_coef_table,
  file.path(table_dir, "ch04_mlr_full_coefficient_table.csv")
)

main_numeric_coef_table <- tidy(mlr_full, conf.int = TRUE) %>%
  filter(
    term %in% c(
      "lap_number",
      "tyre_life",
      "normalized_tyre_life",
      "race_progress",
      "stint",
      "position",
      "position_change"
    )
  ) %>%
  mutate(
    estimate = round(estimate, 4),
    std.error = round(std.error, 4),
    statistic = round(statistic, 4),
    p_value = p.value,
    p_value_label = case_when(
      p.value < 0.001 ~ "<0.001",
      TRUE ~ as.character(round(p.value, 4))
    ),
    conf.low = round(conf.low, 4),
    conf.high = round(conf.high, 4)
  ) %>%
  select(
    term,
    estimate,
    std.error,
    statistic,
    p_value,
    p_value_label,
    conf.low,
    conf.high
  )

write_csv(
  main_numeric_coef_table,
  file.path(table_dir, "ch04_mlr_main_numeric_coefficients.csv")
)

cat("\nMain numeric coefficients from full MLR:\n")
print(main_numeric_coef_table)

compound_coef_table <- tidy(mlr_full, conf.int = TRUE) %>%
  filter(str_detect(term, "^compound")) %>%
  mutate(
    comparison = str_replace(term, "^compound", ""),
    reference_level = levels(mlr_data$compound)[1],
    estimate = round(estimate, 4),
    std.error = round(std.error, 4),
    statistic = round(statistic, 4),
    p_value = p.value,
    p_value_label = case_when(
      p.value < 0.001 ~ "<0.001",
      TRUE ~ as.character(round(p.value, 4))
    ),
    conf.low = round(conf.low, 4),
    conf.high = round(conf.high, 4)
  ) %>%
  select(
    term,
    comparison,
    reference_level,
    estimate,
    std.error,
    statistic,
    p_value,
    p_value_label,
    conf.low,
    conf.high
  )

write_csv(
  compound_coef_table,
  file.path(table_dir, "ch04_mlr_compound_coefficients.csv")
)

cat("\nCompound dummy-variable coefficients from full MLR:\n")
print(compound_coef_table)

# ----------------------------
# 11. Dummy variable notes
# ----------------------------

dummy_variable_notes <- tibble(
  categorical_variable = c("compound", "driver", "race", "year_fct"),
  number_of_levels = c(
    nlevels(mlr_data$compound),
    nlevels(mlr_data$driver),
    nlevels(mlr_data$race),
    nlevels(mlr_data$year_fct)
  ),
  reference_level = c(
    levels(mlr_data$compound)[1],
    levels(mlr_data$driver)[1],
    levels(mlr_data$race)[1],
    levels(mlr_data$year_fct)[1]
  ),
  interpretation = c(
    "Each compound coefficient compares that compound against the reference compound, holding the other predictors fixed.",
    "Each driver coefficient compares that driver against the reference driver, holding the other predictors fixed.",
    "Each race coefficient compares that race against the reference race, holding the other predictors fixed.",
    "Each year coefficient compares that year against the reference year, holding the other predictors fixed."
  )
)

write_csv(
  dummy_variable_notes,
  file.path(table_dir, "ch04_dummy_variable_notes.csv")
)

cat("\nDummy variable notes:\n")
print(dummy_variable_notes)

# ----------------------------
# 12. VIF / multicollinearity check
# ----------------------------

# VIF is checked on the compound + year model.
# This keeps the VIF table readable while still showing multicollinearity
# among numeric timing variables and compound/year dummy variables.

vif_model <- lm(compound_formula, data = mlr_data)

vif_raw <- car::vif(vif_model)

if (is.matrix(vif_raw)) {
  vif_table <- data.frame(
    term = rownames(vif_raw),
    vif_raw,
    row.names = NULL,
    check.names = FALSE
  ) %>%
    as_tibble() %>%
    rename(
      gvif = GVIF,
      df = Df,
      gvif_adjusted = `GVIF^(1/(2*Df))`
    ) %>%
    mutate(
      gvif = round(gvif, 4),
      gvif_adjusted = round(gvif_adjusted, 4)
    )
} else {
  vif_table <- tibble(
    term = names(vif_raw),
    vif = as.numeric(vif_raw),
    df = 1,
    gvif_adjusted = as.numeric(vif_raw)
  ) %>%
    mutate(
      vif = round(vif, 4),
      gvif_adjusted = round(gvif_adjusted, 4)
    )
}

write_csv(
  vif_table,
  file.path(table_dir, "ch04_vif_table.csv")
)

cat("\nVIF table:\n")
print(vif_table)

# ----------------------------
# 13. Subset selection
# ----------------------------

# Best subset selection is run on numeric predictors only.
# This keeps the subset selection readable and avoids creating a large
# dummy-variable search across driver and race categories.

subset_data <- mlr_data %>%
  select(
    lap_time_s,
    lap_number,
    tyre_life,
    normalized_tyre_life,
    race_progress,
    stint,
    position,
    position_change
  )

subset_predictors <- c(
  "lap_number",
  "tyre_life",
  "normalized_tyre_life",
  "race_progress",
  "stint",
  "position",
  "position_change"
)

subset_formula <- as.formula(
  paste("lap_time_s ~", paste(subset_predictors, collapse = " + "))
)

best_subset <- regsubsets(
  subset_formula,
  data = subset_data,
  nvmax = length(subset_predictors),
  method = "exhaustive"
)

subset_summary <- summary(best_subset)

subset_table <- tibble(
  model_size = 1:length(subset_predictors),
  adj_r_squared = subset_summary$adjr2,
  cp = subset_summary$cp,
  bic_regsubsets = subset_summary$bic
)

subset_model_details <- map_dfr(
  1:length(subset_predictors),
  function(i) {
    selected_vars <- names(coef(best_subset, id = i))[-1]
    
    formula_i <- as.formula(
      paste("lap_time_s ~", paste(selected_vars, collapse = " + "))
    )
    
    fit_i <- lm(formula_i, data = subset_data)
    preds_i <- predict(fit_i, newdata = subset_data)
    
    tibble(
      model_size = i,
      selected_predictors = paste(selected_vars, collapse = ", "),
      train_rmse = rmse(subset_data$lap_time_s, preds_i),
      train_mae = mae(subset_data$lap_time_s, preds_i),
      aic = AIC(fit_i),
      bic = BIC(fit_i)
    )
  }
)

subset_table <- subset_table %>%
  left_join(subset_model_details, by = "model_size") %>%
  mutate(
    across(
      c(adj_r_squared, cp, bic_regsubsets, train_rmse, train_mae, aic, bic),
      ~ round(.x, 4)
    )
  )

write_csv(
  subset_table,
  file.path(table_dir, "ch04_subset_selection_table.csv")
)

cat("\nSubset selection table:\n")
print(subset_table)

subset_predictor_table <- subset_table %>%
  select(model_size, selected_predictors)

write_csv(
  subset_predictor_table,
  file.path(table_dir, "ch04_subset_selected_predictors.csv")
)

# ----------------------------
# 14. AIC, BIC, and Mallows Cp summary
# ----------------------------

criterion_summary <- tibble(
  criterion = c("AIC", "BIC", "Mallows Cp"),
  best_model_size = c(
    subset_table$model_size[which.min(subset_table$aic)],
    subset_table$model_size[which.min(subset_table$bic)],
    subset_table$model_size[which.min(subset_table$cp)]
  ),
  selected_predictors = c(
    subset_table$selected_predictors[which.min(subset_table$aic)],
    subset_table$selected_predictors[which.min(subset_table$bic)],
    subset_table$selected_predictors[which.min(subset_table$cp)]
  ),
  note = c(
    "AIC selected the numeric subset with the lowest AIC.",
    "BIC selected the numeric subset with the lowest BIC.",
    "Mallows Cp selected the numeric subset with the lowest Cp."
  )
)

write_csv(
  criterion_summary,
  file.path(table_dir, "ch04_aic_bic_cp_summary.csv")
)

cat("\nAIC, BIC, and Cp summary:\n")
print(criterion_summary)

# ----------------------------
# 15. Save model summaries
# ----------------------------

full_model_summary <- glance(mlr_full) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))

numeric_model_summary <- glance(mlr_numeric) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))

compound_model_summary <- glance(mlr_compound) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))

write_csv(
  full_model_summary,
  file.path(table_dir, "ch04_mlr_full_model_summary.csv")
)

write_csv(
  numeric_model_summary,
  file.path(table_dir, "ch04_mlr_numeric_model_summary.csv")
)

write_csv(
  compound_model_summary,
  file.path(table_dir, "ch04_mlr_compound_model_summary.csv")
)

# ----------------------------
# 16. Diagnostic data
# ----------------------------

mlr_augmented <- augment(mlr_full) %>%
  select(
    lap_time_s,
    .fitted,
    .resid,
    .hat,
    .cooksd,
    .std.resid
  )

write_csv(
  mlr_augmented,
  file.path(table_dir, "ch04_mlr_full_fitted_residuals.csv")
)

residual_summary <- mlr_augmented %>%
  summarise(
    n = n(),
    mean_residual = round(mean(.resid, na.rm = TRUE), 4),
    sd_residual = round(sd(.resid, na.rm = TRUE), 4),
    min_residual = round(min(.resid, na.rm = TRUE), 4),
    q1_residual = round(quantile(.resid, 0.25, na.rm = TRUE), 4),
    median_residual = round(median(.resid, na.rm = TRUE), 4),
    q3_residual = round(quantile(.resid, 0.75, na.rm = TRUE), 4),
    max_residual = round(max(.resid, na.rm = TRUE), 4)
  )

write_csv(
  residual_summary,
  file.path(table_dir, "ch04_mlr_residual_summary.csv")
)

# ----------------------------
# 17. Diagnostic plots
# ----------------------------

set.seed(4230)

residual_limits <- quantile(mlr_augmented$.resid, c(0.01, 0.99), na.rm = TRUE)
lap_time_limits <- quantile(mlr_augmented$lap_time_s, c(0.01, 0.99), na.rm = TRUE)
fitted_limits <- quantile(mlr_augmented$.fitted, c(0.01, 0.99), na.rm = TRUE)

diagnostic_data <- mlr_augmented %>%
  filter(
    .resid >= residual_limits[1],
    .resid <= residual_limits[2]
  ) %>%
  sample_n(size = min(12000, nrow(.)))

observed_fitted_data <- mlr_augmented %>%
  filter(
    lap_time_s >= lap_time_limits[1],
    lap_time_s <= lap_time_limits[2],
    .fitted >= fitted_limits[1],
    .fitted <= fitted_limits[2]
  ) %>%
  sample_n(size = min(12000, nrow(.)))

p_resid_fitted <- ggplot(diagnostic_data, aes(x = .fitted, y = .resid)) +
  geom_point(alpha = 0.18, color = "gray35") +
  geom_hline(yintercept = 0, color = f1_red, linewidth = 1) +
  geom_smooth(
    method = "loess",
    formula = y ~ x,
    se = FALSE,
    color = "black",
    linewidth = 0.8
  ) +
  labs(
    title = "MLR Residuals vs Fitted Values",
    subtitle = "The curved pattern suggests the linear model is missing some structure.",
    x = "Fitted Lap Time",
    y = "Residual"
  )

safe_ggsave(
  plot_object = p_resid_fitted,
  filename = "ch04_fig01_residuals_vs_fitted.png"
)

p_observed_fitted <- ggplot(observed_fitted_data, aes(x = .fitted, y = lap_time_s)) +
  geom_point(alpha = 0.18, color = "gray35") +
  geom_abline(intercept = 0, slope = 1, color = f1_red, linewidth = 1) +
  labs(
    title = "Observed vs Fitted Lap Times for MLR",
    subtitle = "The model follows the general direction, but many laps remain far from the perfect-fit line.",
    x = "Fitted Lap Time",
    y = "Observed Lap Time"
  )

safe_ggsave(
  plot_object = p_observed_fitted,
  filename = "ch04_fig02_observed_vs_fitted.png"
)

p_resid_hist <- ggplot(diagnostic_data, aes(x = .resid)) +
  geom_histogram(bins = 40, fill = f1_red, color = "white", linewidth = 0.25) +
  labs(
    title = "Distribution of MLR Residuals",
    subtitle = "The right tail shows that some laps are much slower than the model expects.",
    x = "Residual",
    y = "Count"
  ) +
  scale_y_continuous(labels = scales::comma)

safe_ggsave(
  plot_object = p_resid_hist,
  filename = "ch04_fig03_residual_histogram.png"
)

qq_data <- diagnostic_data %>%
  sample_n(size = min(5000, nrow(.)))

p_qq <- ggplot(qq_data, aes(sample = .resid)) +
  stat_qq(alpha = 0.35, color = "gray35") +
  stat_qq_line(color = f1_red, linewidth = 1) +
  labs(
    title = "QQ Plot of MLR Residuals",
    subtitle = "The strong tail departures show that normality is not a good fit for all residuals.",
    x = "Theoretical Quantiles",
    y = "Sample Quantiles"
  )

safe_ggsave(
  plot_object = p_qq,
  filename = "ch04_fig04_qq_plot.png"
)

comparison_plot_data <- model_comparison %>%
  select(model_name, train_rmse, adj_r_squared) %>%
  pivot_longer(
    cols = c(train_rmse, adj_r_squared),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = case_when(
      metric == "train_rmse" ~ "Training RMSE",
      metric == "adj_r_squared" ~ "Adjusted R-squared",
      TRUE ~ metric
    )
  )

p_model_comparison <- ggplot(
  comparison_plot_data,
  aes(x = model_name, y = value, fill = metric)
) +
  geom_col(width = 0.7) +
  coord_flip() +
  facet_wrap(~ metric, scales = "free_x") +
  labs(
    title = "MLR Model Comparison",
    subtitle = "The full context model performs best, but the explained variation is still limited.",
    x = "Model",
    y = "Value",
    fill = "Metric"
  ) +
  scale_fill_manual(
    values = c(
      "Training RMSE" = f1_gray,
      "Adjusted R-squared" = f1_red
    )
  ) +
  theme(legend.position = "none")

safe_ggsave(
  plot_object = p_model_comparison,
  filename = "ch04_fig05_model_comparison.png",
  width = 9,
  height = 5
)

subset_delta_table <- subset_table %>%
  mutate(
    delta_aic = aic - min(aic),
    delta_bic = bic - min(bic),
    delta_cp = cp - min(cp)
  ) %>%
  select(
    model_size,
    selected_predictors,
    delta_aic,
    delta_bic,
    delta_cp
  )

write_csv(
  subset_delta_table,
  file.path(table_dir, "ch04_subset_selection_delta_table.csv")
)

subset_plot_data <- subset_delta_table %>%
  select(model_size, delta_aic, delta_bic, delta_cp) %>%
  pivot_longer(
    cols = c(delta_aic, delta_bic, delta_cp),
    names_to = "criterion",
    values_to = "delta_value"
  ) %>%
  mutate(
    criterion = case_when(
      criterion == "delta_aic" ~ "Delta AIC",
      criterion == "delta_bic" ~ "Delta BIC",
      criterion == "delta_cp" ~ "Delta Mallows Cp",
      TRUE ~ criterion
    )
  )

p_subset <- ggplot(subset_plot_data, aes(x = model_size, y = delta_value, color = criterion)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  facet_wrap(~ criterion, scales = "free_y") +
  labs(
    title = "Subset Selection Criteria",
    subtitle = "Values are shown relative to each criterion's best model, so lower is better.",
    x = "Number of Predictors",
    y = "Difference from Best Value",
    color = "Criterion"
  ) +
  scale_color_manual(
    values = c(
      "Delta AIC" = f1_red,
      "Delta BIC" = f1_dark,
      "Delta Mallows Cp" = f1_gray
    )
  ) +
  theme(legend.position = "none")

safe_ggsave(
  plot_object = p_subset,
  filename = "ch04_fig06_subset_selection_criteria.png",
  width = 9,
  height = 5
)

p_vif <- vif_table %>%
  mutate(term = fct_reorder(term, gvif_adjusted)) %>%
  ggplot(aes(x = term, y = gvif_adjusted)) +
  geom_col(fill = f1_red) +
  coord_flip() +
  labs(
    title = "Adjusted VIF Values",
    subtitle = "Lap number and race progress have the largest VIF values, showing timing-variable overlap.",
    x = "Predictor",
    y = "Adjusted VIF"
  )

safe_ggsave(
  plot_object = p_vif,
  filename = "ch04_fig07_vif_values.png",
  width = 8,
  height = 5
)

if (nrow(compound_coef_table) > 0) {
  p_compound_effects <- compound_coef_table %>%
    mutate(comparison = fct_reorder(comparison, estimate)) %>%
    ggplot(aes(x = comparison, y = estimate)) +
    geom_col(fill = f1_red) +
    geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.2) +
    coord_flip() +
    labs(
      title = "Compound Dummy-Variable Effects",
      subtitle = paste0("Each estimate is compared against the reference compound: ", levels(mlr_data$compound)[1], "."),
      x = "Compound",
      y = "Estimated Difference in Lap Time"
    )
  
  safe_ggsave(
    plot_object = p_compound_effects,
    filename = "ch04_fig08_compound_effects.png",
    width = 8,
    height = 5
  )
}

# ----------------------------
# 18. Save lightweight model records
# ----------------------------

numeric_record <- make_light_lm_record(
  fit = mlr_numeric,
  model_name = "Numeric MLR",
  model_description = "Uses only numeric lap-level predictors.",
  metrics_row = model_comparison %>% filter(model_name == "Numeric MLR")
)

compound_record <- make_light_lm_record(
  fit = mlr_compound,
  model_name = "Numeric + Compound + Year MLR",
  model_description = "Adds tire compound dummy variables and year indicators.",
  metrics_row = model_comparison %>% filter(model_name == "Numeric + Compound + Year MLR")
)

full_record <- make_light_lm_record(
  fit = mlr_full,
  model_name = "Full Context MLR",
  model_description = "Adds compound, driver, race, and year dummy variables.",
  metrics_row = model_comparison %>% filter(model_name == "Full Context MLR")
)

subset_record <- list(
  model_name = "Best subset numeric search",
  formula = deparse(subset_formula),
  subset_table = subset_table,
  criterion_summary = criterion_summary,
  note = "This is a lightweight subset-selection record. Re-run scripts/05_mlr_lap_time.R to recreate the full regsubsets object."
)

safe_save_rds(
  object = numeric_record,
  filename = "ch04_mlr_numeric_record.rds"
)

safe_save_rds(
  object = compound_record,
  filename = "ch04_mlr_compound_year_record.rds"
)

safe_save_rds(
  object = full_record,
  filename = "ch04_mlr_full_context_record.rds"
)

safe_save_rds(
  object = subset_record,
  filename = "ch04_best_subset_numeric_record.rds"
)

# ----------------------------
# 19. Updated interpretation notes
# ----------------------------

best_model_by_adj_r2 <- model_comparison %>%
  arrange(desc(adj_r_squared)) %>%
  slice(1)

best_model_by_aic <- model_comparison %>%
  arrange(aic) %>%
  slice(1)

best_model_by_bic <- model_comparison %>%
  arrange(bic) %>%
  slice(1)

highest_vif <- vif_table %>%
  arrange(desc(gvif_adjusted)) %>%
  slice(1)

second_highest_vif <- vif_table %>%
  arrange(desc(gvif_adjusted)) %>%
  slice(2)

aic_choice <- criterion_summary %>%
  filter(criterion == "AIC") %>%
  slice(1)

bic_choice <- criterion_summary %>%
  filter(criterion == "BIC") %>%
  slice(1)

cp_choice <- criterion_summary %>%
  filter(criterion == "Mallows Cp") %>%
  slice(1)

interpretation <- tibble(
  item = c(
    "Main result",
    "Why MLR improved on SLR",
    "Dummy variable interpretation",
    "Compound interpretation",
    "Model fit takeaway",
    "Best model by adjusted R-squared",
    "Best model by AIC",
    "Best model by BIC",
    "VIF result",
    "VIF interpretation",
    "Subset selection result",
    "Diagnostic plot takeaway",
    "QQ plot takeaway",
    "Chapter 4 real-world decision",
    "Test set note",
    "Saved model note"
  ),
  note = c(
    paste0(
      "The Full Context MLR was the best Chapter 4 model, with training RMSE = ",
      best_model_by_adj_r2$train_rmse,
      ", training MAE = ",
      best_model_by_adj_r2$train_mae,
      ", and adjusted R-squared = ",
      best_model_by_adj_r2$adj_r_squared,
      "."
    ),
    "MLR improved on simple linear regression because it included race context, tire compound, driver, race, year, and several timing-related predictors instead of using only one predictor.",
    paste0(
      "The categorical variables were handled with dummy variables. For example, compound effects are interpreted relative to the reference compound, ",
      levels(mlr_data$compound)[1],
      ", while holding the other predictors fixed."
    ),
    "The compound coefficients should be interpreted as controlled differences in average lap time, not as pure tire-performance effects, because compound choice is also connected to race strategy and track conditions.",
    paste0(
      "Even the best MLR model only explained about ",
      round(100 * best_model_by_adj_r2$adj_r_squared, 1),
      "% of the variation in lap time, so lap time is still mostly explained by factors not fully captured in this dataset."
    ),
    paste0(
      best_model_by_adj_r2$model_name,
      " had the highest adjusted R-squared among the three MLR models."
    ),
    paste0(
      best_model_by_aic$model_name,
      " had the lowest AIC among the three MLR models."
    ),
    paste0(
      best_model_by_bic$model_name,
      " had the lowest BIC among the three MLR models."
    ),
    paste0(
      "The largest adjusted VIF was for ",
      highest_vif$term,
      " at ",
      highest_vif$gvif_adjusted,
      ". The next largest was ",
      second_highest_vif$term,
      " at ",
      second_highest_vif$gvif_adjusted,
      "."
    ),
    "The VIF results show that timing variables overlap. This is not surprising because lap number, race progress, stint, and tire life all describe where the car is in the race or stint. This means individual coefficients should be interpreted carefully.",
    paste0(
      "For the numeric-only subset search, AIC chose ",
      aic_choice$best_model_size,
      " predictors, BIC chose ",
      bic_choice$best_model_size,
      " predictors, and Mallows Cp chose ",
      cp_choice$best_model_size,
      " predictors."
    ),
    "The residual plot shows a curved pattern instead of a completely random cloud, which suggests that a linear model misses some nonlinear race structure.",
    "The QQ plot and residual histogram show strong tail behavior, meaning the residuals are not normally distributed. This is reasonable because some laps are unusual due to pits, traffic, incidents, tire effects, or other race conditions.",
    "A race strategist should treat MLR as a useful interpretable baseline, but not as the final decision model. It is better than SLR, but the low adjusted R-squared and residual patterns show that more flexible methods are worth trying.",
    "The 2025 test set was not used in this script. Final test evaluation should be saved for later model comparison.",
    "This script saves lightweight model records instead of full lm objects to avoid oversized RDS files and file-writing issues."
  )
)

write_csv(
  interpretation,
  file.path(table_dir, "ch04_mlr_interpretation_notes.csv")
)

report_notes <- c(
  "Chapter 4 Report Notes",
  "",
  paste0(
    "The main multiple linear regression model was the Full Context MLR. It used numeric lap-level predictors and categorical predictors for compound, driver, race, and year. This model had training RMSE = ",
    best_model_by_adj_r2$train_rmse,
    ", training MAE = ",
    best_model_by_adj_r2$train_mae,
    ", and adjusted R-squared = ",
    best_model_by_adj_r2$adj_r_squared,
    "."
  ),
  "",
  paste0(
    "The Full Context MLR performed better than the simpler numeric-only model and the compound/year model. However, the adjusted R-squared was still only about ",
    round(100 * best_model_by_adj_r2$adj_r_squared, 1),
    "%, so the model explains only a small part of the total variation in lap time."
  ),
  "",
  paste0(
    "Dummy variables were used for categorical predictors. The reference compound was ",
    levels(mlr_data$compound)[1],
    ". Each compound coefficient compares that compound against the reference compound while holding the other predictors fixed."
  ),
  "",
  paste0(
    "The VIF check showed the largest adjusted VIF for ",
    highest_vif$term,
    " at ",
    highest_vif$gvif_adjusted,
    ". This means the timing-related variables have overlap, so the individual coefficients should be interpreted carefully."
  ),
  "",
  paste0(
    "For numeric-only subset selection, AIC selected ",
    aic_choice$best_model_size,
    " predictors, BIC selected ",
    bic_choice$best_model_size,
    " predictors, and Mallows Cp selected ",
    cp_choice$best_model_size,
    " predictors."
  ),
  "",
  "The diagnostic plots showed that the residuals are not perfectly random or normally distributed. The residuals vs fitted plot had a curved pattern, and the QQ plot showed strong tail behavior. This suggests that MLR is useful as an interpretable baseline, but it is probably too simple to fully capture Formula 1 lap-time behavior.",
  "",
  "Real-world decision: A strategist should not rely on MLR alone for final race strategy. It is useful for understanding broad patterns, but later models should be tested because lap time is affected by many nonlinear and race-specific factors."
)

writeLines(
  report_notes,
  file.path(table_dir, "ch04_report_notes.txt")
)

cat("\nUpdated interpretation notes:\n")
print(interpretation)

# ----------------------------
# 20. Final confirmation
# ----------------------------

cat("\n05_mlr_lap_time.R ran successfully.\n")
cat("Chapter 4 tables saved to: ", table_dir, "\n", sep = "")
cat("Chapter 4 figures saved to: ", figure_dir, "\n", sep = "")
cat("Chapter 4 lightweight model records saved to: ", model_dir, "\n", sep = "")
cat("Report notes saved to: ", file.path(table_dir, "ch04_report_notes.txt"), "\n", sep = "")
cat("The 2025 test set was not used in this script.\n")