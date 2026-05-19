# ============================================================
# MATH 4230 Capstone Project
# Script: scripts/04_slr_lap_time.R
# Purpose: Chapter 3 - Simple Linear Regression for lap time
# ============================================================

# ----------------------------
# 1. Run setup
# ----------------------------

source("scripts/00_setup.R")

# ----------------------------
# 2. Organized output folders
# ----------------------------

table_dir <- "results/tables/ch03_slr"
model_dir <- "results/models/ch03_slr"
figure_dir <- "figures/ch03_slr"

dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# 3. Load training data only
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
# 4. Keep only needed variables
# ----------------------------

slr_data <- f1_train %>%
  select(lap_time_s, tyre_life, race_progress) %>%
  drop_na()

cat("Training rows used for SLR:\n")
print(nrow(slr_data))

# ----------------------------
# 5. Fit simple linear regression models
# ----------------------------

slr_tyre_life <- lm(lap_time_s ~ tyre_life, data = slr_data)

slr_race_progress <- lm(lap_time_s ~ race_progress, data = slr_data)

# ----------------------------
# 6. Helper functions
# ----------------------------

rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

mae <- function(actual, predicted) {
  mean(abs(actual - predicted), na.rm = TRUE)
}

# ----------------------------
# 7. Compare SLR models using training data only
# ----------------------------

model_comparison <- tibble(
  model = c(
    "lap_time_s ~ tyre_life",
    "lap_time_s ~ race_progress"
  ),
  chapter_use = c(
    "Chosen for Chapter 3 because it matches the tire-life research question.",
    "Alternate baseline model only."
  ),
  train_rmse = c(
    rmse(slr_data$lap_time_s, predict(slr_tyre_life, newdata = slr_data)),
    rmse(slr_data$lap_time_s, predict(slr_race_progress, newdata = slr_data))
  ),
  train_mae = c(
    mae(slr_data$lap_time_s, predict(slr_tyre_life, newdata = slr_data)),
    mae(slr_data$lap_time_s, predict(slr_race_progress, newdata = slr_data))
  ),
  r_squared = c(
    summary(slr_tyre_life)$r.squared,
    summary(slr_race_progress)$r.squared
  ),
  adj_r_squared = c(
    summary(slr_tyre_life)$adj.r.squared,
    summary(slr_race_progress)$adj.r.squared
  ),
  aic = c(
    AIC(slr_tyre_life),
    AIC(slr_race_progress)
  ),
  bic = c(
    BIC(slr_tyre_life),
    BIC(slr_race_progress)
  )
) %>%
  mutate(
    across(
      c(train_rmse, train_mae, r_squared, adj_r_squared, aic, bic),
      ~ round(.x, 4)
    )
  )

write_csv(
  model_comparison,
  file.path(table_dir, "ch03_slr_model_comparison.csv")
)

cat("\nSLR model comparison using training data only:\n")
print(model_comparison)

# ----------------------------
# 8. Choose Chapter 3 model
# ----------------------------

# Chapter 3 question:
# How strongly does tire life explain lap time by itself?
#
# Because of that, the chapter model is intentionally:
# lap_time_s ~ tyre_life

chosen_model <- slr_tyre_life
chosen_model_name <- "lap_time_s ~ tyre_life"
chosen_predictor <- "tyre_life"
chosen_x_label <- "Tire Life"

cat("\nChosen SLR model for Chapter 3:\n")
cat(chosen_model_name, "\n")

# ----------------------------
# 9. Save coefficient table
# ----------------------------

coef_table <- tidy(chosen_model, conf.int = TRUE) %>%
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
  coef_table,
  file.path(table_dir, "ch03_slr_coefficient_table.csv")
)

cat("\nChosen model coefficient table:\n")
print(coef_table)

# ----------------------------
# 10. Save model summary table
# ----------------------------

model_summary <- glance(chosen_model) %>%
  mutate(
    across(where(is.numeric), ~ round(.x, 4))
  )

write_csv(
  model_summary,
  file.path(table_dir, "ch03_slr_model_summary.csv")
)

cat("\nChosen model summary:\n")
print(model_summary)

# ----------------------------
# 11. Residual and fitted value data
# ----------------------------

slr_augmented <- augment(chosen_model)

write_csv(
  slr_augmented,
  file.path(table_dir, "ch03_slr_fitted_residuals.csv")
)

residual_summary <- slr_augmented %>%
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
  file.path(table_dir, "ch03_slr_residual_summary.csv")
)

# ----------------------------
# 12. Plot data
# ----------------------------

set.seed(4230)

lap_time_limits <- quantile(slr_data$lap_time_s, c(0.01, 0.99), na.rm = TRUE)
residual_limits <- quantile(slr_augmented$.resid, c(0.01, 0.99), na.rm = TRUE)

plot_data <- slr_data %>%
  filter(
    lap_time_s >= lap_time_limits[1],
    lap_time_s <= lap_time_limits[2]
  ) %>%
  sample_n(size = min(12000, nrow(.)))

resid_plot_data <- slr_augmented %>%
  filter(
    .resid >= residual_limits[1],
    .resid <= residual_limits[2]
  ) %>%
  sample_n(size = min(12000, nrow(.)))

# ----------------------------
# 13. Fitted line plot
# ----------------------------

p_fitted_line <- ggplot(
  plot_data,
  aes(x = .data[[chosen_predictor]], y = lap_time_s)
) +
  geom_point(alpha = 0.18, color = "gray35") +
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = TRUE,
    color = "#C8102E",
    linewidth = 1.2
  ) +
  labs(
    title = "Simple Linear Regression: Lap Time by Tire Life",
    subtitle = "The fitted line shows the average linear relationship between tire age and lap time.",
    x = chosen_x_label,
    y = "Lap Time (seconds)"
  ) +
  scale_y_continuous(labels = scales::comma)

ggsave(
  filename = file.path(figure_dir, "ch03_fig01_slr_fitted_line.png"),
  plot = p_fitted_line,
  width = 8,
  height = 5,
  dpi = 300
)

# ----------------------------
# 14. Residual plot
# ----------------------------

p_residuals <- ggplot(resid_plot_data, aes(x = .fitted, y = .resid)) +
  geom_point(alpha = 0.18, color = "gray35") +
  geom_hline(yintercept = 0, color = "#C8102E", linewidth = 1) +
  geom_smooth(
    method = "loess",
    formula = y ~ x,
    se = FALSE,
    color = "black",
    linewidth = 0.8
  ) +
  labs(
    title = "Residual Plot for Simple Linear Regression",
    subtitle = "Residuals are zoomed to the 1st-99th percentile for readability.",
    x = "Fitted Lap Time",
    y = "Residual"
  )

ggsave(
  filename = file.path(figure_dir, "ch03_fig02_slr_residual_plot.png"),
  plot = p_residuals,
  width = 8,
  height = 5,
  dpi = 300
)

# ----------------------------
# 15. Optional comparison plot
# ----------------------------

comparison_long <- model_comparison %>%
  select(model, train_rmse, r_squared) %>%
  pivot_longer(
    cols = c(train_rmse, r_squared),
    names_to = "metric",
    values_to = "value"
  )

p_model_comparison <- ggplot(
  comparison_long,
  aes(x = model, y = value, fill = metric)
) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(
    title = "Simple Linear Regression Baseline Comparison",
    subtitle = "Both simple models explain very little lap-time variation by themselves.",
    x = "Model",
    y = "Value",
    fill = "Metric"
  ) +
  scale_fill_manual(
    values = c(
      "train_rmse" = "gray55",
      "r_squared" = "#C8102E"
    ),
    labels = c(
      "train_rmse" = "Training RMSE",
      "r_squared" = "R-squared"
    )
  )

ggsave(
  filename = file.path(figure_dir, "ch03_fig03_slr_model_comparison.png"),
  plot = p_model_comparison,
  width = 8,
  height = 5,
  dpi = 300
)

# ----------------------------
# 16. Save chosen model object
# ----------------------------

saveRDS(
  chosen_model,
  file.path(model_dir, "ch03_slr_tyre_life_model.rds")
)

# ----------------------------
# 17. Interpretation notes
# ----------------------------

slope <- coef_table %>%
  filter(term == chosen_predictor) %>%
  pull(estimate)

intercept <- coef_table %>%
  filter(term == "(Intercept)") %>%
  pull(estimate)

chosen_r2 <- model_summary$r.squared
chosen_adj_r2 <- model_summary$adj.r.squared
chosen_rmse <- model_comparison %>%
  filter(model == chosen_model_name) %>%
  pull(train_rmse)

interpretation <- tibble(
  item = c(
    "Chosen model",
    "Why this model was chosen",
    "Intercept",
    "Slope",
    "Training RMSE",
    "R-squared",
    "Adjusted R-squared",
    "Plain-language interpretation",
    "Chapter 3 takeaway",
    "Test set note"
  ),
  note = c(
    chosen_model_name,
    "This model directly answers the Chapter 3 question about tire life and lap time.",
    as.character(intercept),
    as.character(slope),
    as.character(chosen_rmse),
    as.character(chosen_r2),
    as.character(chosen_adj_r2),
    paste0(
      "In this simple model, each additional lap of tire life is associated with an estimated change of ",
      slope,
      " seconds in lap time, on average."
    ),
    paste0(
      "The R-squared value is ",
      chosen_r2,
      ", so tire life by itself explains only a small share of lap-time variation. ",
      "This makes SLR useful as a baseline, but not enough as a full race-strategy model."
    ),
    "The 2025 test set was not used in this script because final test evaluation should be saved for later model comparison."
  )
)

write_csv(
  interpretation,
  file.path(table_dir, "ch03_slr_interpretation_notes.csv")
)

cat("\nInterpretation notes:\n")
print(interpretation)

# ----------------------------
# 18. Final confirmation
# ----------------------------

cat("\n04_slr_lap_time.R ran successfully.\n")
cat("Chapter 3 tables saved to: ", table_dir, "\n", sep = "")
cat("Chapter 3 figures saved to: ", figure_dir, "\n", sep = "")
cat("Chapter 3 model saved to: ", model_dir, "\n", sep = "")
cat("The 2025 test set was not used in this script.\n")