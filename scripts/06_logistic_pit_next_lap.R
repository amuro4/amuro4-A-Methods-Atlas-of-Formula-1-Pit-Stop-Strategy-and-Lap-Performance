# ============================================================
# MATH 4230 Capstone Project
# Script: scripts/06_logistic_pit_next_lap.R
# Purpose: Chapter 5 - Logistic Regression for pit_next_lap
# ============================================================

# ----------------------------
# 1. Run setup
# ----------------------------

source("scripts/00_setup.R")

# ----------------------------
# 2. Organized output folders
# ----------------------------

chapter_name <- "ch05_logistic"

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

make_binary_target <- function(x) {
  x_chr <- tolower(as.character(x))
  
  case_when(
    x_chr %in% c("1", "yes", "y", "true", "pit", "pitted") ~ 1,
    x_chr %in% c("0", "no", "n", "false", "not pit", "not_pit") ~ 0,
    TRUE ~ NA_real_
  )
}

evaluate_threshold <- function(data, threshold_value) {
  temp <- data %>%
    mutate(
      predicted_class = factor(
        if_else(predicted_probability >= threshold_value, "Yes", "No"),
        levels = c("No", "Yes")
      )
    )
  
  cm <- table(
    Actual = temp$pit_next_lap_factor,
    Predicted = temp$predicted_class
  )
  
  tn <- cm["No", "No"]
  fp <- cm["No", "Yes"]
  fn <- cm["Yes", "No"]
  tp <- cm["Yes", "Yes"]
  
  accuracy <- (tp + tn) / sum(cm)
  sensitivity <- tp / (tp + fn)
  specificity <- tn / (tn + fp)
  precision <- if_else((tp + fp) == 0, NA_real_, tp / (tp + fp))
  
  tibble(
    threshold = threshold_value,
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    precision = precision,
    true_negative = as.numeric(tn),
    false_positive = as.numeric(fp),
    false_negative = as.numeric(fn),
    true_positive = as.numeric(tp)
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

logit_data <- f1_train %>%
  mutate(
    pit_next_lap_num = make_binary_target(pit_next_lap),
    pit_next_lap_factor = factor(
      if_else(pit_next_lap_num == 1, "Yes", "No"),
      levels = c("No", "Yes")
    ),
    compound = as.factor(compound),
    year_fct = as.factor(year)
  ) %>%
  select(
    pit_next_lap_num,
    pit_next_lap_factor,
    tyre_life,
    race_progress,
    stint,
    position,
    position_change,
    compound,
    year_fct
  ) %>%
  drop_na()

if (length(unique(logit_data$pit_next_lap_num)) < 2) {
  stop("pit_next_lap does not contain both classes after cleaning.")
}

cat("Training rows used for logistic regression:\n")
print(nrow(logit_data))

cat("\nClass balance for pit_next_lap:\n")
print(table(logit_data$pit_next_lap_factor))

# ----------------------------
# 6. Fit logistic regression
# ----------------------------

# Leakage note:
# pit_stop is excluded because it could leak current-lap pit information.
# lap_time_s, lap_time_delta, and cumulative_degradation are also excluded
# in this first logistic model to keep the setup focused on information
# available before deciding whether to pit next lap.

logit_formula <- pit_next_lap_num ~
  tyre_life +
  race_progress +
  stint +
  position +
  position_change +
  compound +
  year_fct

logit_model <- glm(
  formula = logit_formula,
  data = logit_data,
  family = binomial
)

cat("\nLogistic regression model fitted.\n")

# ----------------------------
# 7. Coefficient table
# ----------------------------

coef_table <- broom::tidy(logit_model) %>%
  mutate(
    estimate = round(estimate, 4),
    std.error = round(std.error, 4),
    statistic = round(statistic, 4),
    p_value = p.value,
    p_value_label = case_when(
      p.value < 0.001 ~ "<0.001",
      TRUE ~ as.character(round(p.value, 4))
    )
  ) %>%
  select(
    term,
    estimate,
    std.error,
    statistic,
    p_value,
    p_value_label
  )

write_csv(
  coef_table,
  file.path(table_dir, "ch05_logistic_coefficient_table.csv")
)

cat("\nCoefficient table:\n")
print(coef_table)

# ----------------------------
# 8. Odds ratio table
# ----------------------------

odds_ratio_table <- broom::tidy(logit_model) %>%
  mutate(
    odds_ratio = exp(estimate),
    conf.low = exp(estimate - 1.96 * std.error),
    conf.high = exp(estimate + 1.96 * std.error),
    estimate = round(estimate, 4),
    std.error = round(std.error, 4),
    odds_ratio = round(odds_ratio, 4),
    conf.low = round(conf.low, 4),
    conf.high = round(conf.high, 4),
    p_value = p.value,
    p_value_label = case_when(
      p.value < 0.001 ~ "<0.001",
      TRUE ~ as.character(round(p.value, 4))
    )
  ) %>%
  select(
    term,
    estimate,
    std.error,
    odds_ratio,
    conf.low,
    conf.high,
    p_value,
    p_value_label
  )

write_csv(
  odds_ratio_table,
  file.path(table_dir, "ch05_logistic_odds_ratio_table.csv")
)

cat("\nOdds ratio table:\n")
print(odds_ratio_table)

# ----------------------------
# 9. Predicted probabilities and classifications
# ----------------------------

pred_results <- logit_data %>%
  mutate(
    predicted_probability = predict(logit_model, newdata = logit_data, type = "response"),
    predicted_class_050 = factor(
      if_else(predicted_probability >= 0.50, "Yes", "No"),
      levels = c("No", "Yes")
    )
  )

write_csv(
  pred_results,
  file.path(table_dir, "ch05_logistic_classification_results.csv")
)

# ----------------------------
# 10. Confusion matrix at 0.50 threshold
# ----------------------------

confusion_matrix <- table(
  Actual = pred_results$pit_next_lap_factor,
  Predicted = pred_results$predicted_class_050
)

confusion_matrix_table <- as.data.frame(confusion_matrix) %>%
  as_tibble()

write_csv(
  confusion_matrix_table,
  file.path(table_dir, "ch05_logistic_confusion_matrix_050.csv")
)

cat("\nConfusion matrix using 0.50 threshold:\n")
print(confusion_matrix)

# ----------------------------
# 11. Threshold comparison
# ----------------------------

threshold_values <- c(0.20, 0.30, 0.40, 0.50, 0.60)

threshold_metrics <- map_dfr(
  threshold_values,
  ~ evaluate_threshold(pred_results, .x)
) %>%
  mutate(
    across(
      c(accuracy, sensitivity, specificity, precision),
      ~ round(.x, 4)
    )
  )

write_csv(
  threshold_metrics,
  file.path(table_dir, "ch05_logistic_threshold_comparison.csv")
)

classification_metrics_050 <- threshold_metrics %>%
  filter(threshold == 0.50)

write_csv(
  classification_metrics_050,
  file.path(table_dir, "ch05_logistic_classification_metrics_050.csv")
)

cat("\nThreshold comparison:\n")
print(threshold_metrics)

# ----------------------------
# 12. ROC and AUC
# ----------------------------

roc_object <- pROC::roc(
  response = pred_results$pit_next_lap_num,
  predictor = pred_results$predicted_probability,
  quiet = TRUE
)

auc_value <- as.numeric(pROC::auc(roc_object))

roc_table <- tibble(
  specificity = roc_object$specificities,
  sensitivity = roc_object$sensitivities,
  false_positive_rate = 1 - specificity
)

auc_table <- tibble(
  auc = round(auc_value, 4)
)

write_csv(
  roc_table,
  file.path(table_dir, "ch05_logistic_roc_curve_data.csv")
)

write_csv(
  auc_table,
  file.path(table_dir, "ch05_logistic_auc.csv")
)

cat("\nAUC:\n")
print(auc_table)

# ----------------------------
# 13. Plot: Pit rate by tire life
# ----------------------------

prob_by_tyre_life <- pred_results %>%
  mutate(
    tyre_life_bin = cut(
      tyre_life,
      breaks = c(0, 5, 10, 15, 20, 30, 40, Inf),
      include.lowest = TRUE,
      right = FALSE
    )
  ) %>%
  group_by(tyre_life_bin) %>%
  summarise(
    n = n(),
    actual_pit_rate = mean(pit_next_lap_num),
    mean_predicted_probability = mean(predicted_probability),
    .groups = "drop"
  )

write_csv(
  prob_by_tyre_life,
  file.path(table_dir, "ch05_pit_rate_by_tyre_life.csv")
)

p_prob_tyre_life <- ggplot(prob_by_tyre_life, aes(x = tyre_life_bin)) +
  geom_col(aes(y = actual_pit_rate), fill = f1_red) +
  geom_point(aes(y = mean_predicted_probability), color = "black", size = 3) +
  geom_line(aes(y = mean_predicted_probability, group = 1), color = "black", linewidth = 1) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Pit-Next-Lap Rate by Tire Life",
    subtitle = "Bars show actual pit rate; black points show mean predicted probability.",
    x = "Tire Life Bin",
    y = "Pit-Next-Lap Rate / Probability"
  )

safe_ggsave(
  plot_object = p_prob_tyre_life,
  filename = "ch05_fig01_probability_by_tyre_life.png",
  width = 8,
  height = 5
)

# ----------------------------
# 14. Plot: ROC curve
# ----------------------------

p_roc <- ggplot(roc_table, aes(x = false_positive_rate, y = sensitivity)) +
  geom_line(color = f1_red, linewidth = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
  coord_equal() +
  labs(
    title = "ROC Curve for Logistic Regression",
    subtitle = paste0("Training AUC = ", round(auc_value, 4)),
    x = "False Positive Rate",
    y = "True Positive Rate"
  )

safe_ggsave(
  plot_object = p_roc,
  filename = "ch05_fig02_roc_curve.png",
  width = 6,
  height = 6
)

# ----------------------------
# 15. Plot: Numeric odds ratios
# ----------------------------

numeric_odds_plot <- odds_ratio_table %>%
  filter(
    term %in% c(
      "tyre_life",
      "race_progress",
      "stint",
      "position",
      "position_change"
    )
  ) %>%
  mutate(term = fct_reorder(term, odds_ratio))

p_numeric_odds <- ggplot(numeric_odds_plot, aes(x = term, y = odds_ratio)) +
  geom_col(fill = f1_red) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray40") +
  coord_flip() +
  labs(
    title = "Odds Ratios for Numeric Predictors",
    subtitle = "Odds ratios are useful, but variables use different units, so compare them carefully.",
    x = "Predictor",
    y = "Odds Ratio"
  )

safe_ggsave(
  plot_object = p_numeric_odds,
  filename = "ch05_fig03_numeric_odds_ratios.png",
  width = 8,
  height = 5
)

# ----------------------------
# 16. Plot: Confusion matrix
# ----------------------------

p_confusion <- confusion_matrix_table %>%
  ggplot(aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = scales::comma(Freq)), size = 6, color = "black") +
  scale_fill_gradient(low = "white", high = f1_red) +
  labs(
    title = "Confusion Matrix at 0.50 Threshold",
    subtitle = "The model is conservative: it predicts No more easily than Yes.",
    x = "Predicted Class",
    y = "Actual Class",
    fill = "Count"
  )

safe_ggsave(
  plot_object = p_confusion,
  filename = "ch05_fig04_confusion_matrix_050.png",
  width = 7,
  height = 5
)

# ----------------------------
# 17. Plot: Threshold comparison
# ----------------------------

threshold_plot_data <- threshold_metrics %>%
  select(threshold, accuracy, sensitivity, specificity, precision) %>%
  pivot_longer(
    cols = c(accuracy, sensitivity, specificity, precision),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = case_when(
      metric == "accuracy" ~ "Accuracy",
      metric == "sensitivity" ~ "Sensitivity",
      metric == "specificity" ~ "Specificity",
      metric == "precision" ~ "Precision",
      TRUE ~ metric
    )
  )

p_threshold <- ggplot(threshold_plot_data, aes(x = threshold, y = value, color = metric)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Classification Metrics by Threshold",
    subtitle = "Lower thresholds catch more pit-next-lap cases but create more false positives.",
    x = "Probability Threshold",
    y = "Metric Value",
    color = "Metric"
  )

safe_ggsave(
  plot_object = p_threshold,
  filename = "ch05_fig05_threshold_comparison.png",
  width = 8,
  height = 5
)

# ----------------------------
# 18. Save lightweight model record
# ----------------------------

model_record <- list(
  model_name = "Logistic Regression for Pit Next Lap",
  formula = deparse(logit_formula),
  coefficients = coef(logit_model),
  odds_ratios = odds_ratio_table,
  threshold_metrics = threshold_metrics,
  auc = auc_table,
  note = "This model excludes pit_stop and uses training data only. Re-run this script to recreate the full glm object."
)

saveRDS(
  model_record,
  file.path(model_dir, "ch05_logistic_model_record.rds")
)

# ----------------------------
# 19. Interpretation notes
# ----------------------------

tyre_life_or <- odds_ratio_table %>%
  filter(term == "tyre_life") %>%
  pull(odds_ratio)

stint_or <- odds_ratio_table %>%
  filter(term == "stint") %>%
  pull(odds_ratio)

race_progress_or <- odds_ratio_table %>%
  filter(term == "race_progress") %>%
  pull(odds_ratio)

interpretation <- tibble(
  item = c(
    "Main model",
    "Target variable",
    "Leakage decision",
    "Tire life odds ratio",
    "Stint odds ratio",
    "Race progress odds ratio",
    "Classification threshold",
    "Accuracy at 0.50",
    "Sensitivity at 0.50",
    "Specificity at 0.50",
    "AUC",
    "Threshold takeaway",
    "Class imbalance note",
    "Real-world decision",
    "Test set note"
  ),
  note = c(
    "Logistic regression was used to model the probability that a driver pits on the next lap.",
    "The response variable is pit_next_lap, coded as No = 0 and Yes = 1.",
    "pit_stop was excluded because current-lap pit information could create leakage.",
    paste0("For each one-lap increase in tire life, the estimated odds of pitting next lap were multiplied by about ", tyre_life_or, ", holding other predictors fixed."),
    paste0("For each one-unit increase in stint, the estimated odds of pitting next lap were multiplied by about ", stint_or, ", holding other predictors fixed."),
    paste0("Race progress had odds ratio ", race_progress_or, ", but this effect is across the 0 to 1 race-progress scale, so it should not be directly compared to one-lap variables."),
    "The main confusion matrix used a 0.50 predicted probability threshold.",
    paste0("Training accuracy at 0.50 was ", round(100 * classification_metrics_050$accuracy, 1), "%."),
    paste0("Training sensitivity at 0.50 was ", round(100 * classification_metrics_050$sensitivity, 1), "%."),
    paste0("Training specificity at 0.50 was ", round(100 * classification_metrics_050$specificity, 1), "%."),
    paste0("Training AUC was ", round(auc_value, 4), "."),
    "The 0.50 threshold is conservative. It has high specificity but misses many actual pit-next-lap cases, so a lower threshold may be better for an early-warning strategy tool.",
    "Because pit_next_lap is imbalanced, accuracy alone is not enough. Sensitivity, specificity, precision, and AUC should also be reported.",
    "A strategist could use logistic regression as an interpretable baseline for pit-stop probability, but the probability threshold should be tuned depending on whether false alarms or missed pit stops are more costly.",
    "The 2025 test set was not used in this script. Final test evaluation should be saved for later model comparison."
  )
)

write_csv(
  interpretation,
  file.path(table_dir, "ch05_logistic_interpretation_notes.csv")
)

report_notes <- c(
  "Chapter 5 Report Notes",
  "",
  paste0(
    "The logistic regression model predicted the probability that pit_next_lap equals Yes. ",
    "The model used tire life, race progress, stint, position, position change, compound, and year as predictors."
  ),
  "",
  "The variable pit_stop was excluded because it could create leakage. Since the goal is to predict whether a driver will pit next lap, current-lap pit information should not be used as a normal predictor.",
  "",
  paste0(
    "The model had training AUC = ",
    round(auc_value, 4),
    ", which means it separated pit-next-lap cases from non-pit-next-lap cases fairly well for a simple interpretable baseline."
  ),
  "",
  paste0(
    "Using a 0.50 probability threshold, the model had training accuracy = ",
    round(100 * classification_metrics_050$accuracy, 1),
    "%, sensitivity = ",
    round(100 * classification_metrics_050$sensitivity, 1),
    "%, specificity = ",
    round(100 * classification_metrics_050$specificity, 1),
    "%, and precision = ",
    round(100 * classification_metrics_050$precision, 1),
    "%."
  ),
  "",
  "The confusion matrix shows that the 0.50 threshold is conservative. It correctly identifies many No cases, but it misses many Yes cases. For a race strategy warning tool, a lower threshold may be more useful because missing a likely pit stop could be worse than giving an early warning.",
  "",
  paste0(
    "The tire life odds ratio was about ",
    tyre_life_or,
    ", meaning older tires were associated with higher odds of pitting next lap. ",
    "The stint odds ratio was about ",
    stint_or,
    ", meaning later stints were also associated with higher pit-next-lap odds."
  ),
  "",
  paste0(
    "Race progress had an odds ratio of ",
    race_progress_or,
    ", but this should be interpreted carefully because race_progress is measured on a 0 to 1 scale. ",
    "Odds ratios should not be compared too casually when predictors use different units."
  ),
  "",
  "Real-world decision: Logistic regression is useful as a simple and interpretable pit-stop prediction baseline. It gives understandable odds ratios and a strong AUC, but the threshold needs tuning before it would be useful as a strategist's warning system."
)

writeLines(
  report_notes,
  file.path(table_dir, "ch05_report_notes.txt")
)

cat("\nInterpretation notes:\n")
print(interpretation)

# ----------------------------
# 20. Final confirmation
# ----------------------------

cat("\n06_logistic_pit_next_lap.R ran successfully.\n")
cat("Chapter 5 tables saved to: ", table_dir, "\n", sep = "")
cat("Chapter 5 figures saved to: ", figure_dir, "\n", sep = "")
cat("Chapter 5 model record saved to: ", model_dir, "\n", sep = "")
cat("Report notes saved to: ", file.path(table_dir, "ch05_report_notes.txt"), "\n", sep = "")
cat("The 2025 test set was not used in this script.\n")