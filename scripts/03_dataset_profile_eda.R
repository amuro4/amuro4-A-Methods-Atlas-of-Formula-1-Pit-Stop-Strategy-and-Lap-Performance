# ============================================================
# MATH 4230 Capstone Project
# Script: scripts/03_dataset_profile_eda.R
# Purpose: Dataset profile and EDA for Chapter 2
# ============================================================

# ----------------------------
# 1. Run setup
# ----------------------------

source("scripts/00_setup.R")

# ----------------------------
# 2. Create output folders
# ----------------------------

dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("figures", recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# 3. Load training data only
# ----------------------------

train_path <- "data/processed/f1_train.csv"

if (!file.exists(train_path)) {
  stop("Training data not found. Run scripts/02_split_data.R first.")
}

f1 <- read_csv(train_path, show_col_types = FALSE)

f1 <- f1 %>%
  mutate(
    race = as.factor(race),
    driver = as.factor(driver),
    compound = as.factor(compound),
    race_id = as.factor(race_id),
    pit_stop = factor(pit_stop, levels = c("No", "Yes")),
    pit_next_lap = factor(pit_next_lap, levels = c("No", "Yes"))
  )

cat("EDA data loaded successfully.\n")
cat("Rows and columns:\n")
print(dim(f1))

# ----------------------------
# 4. Plot theme and save helper
# ----------------------------

theme_f1 <- function() {
  theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", size = 18),
      plot.subtitle = element_text(size = 11, color = "gray35"),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(color = "gray25"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "gray90"),
      legend.position = "bottom",
      legend.title = element_text(face = "bold")
    )
}

theme_set(theme_f1())

save_plot <- function(filename, plot, width = 8, height = 5) {
  ggsave(
    filename = paste0("figures/", filename),
    plot = plot,
    width = width,
    height = height,
    dpi = 300
  )
}

f1_red <- "#C8102E"
f1_dark <- "gray35"
f1_gray <- "gray55"

# ----------------------------
# 5. Data dictionary table
# ----------------------------

data_dictionary <- tibble(
  variable = names(f1),
  type = sapply(f1, function(x) class(x)[1])
) %>%
  mutate(
    units = case_when(
      variable == "lap_time_s" ~ "seconds",
      variable == "race_progress" ~ "proportion from 0 to 1",
      variable %in% c("lap_number", "position", "stint", "tyre_life") ~ "count",
      variable == "year" ~ "year",
      TRUE ~ "none"
    ),
    description = case_when(
      variable == "driver" ~ "Driver abbreviation or name.",
      variable == "lap_number" ~ "Lap number within the race.",
      variable == "compound" ~ "Tire compound used on the lap.",
      variable == "stint" ~ "Current tire stint number.",
      variable == "tyre_life" ~ "Number of laps on the current tire set.",
      variable == "position" ~ "Driver position on the lap.",
      variable == "lap_time_s" ~ "Lap time measured in seconds.",
      variable == "race" ~ "Race name.",
      variable == "year" ~ "Formula 1 season year.",
      variable == "lap_time_delta" ~ "Lap time difference metric from the engineered dataset.",
      variable == "cumulative_degradation" ~ "Cumulative tire degradation metric from the engineered dataset.",
      variable == "pit_stop" ~ "Whether a pit stop occurred on the current lap.",
      variable == "pit_next_lap" ~ "Whether the driver pits on the next lap.",
      variable == "race_progress" ~ "Proportion of the race completed.",
      variable == "normalized_tyre_life" ~ "Tire life normalized within the engineered dataset.",
      variable == "position_change" ~ "Change in driver position.",
      variable == "race_id" ~ "Race-year identifier created for this project.",
      variable == "compound_encoded" ~ "Simple numeric encoding of tire compound.",
      TRUE ~ "Variable from the cleaned Kaggle dataset."
    )
  ) %>%
  arrange(variable)

write_csv(data_dictionary, "results/tables/data_dictionary.csv")

# ----------------------------
# 6. Missingness table
# ----------------------------

missingness_table <- f1 %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "missing_count"
  ) %>%
  mutate(
    total_rows = nrow(f1),
    missing_percent = round(100 * missing_count / total_rows, 2)
  ) %>%
  arrange(desc(missing_count), variable)

write_csv(missingness_table, "results/tables/missingness_table.csv")

# ----------------------------
# 7. Summary statistics table
# ----------------------------

numeric_data <- f1 %>%
  select(where(is.numeric))

summary_stats <- numeric_data %>%
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "value"
  ) %>%
  group_by(variable) %>%
  summarise(
    n = sum(!is.na(value)),
    missing_count = sum(is.na(value)),
    missing_percent = round(100 * missing_count / nrow(f1), 2),
    mean = round(mean(value, na.rm = TRUE), 0),
    sd = round(sd(value, na.rm = TRUE), 3),
    min = round(min(value, na.rm = TRUE), 3),
    q1 = round(quantile(value, 0.25, na.rm = TRUE), 3),
    median = round(median(value, na.rm = TRUE), 3),
    q3 = round(quantile(value, 0.75, na.rm = TRUE), 3),
    max = round(max(value, na.rm = TRUE), 3),
    .groups = "drop"
  ) %>%
  arrange(variable)

write_csv(summary_stats, "results/tables/summary_statistics.csv")

# ----------------------------
# 8. Class balance table
# ----------------------------

class_balance <- f1 %>%
  count(pit_next_lap) %>%
  mutate(
    percent = round(100 * n / sum(n), 2)
  )

write_csv(class_balance, "results/tables/pit_next_lap_class_balance.csv")

# ----------------------------
# 9. Plot limits for outlier-aware figures
# ----------------------------

outlier_summary <- f1 %>%
  summarise(
    lap_time_p01 = quantile(lap_time_s, 0.01, na.rm = TRUE),
    lap_time_p99 = quantile(lap_time_s, 0.99, na.rm = TRUE),
    degradation_p01 = quantile(cumulative_degradation, 0.01, na.rm = TRUE),
    degradation_p99 = quantile(cumulative_degradation, 0.99, na.rm = TRUE),
    tyre_life_p99 = quantile(tyre_life, 0.99, na.rm = TRUE)
  )

write_csv(outlier_summary, "results/tables/outlier_plot_limits.csv")

lap_time_limits <- quantile(f1$lap_time_s, c(0.01, 0.99), na.rm = TRUE)
degradation_limits <- quantile(f1$cumulative_degradation, c(0.01, 0.99), na.rm = TRUE)

# ----------------------------
# 10. Histograms
# ----------------------------

p1 <- f1 %>%
  filter(
    lap_time_s >= lap_time_limits[1],
    lap_time_s <= lap_time_limits[2]
  ) %>%
  ggplot(aes(x = lap_time_s)) +
  geom_histogram(bins = 40, fill = f1_red, color = "white", linewidth = 0.25) +
  labs(
    title = "Distribution of Lap Time",
    subtitle = "Zoomed to the 1st-99th percentile so extreme laps do not hide the main pattern.",
    x = "Lap Time (seconds)",
    y = "Count"
  ) +
  scale_y_continuous(labels = scales::comma)

save_plot("fig01_lap_time_distribution.png", p1)

p4 <- ggplot(f1, aes(x = tyre_life)) +
  geom_histogram(bins = 40, fill = f1_red, color = "white", linewidth = 0.25) +
  labs(
    title = "Distribution of Tire Life",
    subtitle = "Most tire stints are short to moderate, with a long right tail.",
    x = "Tire Life",
    y = "Count"
  ) +
  scale_y_continuous(labels = scales::comma)

save_plot("fig04_tyre_life_distribution.png", p4)

p9 <- ggplot(f1, aes(x = race_progress)) +
  geom_histogram(bins = 40, fill = f1_red, color = "white", linewidth = 0.25) +
  labs(
    title = "Distribution of Race Progress",
    subtitle = "Race progress is measured from the beginning of the race toward the final lap.",
    x = "Race Progress",
    y = "Count"
  ) +
  scale_y_continuous(labels = scales::comma)

save_plot("fig09_race_progress_distribution.png", p9)

p10 <- f1 %>%
  filter(
    cumulative_degradation >= degradation_limits[1],
    cumulative_degradation <= degradation_limits[2]
  ) %>%
  ggplot(aes(x = cumulative_degradation)) +
  geom_histogram(bins = 40, fill = f1_red, color = "white", linewidth = 0.25) +
  labs(
    title = "Distribution of Cumulative Degradation",
    subtitle = "Zoomed to the 1st-99th percentile because a few extreme laps stretch the full scale.",
    x = "Cumulative Degradation",
    y = "Count"
  ) +
  scale_y_continuous(labels = scales::comma)

save_plot("fig10_cumulative_degradation_distribution.png", p10)

# ----------------------------
# 11. Bar charts
# ----------------------------

p2_data <- f1 %>%
  count(pit_next_lap) %>%
  mutate(
    percent = n / sum(n),
    label = paste0(scales::comma(n), " (", scales::percent(percent, accuracy = 0.1), ")")
  )

p2 <- ggplot(p2_data, aes(x = pit_next_lap, y = n, fill = pit_next_lap)) +
  geom_col(width = 0.65, show.legend = FALSE) +
  geom_text(aes(label = label), vjust = -0.4, size = 4) +
  labs(
    title = "Class Balance for Pit Next Lap",
    subtitle = "The classification target is imbalanced, so accuracy alone may be misleading.",
    x = "Pit Next Lap",
    y = "Count"
  ) +
  scale_fill_manual(values = c("No" = f1_gray, "Yes" = f1_red)) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.12)))

save_plot("fig02_pit_next_lap_balance.png", p2)

compound_counts <- f1 %>%
  count(compound) %>%
  mutate(
    compound = fct_reorder(compound, n),
    label = scales::comma(n)
  )

p7 <- ggplot(compound_counts, aes(x = compound, y = n)) +
  geom_col(fill = f1_red, width = 0.7) +
  geom_text(aes(label = label), hjust = -0.1, size = 3.8) +
  coord_flip() +
  labs(
    title = "Tire Compound Counts",
    subtitle = "Hard and medium compounds appear most often in the training data.",
    x = "Compound",
    y = "Count"
  ) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.15)))

save_plot("fig07_compound_counts.png", p7)

driver_counts <- f1 %>%
  count(driver) %>%
  mutate(driver = fct_reorder(driver, n))

p8 <- ggplot(driver_counts, aes(x = driver, y = n)) +
  geom_col(fill = f1_dark, width = 0.7) +
  coord_flip() +
  labs(
    title = "Driver Counts",
    subtitle = "Different drivers have different numbers of laps because seasons and race entries vary.",
    x = "Driver",
    y = "Count"
  ) +
  scale_y_continuous(labels = scales::comma)

save_plot("fig08_driver_counts.png", p8, width = 8, height = 8)

race_counts <- f1 %>%
  count(race) %>%
  mutate(race = fct_reorder(race, n))

p11 <- ggplot(race_counts, aes(x = race, y = n)) +
  geom_col(fill = f1_dark, width = 0.7) +
  coord_flip() +
  labs(
    title = "Race Counts",
    subtitle = "Lap counts vary because race calendars and sessions differ across seasons.",
    x = "Race",
    y = "Count"
  ) +
  scale_y_continuous(labels = scales::comma)

save_plot("fig11_race_counts.png", p11, width = 10, height = 9)

# ----------------------------
# 12. Correlation heatmap
# ----------------------------

corr_vars <- c(
  "lap_number",
  "race_progress",
  "stint",
  "tyre_life",
  "normalized_tyre_life",
  "position",
  "position_change",
  "lap_time_s",
  "lap_time_delta",
  "cumulative_degradation"
)

numeric_for_corr <- f1 %>%
  select(all_of(corr_vars))

cor_matrix <- cor(numeric_for_corr, use = "pairwise.complete.obs")

cor_long <- as.data.frame(as.table(cor_matrix)) %>%
  rename(
    var1 = Var1,
    var2 = Var2,
    correlation = Freq
  ) %>%
  mutate(
    var1 = factor(var1, levels = corr_vars),
    var2 = factor(var2, levels = rev(corr_vars))
  )

p5 <- ggplot(cor_long, aes(x = var1, y = var2, fill = correlation)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(aes(label = round(correlation, 2)), size = 3) +
  scale_fill_gradient2(
    low = "#2166AC",
    mid = "white",
    high = "#B2182B",
    midpoint = 0,
    limits = c(-1, 1)
  ) +
  labs(
    title = "Correlation Heatmap for Numeric Predictors",
    subtitle = "Timing variables are strongly related, which matters for later multicollinearity checks.",
    x = NULL,
    y = NULL,
    fill = "Correlation"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank()
  )

save_plot("fig05_numeric_correlation_heatmap.png", p5, width = 9, height = 7)

# ----------------------------
# 13. Boxplots
# ----------------------------

p3 <- ggplot(f1, aes(x = fct_reorder(compound, lap_time_s, median), y = lap_time_s)) +
  geom_boxplot(fill = f1_red, color = "gray20", outlier.alpha = 0.20) +
  coord_cartesian(ylim = lap_time_limits) +
  labs(
    title = "Lap Time by Tire Compound",
    subtitle = "Y-axis zoomed to the 1st-99th percentile so normal lap-time differences are visible.",
    x = "Compound",
    y = "Lap Time (seconds)"
  )

save_plot("fig03_lap_time_by_compound.png", p3)

p6 <- ggplot(f1, aes(x = pit_next_lap, y = lap_time_s, fill = pit_next_lap)) +
  geom_boxplot(color = "gray20", outlier.alpha = 0.20, show.legend = FALSE) +
  coord_cartesian(ylim = lap_time_limits) +
  labs(
    title = "Lap Time by Pit Next Lap",
    subtitle = "Y-axis zoomed to the 1st-99th percentile because a few very slow laps dominate the full scale.",
    x = "Pit Next Lap",
    y = "Lap Time (seconds)"
  ) +
  scale_fill_manual(values = c("No" = f1_gray, "Yes" = f1_red))

save_plot("fig06_lap_time_by_pit_next_lap.png", p6)

# ----------------------------
# 14. Extra EDA plots for pit strategy
# ----------------------------

tyre_life_pit_rate <- f1 %>%
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
    pit_next_lap_rate = mean(pit_next_lap == "Yes", na.rm = TRUE),
    .groups = "drop"
  )

write_csv(tyre_life_pit_rate, "results/tables/pit_rate_by_tyre_life.csv")

p12 <- ggplot(tyre_life_pit_rate, aes(x = tyre_life_bin, y = pit_next_lap_rate)) +
  geom_col(fill = f1_red, width = 0.7) +
  geom_text(
    aes(label = scales::percent(pit_next_lap_rate, accuracy = 0.1)),
    vjust = -0.4,
    size = 3.6
  ) +
  labs(
    title = "Pit-Next-Lap Rate by Tire Life",
    subtitle = "Older tire stints generally create more reason to consider a pit stop soon.",
    x = "Tire Life Bin",
    y = "Pit-Next-Lap Rate"
  ) +
  scale_y_continuous(
    labels = scales::percent,
    expand = expansion(mult = c(0, 0.12))
  )

save_plot("fig12_pit_rate_by_tyre_life.png", p12)

race_progress_pit_rate <- f1 %>%
  mutate(
    race_progress_capped = pmin(pmax(race_progress, 0), 1),
    race_progress_bin = cut(
      race_progress_capped,
      breaks = seq(0, 1, by = 0.10),
      include.lowest = TRUE
    )
  ) %>%
  group_by(race_progress_bin) %>%
  summarise(
    n = n(),
    pit_next_lap_rate = mean(pit_next_lap == "Yes", na.rm = TRUE),
    .groups = "drop"
  )

write_csv(race_progress_pit_rate, "results/tables/pit_rate_by_race_progress.csv")

p13 <- ggplot(race_progress_pit_rate, aes(x = race_progress_bin, y = pit_next_lap_rate)) +
  geom_col(fill = f1_red, width = 0.7) +
  geom_text(
    aes(label = scales::percent(pit_next_lap_rate, accuracy = 0.1)),
    vjust = -0.4,
    size = 3.4
  ) +
  labs(
    title = "Pit-Next-Lap Rate by Race Progress",
    subtitle = "Pit decisions are not evenly distributed across the race timeline.",
    x = "Race Progress Bin",
    y = "Pit-Next-Lap Rate"
  ) +
  scale_y_continuous(
    labels = scales::percent,
    expand = expansion(mult = c(0, 0.12))
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

save_plot("fig13_pit_rate_by_race_progress.png", p13, width = 9, height = 5)

# ----------------------------
# 15. Split strategy note
# ----------------------------

split_note <- tibble(
  item = c(
    "Training seasons",
    "Testing season",
    "Reason for year-based split",
    "Test set rule"
  ),
  explanation = c(
    "2022, 2023, and 2024",
    "2025",
    "A random row split could leak race context because laps from the same race are connected. A year-based split better mimics training on past seasons and evaluating on a later season.",
    "The 2025 test set should stay untouched until final model evaluation."
  )
)

write_csv(split_note, "results/tables/train_test_split_note.csv")

# ----------------------------
# 16. Final confirmation
# ----------------------------

cat("\n03_dataset_profile_eda.R ran successfully.\n")
cat("Tables saved to: results/tables/\n")
cat("Figures saved to: figures/\n")

cat("\nTables created:\n")
print(list.files("results/tables"))

cat("\nFigures created:\n")
print(list.files("figures"))