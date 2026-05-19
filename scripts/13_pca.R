# ============================================================
# MATH 4230 Capstone Project
# Script: scripts/13_pca.R
# Purpose: Chapter 12 - Principal Components Analysis
# ============================================================

# ----------------------------
# 1. Run setup
# ----------------------------

source("scripts/00_setup.R")

# ----------------------------
# 2. Extra packages for PCA
# ----------------------------

packages <- c(
  "tidyverse",
  "broom"
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

chapter_name <- "ch12_pca"

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

make_binary_target <- function(x) {
  x_chr <- tolower(as.character(x))
  
  case_when(
    x_chr %in% c("1", "yes", "y", "true", "pit", "pitted") ~ 1,
    x_chr %in% c("0", "no", "n", "false", "not pit", "not_pit") ~ 0,
    TRUE ~ NA_real_
  )
}

paste_top_loadings <- function(loadings_data, pc_name, n_terms = 5) {
  text <- loadings_data %>%
    filter(PC == pc_name) %>%
    arrange(desc(abs_loading)) %>%
    slice_head(n = n_terms) %>%
    mutate(label = paste0(variable, " (", round(loading, 3), ")")) %>%
    pull(label) %>%
    paste(collapse = ", ")
  
  if (text == "") {
    text <- "No loadings available"
  }
  
  text
}

# ----------------------------
# 5. Load training data
# ----------------------------

train_path <- "data/processed/f1_train.csv"

if (!file.exists(train_path)) {
  stop("Training data not found. Run scripts/02_split_data.R first.")
}

f1_train <- read_csv(train_path, show_col_types = FALSE)

cat("Training data dimensions:\n")
print(dim(f1_train))

# ----------------------------
# 6. Prepare PCA data
# ----------------------------

# PCA is unsupervised, but I still use only the training data here.
# The response variables are not included in the PCA.
# Only numeric predictors are used.

pca_data <- f1_train %>%
  mutate(
    pit_next_lap_num = make_binary_target(pit_next_lap),
    pit_next_lap_label = factor(
      if_else(pit_next_lap_num == 1, "Pit next lap", "No pit next lap"),
      levels = c("No pit next lap", "Pit next lap")
    )
  ) %>%
  select(
    pit_next_lap_label,
    lap_number,
    tyre_life,
    normalized_tyre_life,
    race_progress,
    stint,
    position,
    position_change,
    year
  ) %>%
  drop_na()

pca_predictors <- pca_data %>%
  select(
    lap_number,
    tyre_life,
    normalized_tyre_life,
    race_progress,
    stint,
    position,
    position_change,
    year
  )

data_summary <- tibble(
  item = c(
    "Training rows available",
    "Rows used for PCA",
    "Numeric predictors used",
    "Response variables included in PCA",
    "Standardization used"
  ),
  value = c(
    nrow(f1_train),
    nrow(pca_predictors),
    ncol(pca_predictors),
    0,
    1
  )
)

write_csv(
  data_summary,
  file.path(table_dir, "ch12_pca_data_summary.csv")
)

cat("\nPCA data summary:\n")
print(data_summary)

cat("\nNumeric predictors used for PCA:\n")
print(names(pca_predictors))

# ----------------------------
# 7. Run PCA
# ----------------------------

pca_fit <- prcomp(
  pca_predictors,
  center = TRUE,
  scale. = TRUE
)

cat("\nPCA finished.\n")

# ----------------------------
# 8. Variance explained
# ----------------------------

pca_sdev <- pca_fit$sdev

variance_explained <- tibble(
  PC = paste0("PC", seq_along(pca_sdev)),
  standard_deviation = pca_sdev,
  eigenvalue = pca_sdev^2,
  proportion_variance = (pca_sdev^2) / sum(pca_sdev^2),
  cumulative_variance = cumsum(proportion_variance)
) %>%
  mutate(
    across(
      c(standard_deviation, eigenvalue, proportion_variance, cumulative_variance),
      ~ round(.x, 4)
    )
  )

write_csv(
  variance_explained,
  file.path(table_dir, "ch12_pca_variance_explained.csv")
)

cat("\nPCA variance explained:\n")
print(variance_explained)

# ----------------------------
# 9. Loadings
# ----------------------------

loadings_table <- as_tibble(
  pca_fit$rotation,
  rownames = "variable"
) %>%
  pivot_longer(
    cols = starts_with("PC"),
    names_to = "PC",
    values_to = "loading"
  ) %>%
  mutate(
    abs_loading = abs(loading),
    loading = round(loading, 4),
    abs_loading = round(abs_loading, 4)
  ) %>%
  arrange(PC, desc(abs_loading))

write_csv(
  loadings_table,
  file.path(table_dir, "ch12_pca_loadings.csv")
)

top_loadings <- loadings_table %>%
  filter(PC %in% c("PC1", "PC2", "PC3")) %>%
  group_by(PC) %>%
  slice_max(abs_loading, n = 5, with_ties = FALSE) %>%
  ungroup() %>%
  arrange(PC, desc(abs_loading))

write_csv(
  top_loadings,
  file.path(table_dir, "ch12_pca_top_loadings.csv")
)

cat("\nTop PCA loadings:\n")
print(top_loadings)

# ----------------------------
# 10. PCA scores for plot
# ----------------------------

pca_scores <- as_tibble(pca_fit$x[, 1:3]) %>%
  bind_cols(
    pca_data %>%
      select(pit_next_lap_label)
  )

# Use a plotting sample to keep the figure readable and lightweight.
set.seed(4230)

plot_n <- min(10000, nrow(pca_scores))

pca_scores_plot <- pca_scores %>%
  slice_sample(n = plot_n)

write_csv(
  pca_scores_plot,
  file.path(table_dir, "ch12_pca_scores_plot_sample.csv")
)

# ----------------------------
# 11. Figures
# ----------------------------

p_scree <- variance_explained %>%
  mutate(PC_number = row_number()) %>%
  ggplot(aes(x = PC_number, y = proportion_variance)) +
  geom_col(fill = f1_red, width = 0.7) +
  geom_line(aes(y = cumulative_variance), color = "gray35", linewidth = 1) +
  geom_point(aes(y = cumulative_variance), color = "gray35", size = 2) +
  scale_x_continuous(
    breaks = seq_len(nrow(variance_explained)),
    labels = variance_explained$PC
  ) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    limits = c(0, 1)
  ) +
  labs(
    title = "PCA Scree Plot",
    subtitle = "Bars show individual variance explained; line shows cumulative variance.",
    x = "Principal Component",
    y = "Variance Explained"
  )

safe_ggsave(
  plot_object = p_scree,
  filename = "ch12_fig01_pca_scree_plot.png",
  width = 8,
  height = 5
)

pc1_var <- variance_explained %>%
  filter(PC == "PC1") %>%
  pull(proportion_variance)

pc2_var <- variance_explained %>%
  filter(PC == "PC2") %>%
  pull(proportion_variance)

p_pc12 <- ggplot(
  pca_scores_plot,
  aes(x = PC1, y = PC2, color = pit_next_lap_label)
) +
  geom_point(alpha = 0.35, size = 1) +
  labs(
    title = "PCA: PC1 vs PC2",
    subtitle = paste0(
      "PC1 explains ",
      scales::percent(pc1_var, accuracy = 0.1),
      "; PC2 explains ",
      scales::percent(pc2_var, accuracy = 0.1),
      "."
    ),
    x = "PC1",
    y = "PC2",
    color = "Class"
  )

safe_ggsave(
  plot_object = p_pc12,
  filename = "ch12_fig02_pca_pc1_pc2.png",
  width = 8,
  height = 5
)

# ----------------------------
# 12. Lightweight model record
# ----------------------------

pca_record <- list(
  chapter_name = "Chapter 12 - Principal Components Analysis",
  predictors_used = names(pca_predictors),
  data_summary = data_summary,
  variance_explained = variance_explained,
  top_loadings = top_loadings,
  pca_fit = pca_fit,
  note = "PCA was run on standardized numeric predictors from the training data only."
)

saveRDS(
  pca_record,
  file.path(model_dir, "ch12_pca_record.rds")
)

# ----------------------------
# 13. Report notes
# ----------------------------

pc1_text <- paste_top_loadings(loadings_table, "PC1", 5)
pc2_text <- paste_top_loadings(loadings_table, "PC2", 5)
pc3_text <- paste_top_loadings(loadings_table, "PC3", 5)

pc1_variance <- variance_explained %>%
  filter(PC == "PC1") %>%
  pull(proportion_variance)

pc2_variance <- variance_explained %>%
  filter(PC == "PC2") %>%
  pull(proportion_variance)

pc3_variance <- variance_explained %>%
  filter(PC == "PC3") %>%
  pull(proportion_variance)

first_two_variance <- variance_explained %>%
  filter(PC == "PC2") %>%
  pull(cumulative_variance)

first_three_variance <- variance_explained %>%
  filter(PC == "PC3") %>%
  pull(cumulative_variance)

report_notes <- c(
  "Chapter 12 Report Notes",
  "",
  "This chapter used principal components analysis on the numeric predictors from the training data.",
  "",
  "The response variables were not included in the PCA. The goal was to understand the structure among the numeric predictors, not to directly predict lap_time_s or pit_next_lap.",
  "",
  "All variables were standardized before PCA. This matters because PCA depends on variance, and variables with larger scales could dominate if the data were not standardized.",
  "",
  paste0(
    "The PCA used ",
    nrow(pca_predictors),
    " rows and ",
    ncol(pca_predictors),
    " numeric predictors."
  ),
  "",
  paste0(
    "PC1 explained ",
    scales::percent(pc1_variance, accuracy = 0.1),
    " of the variance. PC2 explained ",
    scales::percent(pc2_variance, accuracy = 0.1),
    ". PC3 explained ",
    scales::percent(pc3_variance, accuracy = 0.1),
    "."
  ),
  "",
  paste0(
    "The first two PCs explained ",
    scales::percent(first_two_variance, accuracy = 0.1),
    " of the total variance. The first three PCs explained ",
    scales::percent(first_three_variance, accuracy = 0.1),
    "."
  ),
  "",
  paste0(
    "The strongest PC1 loadings were: ",
    pc1_text,
    "."
  ),
  "",
  paste0(
    "The strongest PC2 loadings were: ",
    pc2_text,
    "."
  ),
  "",
  paste0(
    "The strongest PC3 loadings were: ",
    pc3_text,
    "."
  ),
  "",
  "Real-world decision: PCA is useful here for summarizing correlated lap-level predictors. It is not the final predictive model by itself, but it helps show whether the numeric predictors can be reduced into a smaller number of components."
)

writeLines(
  report_notes,
  file.path(table_dir, "ch12_report_notes.txt")
)

cat("\nReport notes:\n")
cat(paste(report_notes, collapse = "\n"))

# ----------------------------
# 14. Final confirmation
# ----------------------------

cat("\n\n13_pca.R ran successfully.\n")
cat("Chapter 12 tables saved to: ", table_dir, "\n", sep = "")
cat("Chapter 12 figures saved to: ", figure_dir, "\n", sep = "")
cat("Chapter 12 model record saved to: ", model_dir, "\n", sep = "")
cat("Report notes saved to: ", file.path(table_dir, "ch12_report_notes.txt"), "\n", sep = "")
cat("PCA used standardized numeric predictors from the training data only.\n")