# EC 522: Economic Forecasting — Problem Set 4
# Scheryll Ouma

library(tidyverse)
library(fpp3)
library(BVAR)
# =============================================================================
# Part 2 – ARIMA vs. ETS
# =============================================================================

# ---- 2.1a Read the data and build a tsibble ----
liquor_raw <- read_csv("data/MRTSSM4453USN.csv",
                       na = ".",
                       col_types = cols(
                         observation_date = col_date(format = "%Y-%m-%d"),
                         MRTSSM4453USN = col_double()
                       ))

liquor <- liquor_raw |>
  transmute(Month = yearmonth(observation_date), sales = MRTSSM4453USN) |>
  as_tsibble(index = Month) |>
  filter(Month <= yearmonth("2019 Dec"))

has_gaps(liquor)
# ---- 2.1b Time plot and season plot ----
liquor |>
  autoplot(sales) +
  labs(title = "U.S. Retail Sales: Beer, Wine, and Liquor Stores (NSA)",
       x = "Month", y = "Millions of dollars")

liquor |>
  gg_season(log(sales), labels = "both") +
  labs(title = "Seasonal Plot: log of Liquor Store Sales",
       x = "Month", y = "log(sales)")
# ---- 2.2a Fit ARIMA and ETS to log(sales) ----
liquor_fit <- liquor |>
  model(arima = ARIMA(log(sales)),
        ets   = ETS(log(sales)))

liquor_fit |> select(arima) |> report()
liquor_fit |> select(ets) |> report()

# ---- 2.2b ARIMA residual diagnostics ----
liquor_fit |> select(arima) |> gg_tsresiduals()

liquor_fit |>
  select(arima) |>
  augment() |>
  features(.innov, ljung_box, lag = 24, dof = 6)
# ---- 2.3a Expanding-window TS-CV (SLOW: 10+ minutes) ----
liquor_cv <- liquor |>
  stretch_tsibble(.init = 120, .step = 1) |>
  filter(.id < max(.id))

n_distinct(liquor_cv$.id)   # number of forecast origins

# Run the model search on several cores at once (much faster)
library(future)
plan(multisession, workers = max(1, parallelly::availableCores() - 1))

liquor_cv_fit <- liquor_cv |>
  model(snaive = SNAIVE(sales),
        ets    = ETS(log(sales)),
        arima  = ARIMA(log(sales)))

plan(sequential)   # back to normal, single-core
saveRDS(liquor_cv_fit, "results/liquor_cv_fit.rds")
# In a new session, skip the fit above and use:
# liquor_cv_fit <- readRDS("results/liquor_cv_fit.rds")

# ---- 2.3b One-step-ahead forecasts and accuracy ----
liquor_cv_fc <- liquor_cv_fit |> forecast(h = 1)

range(liquor_cv_fc$Month)   # first and last forecast dates

liquor_acc <- liquor_cv_fc |>
  accuracy(liquor, measures = list(RMSE = RMSE, MAE = MAE))
liquor_acc
saveRDS(liquor_acc, "results/liquor_acc.rds")

# ---- 2.4 Diebold–Mariano tests ----
liquor_errs <- liquor_cv_fc |>
  as_tibble() |>
  left_join(as_tibble(liquor) |> select(Month, actual = sales),
            by = "Month") |>
  mutate(e = actual - .mean) |>
  select(Month, .model, e) |>
  pivot_wider(names_from = .model, values_from = e)

# ARIMA vs. ETS
dm_arima_ets <- forecast::dm.test(liquor_errs$arima, liquor_errs$ets,
                                  h = 1, power = 2)
dm_arima_ets

# ARIMA vs. seasonal random walk benchmark
dm_arima_snaive <- forecast::dm.test(liquor_errs$arima, liquor_errs$snaive,
                                     h = 1, power = 2)
dm_arima_snaive

# Save for the write-up
saveRDS(liquor_errs, "results/liquor_errs.rds")
saveRDS(list(arima_ets = dm_arima_ets, arima_snaive = dm_arima_snaive),
        "results/liquor_dm.rds")
# =============================================================================
# Part 3 – Forecast Combination
# =============================================================================

# ---- 3a Equal-weight combination of ARIMA and ETS ----
liquor_combo_fc <- liquor_cv_fit |>
  mutate(combo = (arima + ets) / 2) |>
  forecast(h = 1)

combo_acc <- liquor_combo_fc |>
  accuracy(liquor, measures = list(RMSE = RMSE, MAE = MAE))
combo_acc
saveRDS(combo_acc, "results/combo_acc.rds")

# ---- 3b Correlation of ARIMA and ETS forecast errors ----
cor(liquor_errs$arima, liquor_errs$ets)