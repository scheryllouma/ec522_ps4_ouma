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