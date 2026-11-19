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

# =============================================================================
# Part 4 – A Medium-Scale VAR vs. a BVAR
# =============================================================================

# ---- 4.1a Read FRED-QD ----
qd_raw <- read_csv("data/fredqd_2026-09.csv", show_col_types = FALSE) |>
  slice(-(1:2))

# ---- 4.1b Build the eight series ----
growth <- function(x) 100 * difference(log(x))

qd <- qd_raw |>
  mutate(date = as.Date(sasdate, "%m/%d/%Y")) |>
  transmute(date,
            gdp    = growth(GDPC1),       # real GDP
            cons   = growth(PCECC96),     # real consumption
            inv    = growth(GPDIC1),      # real investment
            emp    = growth(PAYEMS),      # payroll employment
            infl   = growth(GDPCTPI),     # GDP price index
            unrate = difference(UNRATE),  # unemployment rate
            ffr    = difference(FEDFUNDS),# federal funds rate
            gs10   = difference(GS10)) |> # 10-year Treasury yield
  filter(date >= as.Date("1960-03-01"), date <= as.Date("2019-12-01"))

y <- qd |> select(-date) |> as.matrix()
nrow(y)
head(y)

# ---- 4.1c TS-CV setup ----
N <- nrow(y)
origins <- 120:(N - 1)
actual <- y[121:N, ]

nrow(actual)          # forecasts per variable
qd$date[c(121, N)]    # dates of first and last forecast

# ---- 4.2 OLS VARs ----
var_ols_fc <- function(z, p) {
  e <- embed(z, p + 1)            # columns: y_t, then its lags 1 to p
  M <- ncol(z)                    # number of variables
  B <- coef(lm(e[, 1:M] ~ e[, -(1:M)]))
  x <- c(1, t(z[nrow(z):(nrow(z) - p + 1), ]))   # constant, latest p lags
  x %*% B
}

fc_var1 <- NULL
fc_var4 <- NULL
for (T in origins) {
  fc_var1 <- rbind(fc_var1, var_ols_fc(y[1:T, ], 1))
  fc_var4 <- rbind(fc_var4, var_ols_fc(y[1:T, ], 4))
}

rmse_by_var <- function(f) sqrt(colMeans((actual - f)^2))
rmse_by_var(fc_var1)
rmse_by_var(fc_var4)

sum(rmse_by_var(fc_var1) < rmse_by_var(fc_var4))   # variables where VAR(1) wins

# ---- 4.3a BVAR(4), Minnesota prior, lambda = 0.3 (1-2 min) ----
bvar_cv <- function(lambda) {
  pr <- bv_priors(hyper = "lambda",
                  mn = bv_mn(lambda = bv_lambda(mode = lambda, sd = 0.4,
                                                min = 0.999 * lambda,
                                                max = 1.001 * lambda),
                             b = 0))
  fc <- NULL
  for (T in origins) {
    fit_T <- bvar(y[1:T, ], lags = 4, priors = pr,
                  n_draw = 2000, n_burn = 1000, verbose = FALSE)
    x <- c(1, t(y[T:(T - 3), ]))         # constant and 4 lags of each variable
    fc <- rbind(fc, x %*% coef(fit_T))   # forecast of y_{T+1}
  }
  fc
}

set.seed(422)
fc_bvar_03 <- bvar_cv(0.3)
saveRDS(fc_bvar_03, "results/fc_bvar_03.rds")

# ---- 4.3b Tighter prior, lambda = 0.05 (1-2 min) ----
set.seed(422)
fc_bvar_005 <- bvar_cv(0.05)
saveRDS(fc_bvar_005, "results/fc_bvar_005.rds")

var_results <- tibble(
  Variable = colnames(y),
  `VAR(1)` = rmse_by_var(fc_var1),
  `VAR(4)` = rmse_by_var(fc_var4),
  `BVAR(4), lambda = 0.3` = rmse_by_var(fc_bvar_03),
  `BVAR(4), lambda = 0.05` = rmse_by_var(fc_bvar_005)
)
saveRDS(var_results, "results/var_results.rds")
var_results |> knitr::kable(digits = 3)

# ---- 4.3c DM tests: VAR(4) vs. BVAR(4), lambda = 0.3 ----
# GDP growth (column 1)
e_var4 <- actual[, "gdp"] - fc_var4[, 1]
e_bvar <- actual[, "gdp"] - fc_bvar_03[, 1]
dm_gdp <- forecast::dm.test(e_var4, e_bvar, h = 1, power = 2)
dm_gdp

# Second variable: consumption growth (column 2)
e_var4_cons <- actual[, "cons"] - fc_var4[, 2]
e_bvar_cons <- actual[, "cons"] - fc_bvar_03[, 2]
dm_cons <- forecast::dm.test(e_var4_cons, e_bvar_cons, h = 1, power = 2)
dm_cons

saveRDS(list(gdp = dm_gdp, cons = dm_cons), "results/var_dm.rds")

# =============================================================================
# Part 5 – A Factor Model with FRED-MD
# =============================================================================

# ---- 5.1a Read FRED-MD and the transformation codes ----
md_raw <- read_csv("data/fredmd_2026-09.csv", show_col_types = FALSE)
tcode <- unlist(md_raw[1, -1])
md <- md_raw[-1, ] |> mutate(date = as.Date(sasdate, "%m/%d/%Y"))

transform_series <- function(x, code) {
  d <- function(v) c(NA, diff(v))
  switch(as.character(code),
         "1" = x,                      # level
         "2" = d(x),                   # first difference
         "3" = d(d(x)),                # second difference
         "4" = log(x),                 # log
         "5" = 100 * d(log(x)),        # growth rate
         "6" = 100 * d(d(log(x))),     # change in the growth rate
         "7" = d(x / dplyr::lag(x) - 1))
}

# ---- 5.1b Transform, restrict sample, drop series with NAs ----
fm <- map2_dfc(md[names(tcode)], tcode, transform_series) |>
  mutate(date = md$date, .before = 1) |>
  filter(date >= as.Date("1960-03-01"), date <= as.Date("2019-12-01"))
fm <- fm[, colSums(is.na(fm)) == 0]
dim(fm)

# ---- 5.1c Transformation codes for four series ----
tcode[c("INDPRO", "PAYEMS", "CPIAUCSL", "FEDFUNDS")]

# ---- 5.2a Principal components ----
Z <- fm |> select(-date) |> as.matrix()
pc <- prcomp(scale(Z))
summary(pc)$importance[, 1:5]

# ---- 5.2b Plot the first factor ----
F1 <- pc$x[, 1] * sign(cor(pc$x[, 1], fm$INDPRO))
tibble(date = fm$date, F1 = F1) |>
  ggplot(aes(date, F1)) +
  geom_line() +
  labs(title = "First Factor from FRED-MD", x = NULL, y = NULL) +
  theme_minimal()

# ---- 5.2c Series most correlated with factors 1 and 2 ----
sort(abs(cor(pc$x[, 1], Z))[1, ], decreasing = TRUE)[1:8]
sort(abs(cor(pc$x[, 2], Z))[1, ], decreasing = TRUE)[1:8]