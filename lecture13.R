# ---- Activity 1: Forecast combination ----
library(fpp3)

# 1. Arrivals to Australia from Japan
japan <- aus_arrivals |> filter(Origin == "Japan")
japan |> autoplot(Arrivals)

# 2. TS-CV: first training sample 1981 Q1 - 1999 Q4 (76 quarters), h = 1.
#    The filter() drops the last training sample, which has no
#    observation left to forecast.
japan |>
  stretch_tsibble(.init = 76, .step = 1) |>
  filter(.id <= nrow(japan) - 76) |>
  model(arima = ARIMA(Arrivals),
        ets   = ETS(Arrivals)) |>
  mutate(combo = (arima + ets) / 2) |>
  forecast(h = 1) |>
  accuracy(japan, measures = list(RMSE = RMSE, MAE = MAE))