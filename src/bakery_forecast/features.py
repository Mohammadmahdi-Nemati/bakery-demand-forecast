"""Merkmale (Features) für die Prognose.

Wichtig: Die Produktionsmenge für Tag t wird am Abend von Tag t-1 festgelegt.
Deshalb nutzt jedes Merkmal nur Informationen bis einschließlich t-1.
"""

import numpy as np
import pandas as pd

LAG_DAYS = (1, 7, 14, 21, 28)
CATEGORICAL = ["branch_id", "product_id", "category_id"]
# Bewusst schlank: In Experimenten waren lag_1 und rollierende Mittel eher Rauschen
# (siehe README, Abschnitt "Was nicht funktioniert hat").
FEATURES = CATEGORICAL + [
    "weekday", "is_holiday", "is_sundaylike",
    "same_weekday_mean4", "sunday_mean4", "share_stockout_28",
]


def add_features(df: pd.DataFrame, target: str = "demand_est") -> pd.DataFrame:
    """Kalender-Merkmale und Verzögerungen (Lags) der Zielgröße je Filiale und Produkt."""
    out = df.copy()
    out["weekday"] = out["date"].dt.weekday
    out["month"] = out["date"].dt.month
    out["is_sundaylike"] = (out["weekday"] == 6) | out["is_holiday"]

    keys = ["branch_id", "product_id"]
    base = out[keys + ["date", target, "stockout"]]
    for k in LAG_DAYS:
        lagged = base[keys + ["date", target]].copy()
        lagged["date"] = lagged["date"] + pd.Timedelta(days=k)
        out = out.merge(lagged.rename(columns={target: f"lag_{k}"}), on=keys + ["date"], how="left")

    out["same_weekday_mean4"] = out[["lag_7", "lag_14", "lag_21", "lag_28"]].mean(axis=1)
    out["sunday_mean4"] = _last_sundays_mean(base, target, keys).reindex(
        pd.MultiIndex.from_frame(out[keys + ["date"]])).to_numpy()

    # Rollierende Kennzahlen über die letzten 7 bzw. 28 Kalendertage (ohne den Tag selbst)
    rolled = []
    for _, g in base.groupby(keys):
        s = g.set_index("date").asfreq("D")
        r = pd.DataFrame(index=s.index)
        r["rolling_mean_7"] = s[target].rolling(7, min_periods=1).mean().shift(1)
        r["share_stockout_28"] = s["stockout"].astype(float).rolling(28, min_periods=1).mean().shift(1)
        r[keys[0]], r[keys[1]] = g[keys[0]].iloc[0], g[keys[1]].iloc[0]
        rolled.append(r.reset_index())
    out = out.merge(pd.concat(rolled), on=keys + ["date"], how="left")

    out["is_holiday"] = out["is_holiday"].astype(int)
    out["is_sundaylike"] = out["is_sundaylike"].astype(int)
    return out


def _last_sundays_mean(base: pd.DataFrame, target: str, keys: list[str]) -> pd.Series:
    """Ø der letzten 4 Sonntage vor jedem Datum. Hilft dem Modell an Feiertagen."""
    out = []
    for key, g in base.groupby(keys):
        s = g.set_index("date")[target].asfreq("D")
        sundays = s[s.index.weekday == 6].dropna()
        m = sundays.rolling(4, min_periods=1).mean().shift(1).reindex(s.index, method="ffill")
        # Ein Sonntag selbst darf seinen eigenen Wert nicht sehen: shift(1) oben, ffill trägt Vorwochen weiter
        idx = pd.MultiIndex.from_arrays([[key[0]] * len(m), [key[1]] * len(m), m.index], names=keys + ["date"])
        out.append(pd.Series(m.to_numpy(), index=idx))
    return pd.concat(out)


def current_practice(df: pd.DataFrame, buffer: float = 1.12) -> pd.Series:
    """Heutige Planung: Ø-Absatz der letzten 4 gleichen Wochentage plus Puffer.

    Nutzt den VERKAUFTEN Absatz, weil die Bäckerei nur den kennt.
    """
    sold_lags = []
    keys = ["branch_id", "product_id"]
    for k in (7, 14, 21, 28):
        lagged = df[keys + ["date", "sold"]].copy()
        lagged["date"] = lagged["date"] + pd.Timedelta(days=k)
        sold_lags.append(df[keys + ["date"]].merge(lagged, on=keys + ["date"], how="left")["sold"])
    mean4 = pd.concat(sold_lags, axis=1).mean(axis=1)
    return np.ceil((mean4 * buffer).round(6)).fillna(0).astype(int)  # round: 100 * 1.12 = 112.00000000000001


def holiday_rule(df: pd.DataFrame, buffer: float = 1.12) -> pd.Series:
    """Heutige Planung, aber an Feiertagen wie an einem Sonntag planen (Ø der letzten 4 Sonntage)."""
    plan = current_practice(df, buffer)
    sunday = _last_sundays_mean(df[["branch_id", "product_id", "date", "sold"]], "sold",
                                ["branch_id", "product_id"])
    sunday = sunday.reindex(pd.MultiIndex.from_frame(df[["branch_id", "product_id", "date"]])).to_numpy()
    on_holiday = df["is_holiday"].astype(bool).to_numpy() & ~np.isnan(sunday)
    plan = plan.to_numpy().copy()
    plan[on_holiday] = np.ceil(np.round(sunday[on_holiday] * buffer, 6))
    return pd.Series(plan, index=df.index).astype(int)
