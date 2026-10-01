"""Ausverkaufs-Korrektur (Zensierung).

Problem: Ist ein Produkt um 10 Uhr ausverkauft, zeigen die Kassendaten nur den Absatz bis 10 Uhr.
Die echte Nachfrage war höher. Wer nur auf den Absatz schaut, plant deshalb immer zu knapp.

Idee: An Tagen OHNE Ausverkauf sieht man, welcher Anteil des Tagesabsatzes eines Produkts
typischerweise bis zu einer bestimmten Uhrzeit verkauft ist (z. B. Weizenbrötchen: rund zwei Drittel bis 11 Uhr).
An einem Ausverkaufstag gilt dann:  geschätzte Nachfrage = Absatz / Anteil bis zur letzten Verkaufsstunde.
"""

import numpy as np
import pandas as pd

MIN_SHARE = 0.2  # höchstens Faktor 5, damit sehr frühe Ausverkäufe nicht explodieren


def day_type(dates: pd.Series, is_holiday: pd.Series) -> pd.Series:
    """Werktag / Samstag / Sonntag. Feiertage zählen wie Sonntag."""
    wd = dates.dt.weekday
    out = np.where(wd == 5, "sa", "wd")
    out = np.where((wd == 6) | is_holiday.to_numpy(), "so", out)
    return pd.Series(out, index=dates.index)


def sales_profiles(panel: pd.DataFrame, hourly: pd.DataFrame, until: pd.Timestamp) -> pd.DataFrame:
    """Kumulierter Anteil des Tagesabsatzes bis Ende jeder Stunde.

    Je (Filiale, Produkt, Tagestyp, Stunde), berechnet nur aus Tagen ohne Ausverkauf vor `until`.
    """
    ok_days = panel[(~panel["stockout"]) & (panel["date"] < until) & (panel["sold"] > 0)].copy()
    ok_days["day_type"] = day_type(ok_days["date"], ok_days["is_holiday"])
    h = hourly.merge(ok_days[["branch_id", "product_id", "date", "day_type", "sold"]],
                     on=["branch_id", "product_id", "date"])
    h = h.sort_values("hour")
    h["cum"] = h.groupby(["branch_id", "product_id", "date"])["quantity"].cumsum()
    h["cum_share"] = h["cum"] / h["sold"]

    # Für jede Stunde den Anteil bis dahin; fehlende Stunden (kein Verkauf) mit dem letzten Wert füllen
    rows = []
    for (b, p, dt), g in h.groupby(["branch_id", "product_id", "day_type"]):
        per_day = g.pivot_table(index="date", columns="hour", values="cum_share")
        per_day = per_day.reindex(columns=range(6, 21)).ffill(axis=1).fillna(0.0)
        mean = per_day.mean(axis=0)
        rows += [(b, p, dt, hr, share) for hr, share in mean.items()]
    return pd.DataFrame(rows, columns=["branch_id", "product_id", "day_type", "hour", "cum_share"])


def share_until(profiles: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
    """Erwarteter Anteil des Tagesabsatzes bis zur Uhrzeit `last_sale_time` (linear innerhalb der Stunde)."""
    k = keys.copy()
    k["hour"] = np.floor(k["last_sale_time"]).astype("Int64")
    k["frac"] = k["last_sale_time"] - k["hour"].astype(float)
    cols = ["branch_id", "product_id", "day_type", "hour"]
    end = k.merge(profiles, on=cols, how="left")["cum_share"]
    prev = profiles.assign(hour=profiles["hour"] + 1)
    start = k.merge(prev, on=cols, how="left")["cum_share"].fillna(0.0)
    share = start + k["frac"].to_numpy() * (end - start)
    return share.fillna(1.0).to_numpy()


def impute_demand(panel: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Fügt `demand_est` hinzu: Absatz an normalen Tagen, hochgerechnete Nachfrage an Ausverkaufstagen."""
    out = panel.copy()
    out["day_type"] = day_type(out["date"], out["is_holiday"])
    share = np.ones(len(out))
    so = out["stockout"].to_numpy() & out["last_sale_time"].notna().to_numpy()
    share[so] = share_until(profiles, out.loc[so, ["branch_id", "product_id", "day_type", "last_sale_time"]])
    share = np.clip(share, MIN_SHARE, 1.0)
    out["demand_est"] = np.maximum(out["sold"] / share, out["sold"])  # nie unter dem Absatz
    return out
