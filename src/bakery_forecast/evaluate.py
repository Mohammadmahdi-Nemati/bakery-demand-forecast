"""Bewertung: Was hätte eine Planungsregel in der Testperiode gekostet?

Da die Daten simuliert sind, kennen wir die echte Nachfrage (auch an Ausverkaufstagen).
Damit lässt sich jede Planungsregel fair durchrechnen.
"""

import numpy as np
import pandas as pd


def simulate_policy(df: pd.DataFrame, produce: pd.Series | np.ndarray) -> pd.DataFrame:
    """Für eine Produktionsmenge je Zeile: Absatz, Abfall, entgangene Verkäufe und Kosten."""
    out = df[["branch_id", "product_id", "date", "is_holiday", "price", "unit_cost", "true_demand"]].copy()
    out["produce"] = np.asarray(produce)
    out["sold"] = np.minimum(out["produce"], out["true_demand"])
    out["waste"] = out["produce"] - out["sold"]
    out["lost"] = out["true_demand"] - out["sold"]
    out["waste_cost"] = out["waste"] * out["unit_cost"]
    out["lost_margin"] = out["lost"] * (out["price"] - out["unit_cost"])
    out["profit"] = out["sold"] * out["price"] - out["produce"] * out["unit_cost"]
    return out


def summarize(sim: pd.DataFrame) -> dict:
    produced = sim["produce"].sum()
    demand = sim["true_demand"].sum()
    return {
        "produziert": int(produced),
        "abfallquote_%": round(100 * sim["waste"].sum() / produced, 1),
        "abfall_kosten_eur": round(sim["waste_cost"].sum(), 0),
        "lieferfaehigkeit_%": round(100 * sim["sold"].sum() / demand, 1),
        "entgangene_marge_eur": round(sim["lost_margin"].sum(), 0),
        "gesamtkosten_eur": round(sim["waste_cost"].sum() + sim["lost_margin"].sum(), 0),
        "gewinn_eur": round(sim["profit"].sum(), 0),
    }


def wape(actual: pd.Series, forecast: pd.Series | np.ndarray) -> float:
    """Weighted Absolute Percentage Error: Summe |Fehler| / Summe Ist. Robust bei kleinen Mengen."""
    return float(np.abs(np.asarray(actual) - np.asarray(forecast)).sum() / np.asarray(actual).sum())
