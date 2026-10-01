"""Tests mit kleinen, von Hand nachrechenbaren Beispielen."""

import numpy as np
import pandas as pd
import pytest

from bakery_forecast import censoring, evaluate, features, models


def make_panel(values, start="2026-01-05", stockout=None):
    """Eine Zeitreihe (Filiale 1, Produkt 1) mit täglichen Werten."""
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({
        "branch_id": 1, "product_id": 1, "category_id": 1, "date": dates,
        "sold": values, "demand_est": values, "is_holiday": False,
        "stockout": stockout if stockout is not None else [False] * len(values),
    })


def test_same_weekday_mean_uses_only_past_values():
    # 5 Wochen: Wert = Wochennummer * 10 (Woche 1 = 10, ... Woche 5 = 50)
    values = [10 * (i // 7 + 1) for i in range(35)]
    out = features.add_features(make_panel(values))
    last = out.iloc[-1]  # Tag in Woche 5
    assert last["same_weekday_mean4"] == pytest.approx((10 + 20 + 30 + 40) / 4)
    assert last["lag_7"] == 40  # Vorwoche, nicht der Tag selbst


def test_current_practice_adds_buffer_and_rounds_up():
    values = [100] * 35
    plan = features.current_practice(make_panel(values), buffer=1.12)
    assert plan.iloc[-1] == 112
    assert plan.iloc[0] == 0  # ohne Historie keine Planung


def test_imputation_scales_up_early_stockouts():
    panel = make_panel([60], stockout=[True])
    panel["last_sale_time"] = 10.5            # ausverkauft um 10:30
    panel["day_type"] = "wd"
    profiles = pd.DataFrame({
        "branch_id": 1, "product_id": 1, "day_type": "wd",
        "hour": [9, 10], "cum_share": [0.4, 0.8],
    })
    out = censoring.impute_demand(panel.drop(columns=["day_type"]), profiles)
    # Anteil bis 10:30 = 0,4 + 0,5 * (0,8 - 0,4) = 0,6  ->  Nachfrage = 60 / 0,6 = 100
    assert out["demand_est"].iloc[0] == pytest.approx(100)


def test_imputation_leaves_normal_days_unchanged():
    panel = make_panel([42])
    panel["last_sale_time"] = 18.0
    profiles = pd.DataFrame(columns=["branch_id", "product_id", "day_type", "hour", "cum_share"])
    assert censoring.impute_demand(panel, profiles)["demand_est"].iloc[0] == 42


def test_simulate_policy_costs():
    df = pd.DataFrame({
        "branch_id": [1, 1], "product_id": [1, 1], "date": pd.to_datetime(["2026-05-04", "2026-05-05"]),
        "is_holiday": [0, 0], "price": [1.0, 1.0], "unit_cost": [0.3, 0.3], "true_demand": [10, 10],
    })
    sim = evaluate.simulate_policy(df, [12, 8])
    assert sim["waste"].tolist() == [2, 0]
    assert sim["lost"].tolist() == [0, 2]
    assert sim["waste_cost"].sum() == pytest.approx(0.6)       # 2 Stück * 0,30 €
    assert sim["lost_margin"].sum() == pytest.approx(1.4)      # 2 Stück * 0,70 €


def test_critical_ratio():
    assert models.critical_ratio(pd.Series([0.45]), pd.Series([0.12])).iloc[0] == pytest.approx(0.7333, abs=1e-3)


def test_wape():
    assert evaluate.wape(pd.Series([10, 10]), np.array([12, 8])) == pytest.approx(0.2)
