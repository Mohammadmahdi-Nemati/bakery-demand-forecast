"""Gesamter Ablauf: Daten laden → Ausverkäufe korrigieren → Merkmale → Modelle → Bewertung."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import censoring, data, evaluate, features, models

TEST_START = pd.Timestamp("2026-05-04")  # letzte 8 Wochen (Mo. 04.05. bis Di. 30.06.2026)
WARMUP_END = pd.Timestamp("2026-01-29")  # erste 4 Wochen haben noch keine vollständigen Lags


@dataclass
class Results:
    test: pd.DataFrame
    accuracy: pd.DataFrame
    policies: pd.DataFrame
    policies_holiday: pd.DataFrame
    imputation: dict
    sims: dict = field(default_factory=dict)


def run(data_dir: str | Path, test_start: pd.Timestamp = TEST_START) -> Results:
    panel = data.load_daily_panel(data_dir)
    hourly = data.load_hourly_sales(data_dir)
    truth = data.load_true_demand(data_dir)

    # 1) Ausverkaufs-Korrektur (Profile nur aus der Trainingszeit)
    profiles = censoring.sales_profiles(panel, hourly, until=test_start)
    panel = censoring.impute_demand(panel, profiles)
    panel = panel.merge(truth, on=["branch_id", "product_id", "date"], how="left")

    # 2) Merkmale: einmal auf Basis der korrigierten Nachfrage, einmal naiv auf Basis des Absatzes
    feat = features.add_features(panel, target="demand_est")
    feat_naive = features.add_features(panel, target="sold")
    feat["current_practice"] = features.current_practice(feat)
    feat["holiday_rule"] = features.holiday_rule(feat)

    in_train = (feat["date"] >= WARMUP_END) & (feat["date"] < test_start)
    train, test = feat[in_train], feat[feat["date"] >= test_start].copy()
    train_naive = feat_naive[in_train]
    test_naive = feat_naive[feat_naive["date"] >= test_start]

    # 3) Modelle
    point = models.fit_point(train, "demand_est")
    point_naive = models.fit_point(train_naive, "sold")
    quantiles = models.fit_quantiles(train, "demand_est")

    test["forecast"] = point.predict(test[features.FEATURES])
    test["forecast_naive"] = point_naive.predict(test_naive[features.FEATURES])
    test["ratio"] = models.critical_ratio(test["price"], test["unit_cost"])
    test["model_newsvendor"] = models.newsvendor_quantity(quantiles, test, test["ratio"])
    test["baseline_mean4"] = test["current_practice"] / 1.12
    # Kombination: Modell an normalen Tagen, Feiertagsregel an Feiertagen
    # (im Training gab es nur 3 Feiertage, zu wenig, damit das Modell sie selbst lernt)
    test["model_plus_holiday"] = np.where(test["is_holiday"] == 1, test["holiday_rule"], test["model_newsvendor"])

    # 4) Genauigkeit der Nachfrageprognose (gegen die echte Nachfrage)
    acc = pd.DataFrame(
        {
            "Methode": [
                "Ø letzte 4 gleiche Wochentage (Absatz)",
                "Modell ohne Ausverkaufs-Korrektur",
                "Modell mit Ausverkaufs-Korrektur",
            ],
            "WAPE_%": [
                100 * evaluate.wape(test["true_demand"], test["baseline_mean4"]),
                100 * evaluate.wape(test["true_demand"], test["forecast_naive"]),
                100 * evaluate.wape(test["true_demand"], test["forecast"]),
            ],
            "Bias_%": [
                100 * (test[c].sum() / test["true_demand"].sum() - 1)
                for c in ("baseline_mean4", "forecast_naive", "forecast")
            ],
        }
    ).round(1)

    # 5) Planungsregeln durchrechnen
    sims = {
        "Heutige Planung (Ø 4 Wochen + 12 %)": evaluate.simulate_policy(test, test["current_practice"]),
        "Heutige Planung + Feiertagsregel": evaluate.simulate_policy(test, test["holiday_rule"]),
        "Modell + Newsvendor-Menge": evaluate.simulate_policy(test, test["model_newsvendor"]),
        "Modell + Newsvendor + Feiertagsregel": evaluate.simulate_policy(test, test["model_plus_holiday"]),
    }
    policies = pd.DataFrame({k: evaluate.summarize(v) for k, v in sims.items()}).T
    policies_holiday = pd.DataFrame(
        {k: evaluate.summarize(v[v["is_holiday"] == 1]) for k, v in sims.items()}
    ).T

    # 6) Wie gut schätzt die Korrektur die Nachfrage an Ausverkaufstagen? (gesamte Historie)
    so = panel[panel["stockout"]]
    imputation = {
        "ausverkaufstage": int(len(so)),
        "absatz_unterschaetzt_um_%": round(100 * (1 - so["sold"].sum() / so["true_demand"].sum()), 1),
        "korrektur_unterschaetzt_um_%": round(100 * (1 - so["demand_est"].sum() / so["true_demand"].sum()), 1),
    }
    return Results(test, acc, policies, policies_holiday, imputation, sims)
