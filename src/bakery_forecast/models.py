"""Prognosemodelle.

1. Punktprognose (Poisson-Verlust): erwartete Nachfrage, für die Genauigkeitsmessung.
2. Quantilsprognosen: Wie viel sollte man backen? (Newsvendor-Prinzip)

Newsvendor: Ein übrig gebliebenes Teil kostet die Herstellkosten c. Ein fehlendes Teil kostet die
entgangene Marge (p - c). Optimal ist, das Quantil  q* = (p - c) / p  der Nachfrage zu backen.
Bei Weizenbrötchen (p = 0,45 €, c = 0,12 €) ist q* ≈ 0,73: An 73 % der Tage soll die Menge reichen.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import CATEGORICAL, FEATURES

QUANTILES = (0.65, 0.70, 0.75)


def _model(**kwargs) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        categorical_features=[FEATURES.index(c) for c in CATEGORICAL],
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=0,
        **kwargs,
    )


def fit_point(train: pd.DataFrame, target: str) -> HistGradientBoostingRegressor:
    return _model(loss="poisson").fit(train[FEATURES], train[target])


def fit_quantiles(train: pd.DataFrame, target: str) -> dict[float, HistGradientBoostingRegressor]:
    return {q: _model(loss="quantile", quantile=q).fit(train[FEATURES], train[target]) for q in QUANTILES}


def critical_ratio(price: pd.Series, unit_cost: pd.Series) -> pd.Series:
    return (price - unit_cost) / price


def newsvendor_quantity(models: dict, X: pd.DataFrame, ratio: pd.Series) -> np.ndarray:
    """Pro Zeile das Quantilmodell nehmen, das dem kritischen Verhältnis des Produkts am nächsten liegt."""
    qs = np.array(sorted(models))
    preds = np.column_stack([models[q].predict(X[FEATURES]) for q in qs])
    idx = np.abs(ratio.to_numpy()[:, None] - qs[None, :]).argmin(axis=1)
    chosen = preds[np.arange(len(X)), idx]
    return np.ceil(np.clip(chosen, 0, None)).astype(int)
