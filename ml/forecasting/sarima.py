"""
ml/forecasting/sarima.py
-------------------------
Wrapper SARIMAX avec sélection automatique de l'ordre (p,d,q)(P,D,Q,s).

Choix : pmdarima.auto_arima pour la sélection — évite la recherche naïve
sur grille complète qui est O(n⁵) en complexité.
Critère de sélection : AIC (pas BIC — BIC pénalise trop les modèles
saisonniers sur séries courtes industrielles).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseForecaster


class SARIMAXForecaster(BaseForecaster):

    def __init__(
        self,
        seasonal_period: int = 7,
        confidence_level: float = 0.90,
        information_criterion: str = 'aic',
        max_p: int = 3,
        max_q: int = 3,
        max_P: int = 2,
        max_Q: int = 2,
    ):
        super().__init__(confidence_level)
        self.seasonal_period = seasonal_period
        self.information_criterion = information_criterion
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self._model = None
        self._fitted_model = None

    @property
    def model_name(self) -> str:
        return "SARIMAX"

    def _fit_internal(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        try:
            import pmdarima as pm
        except ImportError:
            raise ImportError("pmdarima requis : pip install pmdarima")

        # Transformée log pour stabiliser la variance (série toujours positive après +1)
        series_log = np.log1p(series.values)

        self._model = pm.auto_arima(
            series_log,
            X=exog.values if exog is not None else None,
            seasonal=True,
            m=self.seasonal_period,
            information_criterion=self.information_criterion,
            max_p=self.max_p, max_q=self.max_q,
            max_P=self.max_P, max_Q=self.max_Q,
            d=None,   # Ordre d'intégration sélectionné automatiquement
            D=None,
            stepwise=True,      # Algorithme Hyndman-Khandakar — bien plus rapide que grid search
            suppress_warnings=True,
            error_action='ignore',
            n_fits=20,
        )
        self._fitted_model = self._model
        self._series_log = series_log

    def _predict_internal(
        self,
        horizon: int,
        future_exog: Optional[pd.DataFrame] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        alpha = 1 - self.confidence_level
        forecast_log, conf_int_log = self._fitted_model.predict(
            n_periods=horizon,
            X=future_exog.values if future_exog is not None else None,
            return_conf_int=True,
            alpha=alpha,
        )

        # Transformation inverse log1p
        point = np.expm1(forecast_log)
        lower = np.expm1(conf_int_log[:, 0])
        upper = np.expm1(conf_int_log[:, 1])

        return point, lower, upper

    def get_order(self) -> Optional[tuple]:
        """Retourne l'ordre SARIMA sélectionné — utile pour le logging."""
        if self._fitted_model is not None:
            return self._fitted_model.order, self._fitted_model.seasonal_order
        return None
