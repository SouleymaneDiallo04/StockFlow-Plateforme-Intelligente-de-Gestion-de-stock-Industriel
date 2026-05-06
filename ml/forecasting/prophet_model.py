"""
ml/forecasting/prophet_model.py
--------------------------------
Wrapper Prophet avec features industrielles.

Avantage de Prophet sur SARIMAX ici :
- Gestion native des valeurs manquantes (fréquent dans les données industrielles)
- Décomposition additive explicitement visualisable — utile pour l'analyse exploratoire
- Ajout de régresseurs exogènes (événements) simple et robuste
- Estimateur bayésien (Stan) — incertitude mieux calibrée sur séries courtes

Inconvénient accepté : plus lent que SARIMAX sur grandes séries.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from .base import BaseForecaster


class ProphetForecaster(BaseForecaster):

    def __init__(
        self,
        seasonality_mode: str = 'multiplicative',  # Cohérent avec notre générateur multiplicatif
        confidence_level: float = 0.90,
        add_weekly: bool = True,
        add_monthly: bool = True,
        changepoint_prior_scale: float = 0.05,     # Régularisation de la tendance
        seasonality_prior_scale: float = 10.0,
    ):
        super().__init__(confidence_level)
        self.seasonality_mode = seasonality_mode
        self.add_weekly = add_weekly
        self.add_monthly = add_monthly
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self._prophet = None
        self._training_df = None

    @property
    def model_name(self) -> str:
        return "Prophet"

    def _fit_internal(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("prophet requis : pip install prophet")

        df = pd.DataFrame({
            'ds': series.index,
            'y':  series.values
        })

        # Ajout des régresseurs exogènes
        regressors = []
        if exog is not None:
            for col in exog.columns:
                df[col] = exog[col].values
                regressors.append(col)

        self._prophet = Prophet(
            seasonality_mode=self.seasonality_mode,
            interval_width=self.confidence_level,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            weekly_seasonality=self.add_weekly,
            yearly_seasonality=False,  # Notre horizon est < 2 ans, pas de saisonnalité annuelle robuste
            daily_seasonality=False,
        )

        # Saisonnalité mensuelle customisée (30.44 jours)
        if self.add_monthly:
            self._prophet.add_seasonality(
                name='monthly',
                period=30.44,
                fourier_order=5,
            )

        for reg in regressors:
            self._prophet.add_regressor(reg, standardize=True)

        import logging
        logging.getLogger('prophet').setLevel(logging.ERROR)

        self._prophet.fit(df)
        self._training_df = df
        self._regressors = regressors

    def _predict_internal(
        self,
        horizon: int,
        future_exog: Optional[pd.DataFrame] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        future = self._prophet.make_future_dataframe(periods=horizon)
        future = future.tail(horizon).reset_index(drop=True)

        if future_exog is not None and self._regressors:
            for col in self._regressors:
                if col in future_exog.columns:
                    future[col] = future_exog[col].values[:horizon]
                else:
                    future[col] = 0.0

        forecast = self._prophet.predict(future)
        point = forecast['yhat'].values
        lower = forecast['yhat_lower'].values
        upper = forecast['yhat_upper'].values

        return point, lower, upper

    def get_components(self) -> Optional[pd.DataFrame]:
        """Retourne les composantes de la décomposition — utile pour la visualisation."""
        if self._prophet is None or self._training_df is None:
            return None
        future = self._prophet.make_future_dataframe(periods=0)
        if self._regressors:
            for col in self._regressors:
                future[col] = self._training_df[col].values if col in self._training_df.columns else 0
        return self._prophet.predict(future)
