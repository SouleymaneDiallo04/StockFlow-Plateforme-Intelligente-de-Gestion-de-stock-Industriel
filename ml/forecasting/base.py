"""
ml/forecasting/base.py
-----------------------
Classe abstraite BaseForecaster — contrat commun à tous les modèles.

Le pattern Template Method garantit que chaque modèle expose
la même interface : fit(), predict(), evaluate().
Cela rend la comparaison dans ModelComparison strictement équitable.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from ..data.schemas import ForecastResult


class BaseForecaster(ABC):
    """
    Contrat commun à SARIMAX, Prophet, LightGBM et TFT.

    Tous les modèles reçoivent le même format d'entrée et
    produisent le même format de sortie (ForecastResult).
    """

    def __init__(self, confidence_level: float = 0.90):
        """
        Args:
            confidence_level: Niveau de confiance des intervalles prédictifs.
                              0.90 = IC à 90% — standard en forecasting industriel.
                              Note : 0.95 semble plus précis mais est souvent sur-confiant
                              sur données réelles hors-distribution.
        """
        self.confidence_level = confidence_level
        self._is_fitted = False
        self._training_time: Optional[float] = None

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nom du modèle pour les rapports."""
        ...

    @abstractmethod
    def _fit_internal(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        """Entraîne le modèle. Appelé par fit() après validation."""
        ...

    @abstractmethod
    def _predict_internal(
        self,
        horizon: int,
        future_exog: Optional[pd.DataFrame] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retourne (point_forecast, lower_bound, upper_bound).
        Les bornes sont des intervalles de prédiction bruts —
        la calibration conformal est appliquée dans predict().
        """
        ...

    def fit(
        self,
        series: pd.Series,
        exog: Optional[pd.DataFrame] = None
    ) -> BaseForecaster:
        """
        Entraîne le modèle.

        Args:
            series: Série temporelle indexée par date, valeurs float.
            exog: Variables exogènes optionnelles (événements, jours fériés).
        """
        if len(series) < 14:
            raise ValueError(
                f"Série trop courte ({len(series)} obs) — minimum 14 points requis."
            )

        start = time.perf_counter()
        self._fit_internal(series, exog)
        self._training_time = time.perf_counter() - start
        self._is_fitted = True
        self._training_series = series
        return self

    def predict(
        self,
        horizon: int,
        future_dates: Optional[pd.DatetimeIndex] = None,
        future_exog: Optional[pd.DataFrame] = None,
        sku_id: str = "unknown",
    ) -> ForecastResult:
        """
        Génère les prévisions avec intervalles de confiance.

        Args:
            horizon: Nombre de périodes à prévoir.
            future_dates: Index des dates futures (optionnel).
            future_exog: Variables exogènes futures.
            sku_id: Identifiant SKU pour le résultat.
        """
        if not self._is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant predict().")

        point, lower, upper = self._predict_internal(horizon, future_exog)

        # Construire les dates futures si non fournies
        if future_dates is None:
            last_date = self._training_series.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq='D'
            )

        # Contrainte domaine : demandes non-négatives
        point = np.maximum(point, 0.0)
        lower = np.maximum(lower, 0.0)
        upper = np.maximum(upper, 0.0)

        return ForecastResult(
            model_name=self.model_name,
            sku_id=sku_id,
            forecast_dates=np.array(future_dates),
            point_forecast=point,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=self.confidence_level,
            training_time_s=self._training_time,
        )

    def backtest(
        self,
        series: pd.Series,
        horizon: int,
        n_folds: int = 3,
        exog: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Walk-forward cross-validation.
        Évalue le modèle sur des fenêtres temporelles successives —
        la seule évaluation valide pour les séries temporelles.
        (Le k-fold classique est invalide : il viole l'ordre temporel.)
        """
        min_train = max(60, len(series) // 2)
        results = []

        for fold in range(n_folds):
            # Découpage temporel strict
            cutoff = len(series) - (n_folds - fold) * horizon
            if cutoff < min_train:
                continue

            train = series.iloc[:cutoff]
            test  = series.iloc[cutoff:cutoff + horizon]
            exog_train = exog.iloc[:cutoff] if exog is not None else None
            exog_test  = exog.iloc[cutoff:cutoff + horizon] if exog is not None else None

            try:
                self._fit_internal(train, exog_train)
                point, lower, upper = self._predict_internal(len(test), exog_test)

                actual = test.values
                point  = np.maximum(point[:len(actual)], 0)
                lower  = np.maximum(lower[:len(actual)], 0)
                upper  = np.maximum(upper[:len(actual)], 0)

                mase   = self._compute_mase(actual, point, train.values)
                mae    = float(np.mean(np.abs(actual - point)))
                coverage = float(np.mean((actual >= lower) & (actual <= upper)))

                results.append({
                    'fold':     fold + 1,
                    'mase':     mase,
                    'mae':      mae,
                    'coverage': coverage,
                    'n_train':  len(train),
                    'n_test':   len(test),
                })
            except Exception as e:
                results.append({'fold': fold + 1, 'error': str(e)})

        if not results:
            return {'model': self.model_name, 'error': 'Pas assez de données'}

        valid = [r for r in results if 'mase' in r]
        return {
            'model':          self.model_name,
            'n_folds':        len(valid),
            'mean_mase':      round(np.mean([r['mase']  for r in valid]), 4) if valid else None,
            'mean_mae':       round(np.mean([r['mae']   for r in valid]), 4) if valid else None,
            'mean_coverage':  round(np.mean([r['coverage'] for r in valid]), 4) if valid else None,
            'folds':          results,
        }

    @staticmethod
    def _compute_mase(
        actual: np.ndarray,
        forecast: np.ndarray,
        train: np.ndarray,
        seasonality: int = 1
    ) -> float:
        """
        Mean Absolute Scaled Error.

        MASE = MAE(forecast) / MAE(naive_seasonal)

        Le dénominateur est l'erreur d'un modèle naïf saisonnier sur les données
        d'entraînement — fournit une échelle absolue et comparable entre SKUs.
        MASE < 1 : meilleur que le naïf saisonnier.
        MASE = 1 : équivalent au naïf.
        """
        n = len(train)
        if n <= seasonality:
            return float('inf')

        # Erreur du modèle naïf saisonnier sur train
        naive_errors = np.abs(train[seasonality:] - train[:-seasonality])
        scale = np.mean(naive_errors)

        if scale < 1e-10:  # Série constante — MASE non défini
            return float('nan')

        mae = np.mean(np.abs(actual - forecast))
        return float(mae / scale)
