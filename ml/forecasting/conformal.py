"""
ml/forecasting/conformal.py
----------------------------
Conformal Prediction pour la calibration des intervalles de prévision.

Pourquoi conformal prediction plutôt que les IC paramétriques ?
1. Distribution-free : aucune hypothèse sur la loi des erreurs.
   Les résidus industriels sont rarement gaussiens.
2. Coverage garanti : sur données exchangeables, P(Y ∈ Ĉ) ≥ 1-α exactement.
3. Applicable à n'importe quel modèle de base : SARIMAX, Prophet, LGBM, TFT.

Méthode : Conformal Quantile Regression (Romano et al., 2019)
adapté au setting temporel (EnbPI — Xu & Xie, 2021).

Limitation honnête : l'échangeabilité est violée pour les séries temporelles.
L'EnbPI (Ensemble Batch Prediction Intervals) atténue ce problème en utilisant
des résidus sur une fenêtre glissante récente.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


class ConformalCalibrator:
    """
    Calibre les intervalles de prédiction par conformal prediction.

    Usage :
        calibrator = ConformalCalibrator(confidence_level=0.90)
        calibrator.fit(residuals_calibration)  # résidus sur ensemble de calibration
        lower_cal, upper_cal = calibrator.calibrate(point_forecast, lower_raw, upper_raw)
    """

    def __init__(
        self,
        confidence_level: float = 0.90,
        method: str = 'enbpi',          # 'simple' ou 'enbpi'
        window_size: int = 30,           # Taille fenêtre pour EnbPI
    ):
        self.confidence_level = confidence_level
        self.method = method
        self.window_size = window_size
        self._q_hat: Optional[float] = None
        self._calibration_residuals: Optional[np.ndarray] = None

    def fit(self, residuals: np.ndarray) -> 'ConformalCalibrator':
        """
        Calcule le quantile de calibration depuis les résidus de l'ensemble
        de calibration (données que le modèle n'a pas vues à l'entraînement).

        Args:
            residuals: array(n,) — erreurs absolues sur l'ensemble de calibration.
        """
        self._calibration_residuals = np.abs(residuals)

        if self.method == 'enbpi':
            # Utiliser uniquement les résidus récents — plus représentatifs
            recent = self._calibration_residuals[-self.window_size:]
            n = len(recent)
        else:
            recent = self._calibration_residuals
            n = len(recent)

        # Quantile conformal : (1-α)(1 + 1/n)-quantile
        # Le facteur (1 + 1/n) assure la couverture marginale théorique
        alpha = 1 - self.confidence_level
        level = np.ceil((1 - alpha) * (1 + 1/n)) / (1 + 1/n) * (1 - alpha)
        level = min(1 - alpha * n / (n + 1), 1.0)
        effective_quantile = min((1 - alpha) * (1 + 1/n), 1.0)

        self._q_hat = float(np.quantile(recent, effective_quantile))
        return self

    def calibrate(
        self,
        point_forecast: np.ndarray,
        lower_raw: np.ndarray,
        upper_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Ajuste les bornes des intervalles par le quantile conformal.

        Les bornes calibrées sont garanties d'atteindre la couverture
        cible sur des données issues de la même distribution.
        """
        if self._q_hat is None:
            raise RuntimeError("ConformalCalibrator.fit() doit être appelé avant calibrate()")

        lower_cal = np.maximum(point_forecast - self._q_hat, 0.0)
        upper_cal = point_forecast + self._q_hat

        # Élargir si les bornes brutes sont déjà plus larges (prendre le max)
        lower_final = np.minimum(lower_cal, lower_raw)
        upper_final = np.maximum(upper_cal, upper_raw)

        return lower_final, upper_final

    def coverage_rate(
        self,
        actual: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> float:
        """
        Calcule le taux de couverture empirique — test de calibration.
        Doit être proche de confidence_level. Écart > 5pp = problème.
        """
        return float(np.mean((actual >= lower) & (actual <= upper)))

    def interval_width(self, lower: np.ndarray, upper: np.ndarray) -> float:
        """Largeur moyenne des intervalles — mesure d'efficacité."""
        return float(np.mean(upper - lower))
