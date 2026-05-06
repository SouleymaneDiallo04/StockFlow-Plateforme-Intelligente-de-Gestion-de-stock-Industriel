"""
ml/forecasting/evaluator.py
----------------------------
Évaluation rigoureuse et comparaison multi-modèles.

Métriques implémentées :
- MASE  : Mean Absolute Scaled Error — métrique principale, scale-free
- MAE   : Mean Absolute Error — interprétable en unités métier
- RMSE  : pour compatibilité, mais non utilisé comme critère principal
- MAPE  : évité sur séries avec zéros (division par zéro) — remplacé par sMAPE
- sMAPE : Symmetric MAPE — robuste aux zéros
- Coverage Rate : validation de calibration des intervalles
- Interval Width : efficacité des intervalles (plus étroit = mieux, à coverage égale)
- Winkler Score  : combine coverage ET width en une métrique — standard évaluation probabiliste
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from ..data.schemas import ForecastResult, ModelComparison


@dataclass
class EvaluationMetrics:
    model_name:     str
    sku_id:         str
    n_obs:          int
    mase:           float
    mae:            float
    rmse:           float
    smape:          float
    coverage_rate:  float    # % valeurs réelles dans l'IC
    interval_width: float    # Largeur moyenne IC
    winkler_score:  float    # Score de Winkler — pénalise IC trop larges
    training_time_s: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            'model':          self.model_name,
            'sku_id':         self.sku_id,
            'n_obs':          self.n_obs,
            'MASE':           round(self.mase, 4),
            'MAE':            round(self.mae, 4),
            'RMSE':           round(self.rmse, 4),
            'sMAPE (%)':      round(self.smape * 100, 2),
            'Coverage (%)':   round(self.coverage_rate * 100, 2),
            'IC Width':       round(self.interval_width, 2),
            'Winkler':        round(self.winkler_score, 4),
            'Train time (s)': round(self.training_time_s, 2) if self.training_time_s else None,
        }


class ForecastEvaluator:
    """
    Évalue et compare les résultats de prévision de plusieurs modèles.
    """

    def evaluate(
        self,
        result: ForecastResult,
        actual: np.ndarray,
        train_series: np.ndarray,
        seasonality: int = 7,
    ) -> EvaluationMetrics:
        """
        Calcule toutes les métriques pour un résultat de prévision.

        Args:
            result:       ForecastResult du modèle.
            actual:       Valeurs réelles de la période de test.
            train_series: Série d'entraînement — nécessaire pour normaliser MASE.
            seasonality:  Période saisonnière pour le modèle naïf du MASE.
        """
        h = min(len(actual), len(result.point_forecast))
        actual   = actual[:h].astype(float)
        forecast = result.point_forecast[:h].astype(float)
        lower    = result.lower_bound[:h].astype(float)
        upper    = result.upper_bound[:h].astype(float)

        mase  = self._mase(actual, forecast, train_series, seasonality)
        mae   = float(np.mean(np.abs(actual - forecast)))
        rmse  = float(np.sqrt(np.mean((actual - forecast)**2)))
        smape = self._smape(actual, forecast)
        cov   = float(np.mean((actual >= lower) & (actual <= upper)))
        width = float(np.mean(upper - lower))
        wink  = self._winkler(actual, lower, upper, result.confidence_level)

        return EvaluationMetrics(
            model_name=result.model_name,
            sku_id=result.sku_id,
            n_obs=h,
            mase=mase,
            mae=mae,
            rmse=rmse,
            smape=smape,
            coverage_rate=cov,
            interval_width=width,
            winkler_score=wink,
            training_time_s=result.training_time_s,
        )

    def compare_models(
        self,
        results: list[ForecastResult],
        actual: np.ndarray,
        train_series: np.ndarray,
        sku_id: str,
    ) -> ModelComparison:
        """
        Compare plusieurs modèles sur le même SKU.
        Retourne un ModelComparison trié par MASE.
        """
        evaluated = []
        for r in results:
            try:
                metrics = self.evaluate(r, actual, train_series)
                # Injecter les métriques dans le ForecastResult
                r.mase         = metrics.mase
                r.mae          = metrics.mae
                r.coverage_rate = metrics.coverage_rate
                evaluated.append(r)
            except Exception as e:
                print(f"  Évaluation échouée pour {r.model_name} : {e}")

        comparison = ModelComparison(sku_id=sku_id, results=evaluated)
        ranked = comparison.ranked_by_mase()
        if ranked:
            comparison.best_model_by_mase = ranked[0].model_name

        return comparison

    def comparison_dataframe(
        self,
        results: list[ForecastResult],
        actual: np.ndarray,
        train_series: np.ndarray,
    ) -> pd.DataFrame:
        """DataFrame de comparaison prêt à l'affichage."""
        rows = []
        for r in results:
            try:
                m = self.evaluate(r, actual, train_series)
                rows.append(m.to_dict())
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values('MASE').reset_index(drop=True)
        df.index += 1  # Rang 1-based
        return df

    # ── Métriques ─────────────────────────────────────────────────────────────

    @staticmethod
    def _mase(
        actual: np.ndarray,
        forecast: np.ndarray,
        train: np.ndarray,
        seasonality: int = 7,
    ) -> float:
        n = len(train)
        if n <= seasonality:
            return float('inf')
        naive_errors = np.abs(train[seasonality:] - train[:-seasonality])
        scale = np.mean(naive_errors)
        if scale < 1e-10:
            return float('nan')
        return float(np.mean(np.abs(actual - forecast)) / scale)

    @staticmethod
    def _smape(actual: np.ndarray, forecast: np.ndarray) -> float:
        """
        Symmetric MAPE — robuste aux valeurs proches de zéro.
        Formule : mean(2|A-F| / (|A|+|F|+ε))
        """
        eps = 1e-8
        return float(np.mean(
            2 * np.abs(actual - forecast) / (np.abs(actual) + np.abs(forecast) + eps)
        ))

    @staticmethod
    def _winkler(
        actual: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        alpha_level: float,
    ) -> float:
        """
        Score de Winkler (1972) — évalue simultanément la couverture et la largeur.

        W = (u - l) + (2/α)×max(l-y, 0) + (2/α)×max(y-u, 0)

        Un IC étroit bien calibré → score bas.
        Un IC large ou mal calibré → score élevé.
        Standard dans les compétitions de prévision probabiliste (M5, GEFCom).
        """
        alpha = 1 - alpha_level
        width = upper - lower
        penalty_low  = (2 / alpha) * np.maximum(lower - actual, 0)
        penalty_high = (2 / alpha) * np.maximum(actual - upper, 0)
        return float(np.mean(width + penalty_low + penalty_high))

    def portfolio_summary(
        self,
        all_comparisons: dict[str, ModelComparison],
    ) -> pd.DataFrame:
        """
        Résumé agrégé sur l'ensemble du portefeuille de SKUs.
        Utile pour choisir un modèle par défaut sur l'ensemble du catalogue.
        """
        rows = []
        for sku_id, comp in all_comparisons.items():
            for r in comp.results:
                if r.mase is not None:
                    rows.append({
                        'sku_id':     sku_id,
                        'model':      r.model_name,
                        'mase':       r.mase,
                        'mae':        r.mae,
                        'coverage':   r.coverage_rate,
                        'best':       r.model_name == comp.best_model_by_mase,
                    })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Victoires par modèle — combien de fois meilleur MASE ?
        wins = df[df['best']].groupby('model').size().rename('wins')

        # MASE médian par modèle — robuste aux outliers
        agg = df.groupby('model').agg(
            median_mase=('mase', 'median'),
            mean_coverage=('coverage', 'mean'),
        ).round(4)

        summary = agg.join(wins, how='left').fillna(0)
        summary['wins'] = summary['wins'].astype(int)
        return summary.sort_values('median_mase')
