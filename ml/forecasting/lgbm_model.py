"""
ml/forecasting/lgbm_model.py
-----------------------------
LightGBM avec feature engineering temporel complet.

Stratégie de prévision : recursive multi-step forecasting.
Alternative rejetée — direct multi-output : nécessite N modèles séparés,
peu robuste sur des horizons courts avec peu de données.

Features construites :
- Lags : demande à t-1, t-2, t-3, t-7, t-14, t-28
- Rolling stats : mean/std sur 7, 14, 30 jours
- Encodage cyclique des périodes (sin/cos) — évite la discontinuité
  que les encodages linéaires introduisent entre décembre et janvier
- Features calendaires : jour_semaine, semaine_année, mois, est_debut_mois
- Variables exogènes si fournies
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from .base import BaseForecaster


def build_temporal_features(df: pd.DataFrame, target_col: str = 'demand') -> pd.DataFrame:
    """
    Construit les features temporelles à partir d'une série indexée par date.
    """
    feat = df.copy()
    t = feat[target_col]

    # ── Lags ─────────────────────────────────────────────────────────────────
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        feat[f'lag_{lag}'] = t.shift(lag)

    # ── Rolling statistics ────────────────────────────────────────────────────
    for window in [7, 14, 30]:
        feat[f'roll_mean_{window}'] = t.shift(1).rolling(window, min_periods=1).mean()
        feat[f'roll_std_{window}']  = t.shift(1).rolling(window, min_periods=1).std().fillna(0)
        feat[f'roll_max_{window}']  = t.shift(1).rolling(window, min_periods=1).max()

    # ── Encodage cyclique — sine/cosine ───────────────────────────────────────
    # Préserve la continuité temporelle (ex: vendredi→lundi, décembre→janvier)
    idx = feat.index
    feat['sin_dow']   = np.sin(2 * np.pi * idx.dayofweek / 7)
    feat['cos_dow']   = np.cos(2 * np.pi * idx.dayofweek / 7)
    feat['sin_week']  = np.sin(2 * np.pi * idx.isocalendar().week.astype(int) / 52)
    feat['cos_week']  = np.cos(2 * np.pi * idx.isocalendar().week.astype(int) / 52)
    feat['sin_month'] = np.sin(2 * np.pi * idx.month / 12)
    feat['cos_month'] = np.cos(2 * np.pi * idx.month / 12)
    feat['sin_dom']   = np.sin(2 * np.pi * idx.day / 31)
    feat['cos_dom']   = np.cos(2 * np.pi * idx.day / 31)

    # ── Features calendaires binaires ────────────────────────────────────────
    feat['is_monday']      = (idx.dayofweek == 0).astype(int)
    feat['is_friday']      = (idx.dayofweek == 4).astype(int)
    feat['is_weekend']     = (idx.dayofweek >= 5).astype(int)
    feat['is_month_start'] = idx.is_month_start.astype(int)
    feat['is_month_end']   = idx.is_month_end.astype(int)
    feat['quarter']        = idx.quarter

    # ── Tendance simple ───────────────────────────────────────────────────────
    feat['t_index'] = np.arange(len(feat))

    return feat


class LGBMForecaster(BaseForecaster):

    def __init__(
        self,
        confidence_level: float = 0.90,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 10,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,   # L1 — sparsité des features
        reg_lambda: float = 0.1,  # L2 — régularisation générale
    ):
        super().__init__(confidence_level)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self._model_point = None
        self._model_lower = None
        self._model_upper = None
        self._feature_cols: list[str] = []
        self._training_features: Optional[pd.DataFrame] = None

    @property
    def model_name(self) -> str:
        return "LightGBM"

    def _get_lgbm_params(self, objective: str = 'regression') -> dict:
        base = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        if objective == 'quantile_low':
            base['objective'] = 'quantile'
            base['alpha'] = (1 - self.confidence_level) / 2
        elif objective == 'quantile_high':
            base['objective'] = 'quantile'
            base['alpha'] = 1 - (1 - self.confidence_level) / 2
        else:
            base['objective'] = 'regression_l1'  # MAE loss — robuste aux outliers
        return base

    def _fit_internal(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm requis : pip install lightgbm")

        df = pd.DataFrame({'demand': series.values}, index=series.index)
        if exog is not None:
            for col in exog.columns:
                df[col] = exog[col].values

        feat_df = build_temporal_features(df, 'demand')
        feat_df = feat_df.dropna()

        exclude = {'demand'}
        self._feature_cols = [c for c in feat_df.columns if c not in exclude]
        self._training_features = feat_df

        X = feat_df[self._feature_cols].values
        y = feat_df['demand'].values

        # Trois modèles : point (median), borne inférieure, borne supérieure
        # Régression quantile pour les intervalles — plus honnête que les IC paramétriques
        self._model_point = lgb.LGBMRegressor(**self._get_lgbm_params('regression'))
        self._model_lower = lgb.LGBMRegressor(**self._get_lgbm_params('quantile_low'))
        self._model_upper = lgb.LGBMRegressor(**self._get_lgbm_params('quantile_high'))

        self._model_point.fit(X, y)
        self._model_lower.fit(X, y)
        self._model_upper.fit(X, y)

    def _predict_internal(
        self,
        horizon: int,
        future_exog: Optional[pd.DataFrame] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prévision récursive : chaque t+k utilise les prévisions de t+1 ... t+k-1.
        Le biais de propagation d'erreur est accepté pour horizon ≤ 30 jours.
        """
        # Construire le buffer de contexte depuis les données d'entraînement
        context = list(self._training_features['demand'].values[-60:])
        context_dates = list(self._training_features.index[-60:])

        points, lowers, uppers = [], [], []

        last_date = context_dates[-1]

        for step in range(horizon):
            next_date = last_date + pd.Timedelta(days=step + 1)

            # Reconstituer le DataFrame pour le feature engineering
            temp_df = pd.DataFrame(
                {'demand': context[-60:]},
                index=pd.date_range(end=next_date, periods=len(context[-60:]), freq='D')
            )
            if future_exog is not None and step < len(future_exog):
                temp_df.loc[next_date, list(future_exog.columns)] = future_exog.iloc[step].values

            feat = build_temporal_features(temp_df, 'demand')
            last_row = feat.iloc[[-1]][self._feature_cols]

            p = float(self._model_point.predict(last_row)[0])
            l = float(self._model_lower.predict(last_row)[0])
            u = float(self._model_upper.predict(last_row)[0])

            # Cohérence : lower ≤ point ≤ upper
            l = min(l, p)
            u = max(u, p)

            points.append(p)
            lowers.append(l)
            uppers.append(u)

            context.append(p)  # Utiliser la prévision comme contexte futur

        return np.array(points), np.array(lowers), np.array(uppers)

    def feature_importance(self) -> Optional[pd.DataFrame]:
        """Retourne l'importance des features — utile pour l'explicabilité."""
        if self._model_point is None:
            return None
        imp = pd.DataFrame({
            'feature':    self._feature_cols,
            'importance': self._model_point.feature_importances_,
        }).sort_values('importance', ascending=False)
        return imp
