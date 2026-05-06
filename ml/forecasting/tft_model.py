"""
ml/forecasting/tft_model.py
----------------------------
Temporal Fusion Transformer via PyTorch Forecasting.

Architecture TFT (Lim et al., 2021) — apports clés vs LGBM/SARIMA :
1. Variable Selection Networks : sélection automatique des features pertinentes
   par attention — fournit une explicabilité native.
2. Gated Residual Networks : skip connections adaptatives qui permettent
   au modèle d'ignorer les features non-pertinentes.
3. Multi-head attention sur les dépendances temporelles longues.
4. Sorties quantiles simultanées (pas 3 modèles séparés comme LGBM).

Compromis accepté : temps d'entraînement 10-50x supérieur à LGBM.
Justifié ici car le TFT est entraîné une fois et mis en cache (Celery).
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseForecaster


class TFTForecaster(BaseForecaster):

    def __init__(
        self,
        confidence_level: float = 0.90,
        max_encoder_length: int = 60,    # Fenêtre de contexte passée
        max_prediction_length: int = 30, # Horizon maximum de prévision
        hidden_size: int = 32,           # Réduit volontairement — pas de données massives
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 16,
        learning_rate: float = 1e-3,
        max_epochs: int = 50,
        batch_size: int = 64,
        gradient_clip_val: float = 0.1,
        accelerator: str = 'cpu',        # 'gpu' si disponible
    ):
        super().__init__(confidence_level)
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.gradient_clip_val = gradient_clip_val
        self.accelerator = accelerator
        self._tft = None
        self._trainer = None
        self._training_dataset = None

    @property
    def model_name(self) -> str:
        return "TFT"

    def _prepare_dataframe(
        self,
        series: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        group_id: str = "sku_0"
    ) -> pd.DataFrame:
        """
        PyTorch Forecasting attend un format long avec colonnes spécifiques.
        """
        df = pd.DataFrame({
            'time_idx':  np.arange(len(series)),
            'group_id':  group_id,
            'demand':    series.values.astype(float),
            'date':      series.index,
        })

        # Features temporelles
        df['month']       = pd.DatetimeIndex(df['date']).month.astype(str)
        df['dayofweek']   = pd.DatetimeIndex(df['date']).dayofweek.astype(str)
        df['weekofyear']  = pd.DatetimeIndex(df['date']).isocalendar().week.astype(int)

        # Features continues normalisées
        df['t_norm']      = df['time_idx'] / len(df)
        df['sin_dow']     = np.sin(2 * np.pi * pd.DatetimeIndex(df['date']).dayofweek / 7)
        df['cos_dow']     = np.cos(2 * np.pi * pd.DatetimeIndex(df['date']).dayofweek / 7)
        df['sin_month']   = np.sin(2 * np.pi * pd.DatetimeIndex(df['date']).month / 12)
        df['cos_month']   = np.cos(2 * np.pi * pd.DatetimeIndex(df['date']).month / 12)

        if exog is not None:
            for col in exog.columns:
                df[col] = exog[col].values

        return df

    def _fit_internal(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        try:
            import pytorch_forecasting as pf
            import pytorch_lightning as pl
            import torch
        except ImportError:
            raise ImportError(
                "pytorch-forecasting et pytorch-lightning requis :\n"
                "pip install pytorch-forecasting pytorch-lightning"
            )

        warnings.filterwarnings('ignore')

        df = self._prepare_dataframe(series, exog)

        # Séparation train/validation temporelle stricte
        val_cutoff = len(df) - self.max_prediction_length

        time_varying_known_reals = ['t_norm', 'sin_dow', 'cos_dow', 'sin_month', 'cos_month']
        if exog is not None:
            time_varying_known_reals += list(exog.columns)

        training_dataset = pf.TimeSeriesDataSet(
            df[df['time_idx'] <= val_cutoff],
            time_idx='time_idx',
            target='demand',
            group_ids=['group_id'],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=['group_id'],
            time_varying_known_categoricals=['month', 'dayofweek'],
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=['demand'],
            target_normalizer=pf.data.encoders.TorchNormalizer(
                method='robust',    # Robuste aux outliers — médiane/IQR
                center=True,
            ),
            allow_missing_timesteps=True,
        )

        validation_dataset = pf.TimeSeriesDataSet.from_dataset(
            training_dataset,
            df,
            predict=True,
            stop_randomization=True,
        )

        train_loader = training_dataset.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=0,  # 0 pour compatibilité Windows/Mac
        )
        val_loader = validation_dataset.to_dataloader(
            train=False,
            batch_size=self.batch_size * 2,
            num_workers=0,
        )

        # Quantiles pour les intervalles de confiance
        alpha = (1 - self.confidence_level) / 2
        quantiles = [alpha, 0.5, 1 - alpha]

        self._tft = pf.TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            output_size=len(quantiles),
            loss=pf.metrics.QuantileLoss(quantiles=quantiles),
            reduce_on_plateau_patience=5,
            log_interval=-1,
        )

        self._trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            enable_model_summary=False,
            enable_progress_bar=False,
            gradient_clip_val=self.gradient_clip_val,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    mode='min',
                    verbose=False,
                )
            ],
        )

        self._trainer.fit(self._tft, train_loader, val_loader)
        self._training_dataset = training_dataset
        self._full_df = df

    def _predict_internal(
        self,
        horizon: int,
        future_exog: Optional[pd.DataFrame] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        horizon = min(horizon, self.max_prediction_length)

        try:
            raw_predictions = self._tft.predict(
                self._training_dataset,
                mode='raw',
                return_x=False,
            )
            # raw_predictions.output shape : (n_samples, horizon, n_quantiles)
            preds = raw_predictions.output
            if hasattr(preds, 'numpy'):
                preds = preds.numpy()

            # Prendre la moyenne des échantillons si plusieurs
            if preds.ndim == 3:
                preds = preds.mean(axis=0)  # (horizon, n_quantiles)

            preds = preds[:horizon]
            lower = preds[:, 0]
            point = preds[:, 1]
            upper = preds[:, 2]

        except Exception:
            # Fallback : prédictions brutes
            predictions = self._tft.predict(self._training_dataset)
            if hasattr(predictions, 'numpy'):
                predictions = predictions.numpy()
            point = predictions[0, :horizon] if predictions.ndim > 1 else predictions[:horizon]
            spread = np.std(point) * 1.5
            lower = point - spread
            upper = point + spread

        return np.array(point), np.array(lower), np.array(upper)

    def get_interpretation(self) -> Optional[dict]:
        """
        Retourne l'interprétation du TFT :
        - Attention weights (quels pas de temps passés sont importants)
        - Variable importances (quelles features pilotent la prévision)
        Spécifique au TFT — non disponible dans les autres modèles.
        """
        if self._tft is None or self._training_dataset is None:
            return None
        try:
            interpretation = self._tft.interpret_output(
                self._tft.predict(self._training_dataset, mode='raw', return_x=True),
                reduction='sum',
            )
            return {
                'attention':           interpretation.get('attention', None),
                'static_variables':    interpretation.get('static_variables', None),
                'encoder_variables':   interpretation.get('encoder_variables', None),
                'decoder_variables':   interpretation.get('decoder_variables', None),
            }
        except Exception:
            return None
