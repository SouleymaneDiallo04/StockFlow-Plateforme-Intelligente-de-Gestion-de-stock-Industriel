"""
ml/analysis/mlflow_tracker.py
-------------------------------
Intégration MLflow pour le tracking des expériences de prévision.

Enregistre pour chaque run :
- Paramètres du modèle (hyperparamètres)
- Métriques d'évaluation (MASE, MAE, Coverage, Winkler)
- Artefacts (graphiques, importance des features)
- Tags (SKU, profil, date)

Dégradation gracieuse : si MLflow n'est pas installé ou le serveur
non démarré, le tracking est désactivé sans lever d'exception.
"""
from __future__ import annotations

import os
from typing import Optional, Any
from contextlib import contextmanager


def _mlflow_available() -> bool:
    try:
        import mlflow  # noqa
        return True
    except ImportError:
        return False


class MLflowTracker:
    """
    Wrapper MLflow avec dégradation gracieuse.
    Si MLflow n'est pas disponible, toutes les méthodes sont no-ops.
    """

    EXPERIMENT_NAME = "StockFlow Intelligence"

    def __init__(self, tracking_uri: Optional[str] = None):
        self.enabled = _mlflow_available()
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'mlruns')

        if self.enabled:
            try:
                import mlflow
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.EXPERIMENT_NAME)
            except Exception:
                self.enabled = False

    @contextmanager
    def run(self, run_name: str, tags: Optional[dict] = None):
        """Context manager pour un run MLflow."""
        if not self.enabled:
            yield None
            return

        import mlflow
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            yield run

    def log_forecast_run(
        self,
        model_name: str,
        sku_id: str,
        sku_profile: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        training_time_s: Optional[float] = None,
    ) -> Optional[str]:
        """
        Enregistre un run de prévision complet.
        Retourne le run_id MLflow ou None si désactivé.
        """
        if not self.enabled:
            return None

        import mlflow

        tags = {
            'model':   model_name,
            'sku_id':  sku_id,
            'profile': sku_profile,
            'project': 'StockFlow Intelligence',
        }

        with mlflow.start_run(run_name=f"{model_name}_{sku_id}", tags=tags) as run:
            # Paramètres
            mlflow.log_params({k: str(v) for k, v in params.items()})

            # Métriques
            clean_metrics = {k: float(v) for k, v in metrics.items()
                             if v is not None and not (isinstance(v, float) and (v != v))}
            if clean_metrics:
                mlflow.log_metrics(clean_metrics)

            if training_time_s is not None:
                mlflow.log_metric('training_time_s', training_time_s)

            return run.info.run_id

    def log_optimization_run(
        self,
        sku_id: str,
        policy_type: str,
        reorder_point: float,
        order_quantity: float,
        service_level: float,
        annual_cost: float,
        capital_reduction_pct: Optional[float],
        n_simulations: int,
    ) -> Optional[str]:
        """Enregistre un run d'optimisation Monte Carlo."""
        if not self.enabled:
            return None

        import mlflow

        tags = {
            'sku_id':       sku_id,
            'analysis_type': 'optimization',
            'policy':       policy_type,
        }

        with mlflow.start_run(run_name=f"optimize_{sku_id}", tags=tags) as run:
            mlflow.log_params({
                'policy_type':   policy_type,
                'n_simulations': n_simulations,
                'sku_id':        sku_id,
            })
            mlflow.log_metrics({
                'reorder_point':  reorder_point,
                'order_quantity': order_quantity,
                'service_level':  service_level,
                'annual_cost':    annual_cost,
            })
            if capital_reduction_pct is not None:
                mlflow.log_metric('capital_reduction_pct', capital_reduction_pct)

            return run.info.run_id

    def get_best_model(self, sku_id: str, metric: str = 'mase') -> Optional[dict]:
        """
        Récupère le meilleur modèle pour un SKU depuis l'historique MLflow.
        Utile pour charger automatiquement le meilleur modèle en production.
        """
        if not self.enabled:
            return None

        try:
            import mlflow
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(self.EXPERIMENT_NAME)
            if not experiment:
                return None

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.sku_id = '{sku_id}'",
                order_by=[f"metrics.{metric} ASC"],
                max_results=1,
            )

            if not runs:
                return None

            best = runs[0]
            return {
                'run_id':    best.info.run_id,
                'model':     best.data.tags.get('model', 'unknown'),
                'mase':      best.data.metrics.get('mase'),
                'coverage':  best.data.metrics.get('coverage_rate'),
                'winkler':   best.data.metrics.get('winkler_score'),
            }
        except Exception:
            return None

    @property
    def ui_url(self) -> Optional[str]:
        """URL de l'interface MLflow."""
        if not self.enabled:
            return None
        return f"http://localhost:5000" if 'localhost' in self.tracking_uri else self.tracking_uri


# Instance globale
tracker = MLflowTracker()
