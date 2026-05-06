"""
ml/analysis/threshold_optimizer.py
------------------------------------
Calcul automatique des seuils d'alerte stock basé sur les données ML.

Remplace le seuil statique saisi manuellement par un seuil calculé
à partir de la demande historique, du délai fournisseur et du taux
de service cible. Met à jour automatiquement Product.alert_threshold
après chaque optimisation réussie.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ThresholdRecommendation:
    sku_id:              str
    current_threshold:   int
    recommended_threshold: int
    safety_stock:        float
    reorder_point:       float
    service_level_target: float
    method:              str    # 'simulation' | 'analytical'
    confidence:          str    # 'high' | 'medium' | 'low'
    delta:               int    # recommended - current
    rationale:           str


class ThresholdOptimizer:
    """
    Calcule les seuils d'alerte optimaux à partir des résultats ML.
    Peut mettre à jour directement les modèles Django Product.
    """

    def from_policy(
        self,
        sku_id: str,
        reorder_point: float,
        safety_stock: float,
        service_level: float,
        current_threshold: int,
        method: str = 'simulation',
    ) -> ThresholdRecommendation:
        """
        Dérive le seuil d'alerte depuis les résultats d'optimisation.

        Le seuil d'alerte = point de commande arrondi à l'entier supérieur.
        C'est le niveau de stock qui déclenche une alerte visuelle dans l'interface,
        distinct du point de commande (qui déclenche une commande effective).
        """
        recommended = math.ceil(reorder_point)
        delta = recommended - current_threshold

        if service_level >= 0.98:
            confidence = 'high'
        elif service_level >= 0.92:
            confidence = 'medium'
        else:
            confidence = 'low'

        rationale = (
            f"Point de commande optimal : {reorder_point:.1f} unités "
            f"(taux de service {service_level * 100:.1f}%). "
            f"Stock de sécurité inclus : {safety_stock:.1f} unités. "
            f"Méthode : {method}."
        )

        return ThresholdRecommendation(
            sku_id=sku_id,
            current_threshold=current_threshold,
            recommended_threshold=recommended,
            safety_stock=round(safety_stock, 1),
            reorder_point=round(reorder_point, 1),
            service_level_target=service_level,
            method=method,
            confidence=confidence,
            delta=delta,
            rationale=rationale,
        )

    def from_statistics(
        self,
        sku_id: str,
        demand_series: np.ndarray,
        lead_time_mean: float,
        lead_time_std: float,
        service_level: float,
        current_threshold: int,
    ) -> ThresholdRecommendation:
        """
        Calcul analytique (formule classique sigma) quand la simulation n'est pas disponible.
        Utilisé comme fallback si l'optimisation Monte Carlo n'a pas encore tourné.
        """
        mean_d  = float(np.mean(demand_series))
        std_d   = float(np.std(demand_series, ddof=1))

        # Facteur z pour le taux de service (approximation normale)
        z_map = {0.90: 1.282, 0.95: 1.645, 0.97: 1.881, 0.98: 2.054, 0.99: 2.326}
        z = min(z_map.items(), key=lambda x: abs(x[0] - service_level))[1]

        # Stock de sécurité = z × σ_demande_sur_délai
        sigma_lt = math.sqrt(
            lead_time_mean * std_d**2 + mean_d**2 * lead_time_std**2
        )
        safety_stock  = z * sigma_lt
        reorder_point = mean_d * lead_time_mean + safety_stock
        recommended   = math.ceil(reorder_point)

        return ThresholdRecommendation(
            sku_id=sku_id,
            current_threshold=current_threshold,
            recommended_threshold=recommended,
            safety_stock=round(safety_stock, 1),
            reorder_point=round(reorder_point, 1),
            service_level_target=service_level,
            method='analytical',
            confidence='medium',
            delta=recommended - current_threshold,
            rationale=(
                f"Formule analytique (z={z:.3f}, μ_délai={lead_time_mean:.1f}j, "
                f"σ_délai={lead_time_std:.1f}j). "
                f"Recommander la simulation Monte Carlo pour plus de précision."
            ),
        )

    def apply_to_product(
        self,
        recommendation: ThresholdRecommendation,
        min_change_threshold: int = 2,
    ) -> bool:
        """
        Met à jour Product.alert_threshold en base si le delta est significatif.

        Args:
            min_change_threshold: Ne met à jour que si |delta| >= cette valeur.
                                  Évite les mises à jour pour des variations de 1 unité.
        Returns:
            True si la mise à jour a été effectuée.
        """
        if abs(recommendation.delta) < min_change_threshold:
            return False

        try:
            from inventory.models import Product
            updated = Product.objects.filter(
                ml_sku_id=recommendation.sku_id
            ).update(alert_threshold=recommendation.recommended_threshold)
            return updated > 0
        except Exception:
            return False
