"""
ml/optimization/pareto.py
--------------------------
Construction de la frontière de Pareto coût total vs risque de rupture.

Objectifs antagonistes :
  f1 = E[coût total annuel]         → minimiser
  f2 = P(rupture sur l'horizon)     → minimiser

Un point est Pareto-optimal si aucun autre point ne le domine sur les deux objectifs.
La frontière de Pareto représente l'ensemble des politiques non-dominées —
le décideur choisit son point selon son appétit au risque.

Lien Lean Six Sigma :
  - Point sigma-3 : P(rupture) ≤ 0.27% — objectif qualité 3σ
  - Point sigma-6 : P(rupture) ≤ 0.00034% — objectif Six Sigma
  - Le coût associé à chaque point quantifie le prix de la qualité (Cost of Quality).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .monte_carlo import SimulationResult
from ..data.schemas import ParetoPoint, StockPolicy, SKUConfig


def is_pareto_dominated(costs: np.ndarray, risks: np.ndarray) -> np.ndarray:
    """
    Identifie les points dominés dans l'espace (coût, risque).
    Un point i est dominé si ∃j : coût[j] ≤ coût[i] ET risque[j] ≤ risque[i]
    avec au moins une inégalité stricte.

    Returns:
        mask booléen — True = point Pareto-optimal (non dominé).
    """
    n = len(costs)
    is_optimal = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (costs[j] <= costs[i] and risks[j] <= risks[i] and
                    (costs[j] < costs[i] or risks[j] < risks[i])):
                is_optimal[i] = False
                break

    return is_optimal


class ParetoOptimizer:
    """
    Construit et analyse la frontière de Pareto pour un SKU.
    """

    def build_frontier(
        self,
        simulation_results: list[SimulationResult],
    ) -> list[ParetoPoint]:
        """
        Extrait la frontière de Pareto depuis les résultats de simulation.
        """
        if not simulation_results:
            return []

        costs = np.array([r.mean_cost          for r in simulation_results])
        risks = np.array([r.stockout_probability for r in simulation_results])

        # Normalisation pour éviter les biais d'échelle
        costs_norm = (costs - costs.min()) / (costs.max() - costs.min() + 1e-10)
        risks_norm = (risks - risks.min()) / (risks.max() - risks.min() + 1e-10)

        optimal_mask = is_pareto_dominated(costs_norm, risks_norm)

        pareto_points = []
        for i, (is_opt, r) in enumerate(zip(optimal_mask, simulation_results)):
            if is_opt:
                pareto_points.append(ParetoPoint(
                    reorder_point=r.reorder_point,
                    order_quantity=r.order_quantity,
                    total_cost=r.mean_cost,
                    stockout_prob=r.stockout_probability,
                    service_level=r.mean_service_level,
                    safety_stock=r.safety_stock_implied,
                ))

        # Trier par risque croissant
        pareto_points.sort(key=lambda p: p.stockout_prob)
        return pareto_points

    def to_dataframe(self, frontier: list[ParetoPoint]) -> pd.DataFrame:
        if not frontier:
            return pd.DataFrame()
        rows = [{
            'reorder_point':   round(p.reorder_point, 1),
            'order_quantity':  round(p.order_quantity, 1),
            'total_cost':      round(p.total_cost, 2),
            'stockout_prob':   round(p.stockout_prob, 4),
            'service_level':   round(p.service_level, 4),
            'safety_stock':    round(p.safety_stock, 1),
        } for p in frontier]
        return pd.DataFrame(rows)

    def select_policy(
        self,
        frontier: list[ParetoPoint],
        target_service_level: float = 0.95,
    ) -> ParetoPoint:
        """
        Sélectionne le point Pareto le plus proche du taux de service cible.
        Si plusieurs points satisfont la cible, retourne le moins coûteux.
        """
        if not frontier:
            raise ValueError("Frontière Pareto vide")

        candidates = [p for p in frontier if p.service_level >= target_service_level]

        if not candidates:
            # Aucun point n'atteint la cible — retourner le moins risqué
            return min(frontier, key=lambda p: p.stockout_prob)

        return min(candidates, key=lambda p: p.total_cost)

    def lean_analysis(
        self,
        optimal_policy: ParetoPoint,
        naive_policy: ParetoPoint,
        sku_config: SKUConfig,
    ) -> dict:
        """
        Analyse Lean de la réduction par rapport à la politique naïve (Wilson statique).

        Calcule :
        - Réduction du capital immobilisé (stock de sécurité × coût unitaire)
        - Réduction des jours de rupture attendus
        - Équivalent en Cost of Quality (CoQ)
        """
        capital_naive   = naive_policy.safety_stock * sku_config.unit_cost
        capital_optimal = optimal_policy.safety_stock * sku_config.unit_cost
        capital_saving  = capital_naive - capital_optimal
        capital_saving_pct = (capital_saving / capital_naive * 100) if capital_naive > 0 else 0

        cost_saving     = naive_policy.total_cost - optimal_policy.total_cost
        cost_saving_pct = (cost_saving / naive_policy.total_cost * 100) if naive_policy.total_cost > 0 else 0

        service_improvement = optimal_policy.service_level - naive_policy.service_level

        return {
            'capital_immobilise_naive':     round(capital_naive, 2),
            'capital_immobilise_optimal':   round(capital_optimal, 2),
            'reduction_capital':            round(capital_saving, 2),
            'reduction_capital_pct':        round(capital_saving_pct, 1),
            'cout_total_naive':             round(naive_policy.total_cost, 2),
            'cout_total_optimal':           round(optimal_policy.total_cost, 2),
            'economie_annuelle':            round(cost_saving, 2),
            'economie_pct':                 round(cost_saving_pct, 1),
            'amelioration_taux_service_pp': round(service_improvement * 100, 2),
            'muda_type':                    'Surstock (muda de surproduction)',
            'lean_principle':               'Flux tiré — commande sur signal réel',
        }
