"""
ml/optimization/policy.py
--------------------------
Orchestrateur complet de l'optimisation d'une politique de stock.

Enchaîne : scan de grille → simulation fine → frontière Pareto → sélection.
"""
from __future__ import annotations

import numpy as np

from .monte_carlo import MonteCarloSimulator, SimulationConfig
from .pareto import ParetoOptimizer
from ..data.schemas import SKUConfig, StockPolicy, ParetoPoint


class StockPolicyOptimizer:
    """
    Orchestre l'optimisation complète pour un SKU.
    """

    def __init__(
        self,
        n_simulations: int = 10_000,
        horizon_days:  int = 365,
        grid_size:     int = 15,    # 15×15 = 225 points de grille
        n_sim_grid:    int = 2_000,
    ):
        self.n_simulations = n_simulations
        self.horizon_days  = horizon_days
        self.grid_size     = grid_size
        self.n_sim_grid    = n_sim_grid

    def optimize(
        self,
        sku_config:             SKUConfig,
        target_service_level:   float = 0.95,
        policy_type:            str   = "(r,Q)",
    ) -> tuple[StockPolicy, list[ParetoPoint], dict]:
        """
        Optimise la politique de stock pour un SKU.

        Returns:
            (optimal_policy, pareto_frontier, lean_analysis)
        """
        simulator = MonteCarloSimulator(SimulationConfig(
            n_simulations=self.n_sim_grid,
            horizon_days=self.horizon_days,
        ))
        optimizer = ParetoOptimizer()

        # ── Définir la grille de recherche ────────────────────────────────────
        mean_lt    = np.exp(sku_config.lead_time_mu + sku_config.lead_time_sigma**2 / 2)
        mean_demand_lt = sku_config.base_demand * mean_lt

        # Points de commande : de 0 à 3× la demande sur délai moyen
        r_min = 0
        r_max = mean_demand_lt * 3
        reorder_points = np.linspace(r_min, r_max, self.grid_size)

        # Quantités commandées : de 0.5 à 4× la demande mensuelle
        q_min = max(sku_config.base_demand * 7, 1)
        q_max = sku_config.base_demand * 120
        order_quantities = np.linspace(q_min, q_max, self.grid_size)

        # ── Scan de grille ────────────────────────────────────────────────────
        grid_results = simulator.simulate_grid(
            sku_config, reorder_points, order_quantities,
            policy_type=policy_type,
            n_sim_grid=self.n_sim_grid,
        )

        # ── Frontière Pareto ──────────────────────────────────────────────────
        frontier = optimizer.build_frontier(grid_results)

        if not frontier:
            # Fallback Wilson si la simulation échoue
            return self._wilson_fallback(sku_config, target_service_level), [], {}

        # ── Sélection du point optimal ────────────────────────────────────────
        best_pareto = optimizer.select_policy(frontier, target_service_level)

        # ── Simulation fine sur le point optimal ──────────────────────────────
        fine_simulator = MonteCarloSimulator(SimulationConfig(
            n_simulations=self.n_simulations,
            horizon_days=self.horizon_days,
        ))
        fine_result = fine_simulator.simulate_policy(
            sku_config,
            best_pareto.reorder_point,
            best_pareto.order_quantity,
            policy_type,
        )

        # ── Politique Wilson naïve pour comparaison Lean ─────────────────────
        naive = self._wilson_policy(sku_config)
        naive_sim = fine_simulator.simulate_policy(
            sku_config, naive.reorder_point, naive.order_quantity, "(r,Q)"
        )
        naive_pareto = ParetoPoint(
            reorder_point=naive.reorder_point,
            order_quantity=naive.order_quantity,
            total_cost=naive_sim.mean_cost,
            stockout_prob=naive_sim.stockout_probability,
            service_level=naive_sim.mean_service_level,
            safety_stock=naive_sim.safety_stock_implied,
        )

        lean = optimizer.lean_analysis(best_pareto, naive_pareto, sku_config)

        # ── Construire StockPolicy ────────────────────────────────────────────
        policy = StockPolicy(
            sku_id=sku_config.sku_id,
            policy_type=policy_type,
            reorder_point=best_pareto.reorder_point,
            order_quantity=best_pareto.order_quantity,
            safety_stock=fine_result.safety_stock_implied,
            expected_annual_cost=fine_result.mean_cost,
            service_level=fine_result.mean_service_level,
            expected_stockout_days=fine_result.mean_stockout_days,
            expected_avg_stock=fine_result.mean_avg_stock,
            capital_reduction_pct=lean.get('reduction_capital_pct'),
            stockout_reduction_pct=lean.get('amelioration_taux_service_pp'),
        )

        return policy, frontier, lean

    def _wilson_policy(self, sku_config: SKUConfig) -> StockPolicy:
        """
        Politique Wilson (EOQ) comme baseline naïve.
        Hypothèses : demande constante, délai certain, pas de pénurie.
        """
        D  = sku_config.base_demand * 365   # Demande annuelle
        K  = sku_config.ordering_cost
        h  = sku_config.holding_cost_rate * sku_config.unit_cost * 365  # Coût annuel

        eoq = np.sqrt(2 * D * K / max(h, 1e-6))

        mean_lt     = np.exp(sku_config.lead_time_mu + sku_config.lead_time_sigma**2 / 2)
        reorder_pt  = sku_config.base_demand * mean_lt
        safety      = 1.65 * sku_config.base_demand * sku_config.noise_sigma * np.sqrt(mean_lt)

        return StockPolicy(
            sku_id=sku_config.sku_id,
            policy_type="(r,Q)_Wilson",
            reorder_point=reorder_pt + safety,
            order_quantity=eoq,
            safety_stock=safety,
            expected_annual_cost=0,
            service_level=0.95,
            expected_stockout_days=0,
            expected_avg_stock=eoq / 2 + safety,
        )

    def _wilson_fallback(
        self, sku_config: SKUConfig, target_service_level: float
    ) -> StockPolicy:
        return self._wilson_policy(sku_config)
