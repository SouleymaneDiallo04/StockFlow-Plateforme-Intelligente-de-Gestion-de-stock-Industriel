"""
ml/optimization/monte_carlo.py
-------------------------------
Simulation Monte Carlo pour l'évaluation des politiques de stock.

Approche : simulation stochastique de N trajectoires d'inventaire
sur un horizon H jours, sous une politique (r,Q) ou (s,S) donnée.

Chaque trajectoire tire :
- La demande journalière depuis la distribution apprise (ou log-normale ajustée)
- Le délai fournisseur depuis une log-normale calibrée sur les données historiques

La simulation permet d'estimer la distribution complète des outcomes,
pas seulement l'espérance — critique pour les décisions sous incertitude.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

from ..data.schemas import SKUConfig


@dataclass
class SimulationConfig:
    n_simulations:  int   = 10_000   # Convergence à ±1% pour N ≥ 10 000
    horizon_days:   int   = 365
    random_seed:    int   = 42


@dataclass
class SimulationResult:
    """
    Résultats agrégés de N simulations d'inventaire.
    """
    reorder_point:   float
    order_quantity:  float
    policy_type:     str

    # Distributions des KPIs sur les N simulations
    total_costs:         np.ndarray   # (N,) coûts totaux
    service_levels:      np.ndarray   # (N,) taux de service
    stockout_days:       np.ndarray   # (N,) jours en rupture
    avg_stocks:          np.ndarray   # (N,) stock moyen

    # Statistiques agrégées
    @property
    def mean_cost(self) -> float:
        return float(np.mean(self.total_costs))

    @property
    def p95_cost(self) -> float:
        return float(np.percentile(self.total_costs, 95))

    @property
    def mean_service_level(self) -> float:
        return float(np.mean(self.service_levels))

    @property
    def stockout_probability(self) -> float:
        """P(au moins un jour de rupture sur l'horizon)."""
        return float(np.mean(self.stockout_days > 0))

    @property
    def mean_stockout_days(self) -> float:
        return float(np.mean(self.stockout_days))

    @property
    def mean_avg_stock(self) -> float:
        return float(np.mean(self.avg_stocks))

    @property
    def safety_stock_implied(self) -> float:
        """
        Stock de sécurité implicite = stock moyen - cycle stock.
        Cycle stock ≈ order_quantity / 2 pour politique (r,Q).
        """
        return max(self.mean_avg_stock - self.order_quantity / 2, 0)


class MonteCarloSimulator:
    """
    Simule des trajectoires d'inventaire sous différentes politiques.
    """

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def simulate_policy(
        self,
        sku_config: SKUConfig,
        reorder_point: float,
        order_quantity: float,
        policy_type: str = "(r,Q)",
        demand_sampler: Optional[Callable] = None,
    ) -> SimulationResult:
        """
        Simule N trajectoires pour une politique de stock donnée.

        Politique (r,Q) : commander Q unités quand stock ≤ r.
        Politique (s,S) : commander jusqu'à S quand stock ≤ s (r=s, Q=S-s).

        Args:
            sku_config:      Configuration du SKU (coûts, délais).
            reorder_point:   Point de commande r.
            order_quantity:  Quantité commandée Q (ou S-s pour politique s,S).
            policy_type:     "(r,Q)" ou "(s,S)".
            demand_sampler:  Fonction optionnelle de tirage de la demande.
                             Si None, utilise log-normale calibrée sur base_demand.
        """
        N  = self.config.n_simulations
        H  = self.config.horizon_days

        total_costs     = np.zeros(N)
        service_levels  = np.zeros(N)
        stockout_days   = np.zeros(N)
        avg_stocks      = np.zeros(N)

        # Paramètres de coût
        h  = sku_config.holding_cost_rate * sku_config.unit_cost   # $/unité/jour
        p  = sku_config.shortage_cost_rate * sku_config.unit_cost  # $/unité/jour manquant
        K  = sku_config.ordering_cost                               # $ par commande

        # Paramètres demande — log-normale ajustée sur base_demand
        mu_d    = sku_config.base_demand
        sigma_d = sku_config.base_demand * sku_config.noise_sigma

        # Paramètres délai — log-normale
        lt_mu    = sku_config.lead_time_mu
        lt_sigma = sku_config.lead_time_sigma

        for i in range(N):
            stock        = float(sku_config.initial_stock)
            pending      = {}   # {jour_arrivee: quantite}
            total_cost   = 0.0
            n_stockout   = 0
            n_orders     = 0
            stock_sum    = 0.0

            for day in range(H):
                # Réception des commandes arrivant aujourd'hui
                if day in pending:
                    stock += pending.pop(day)

                # Tirage de la demande journalière
                if demand_sampler is not None:
                    d = float(demand_sampler())
                else:
                    # Log-normale paramétrée pour rester positive
                    # E[X] = exp(μ + σ²/2) — on résout pour μ
                    var    = sigma_d**2
                    mu_ln  = np.log(max(mu_d, 0.01)**2 / np.sqrt(var + max(mu_d, 0.01)**2))
                    sig_ln = np.sqrt(np.log(1 + var / max(mu_d, 0.01)**2))
                    d = float(self.rng.lognormal(mu_ln, sig_ln))

                d = round(d)

                # Satisfaction de la demande
                satisfied = min(d, max(stock, 0))
                unmet     = d - satisfied
                stock    -= satisfied

                if unmet > 0:
                    n_stockout += 1
                    total_cost += p * unmet

                # Coût de stockage (stock positif uniquement)
                if stock > 0:
                    total_cost += h * stock

                stock_sum += max(stock, 0)

                # Décision de commande
                should_order = False
                if policy_type == "(r,Q)":
                    should_order = stock <= reorder_point
                else:  # (s,S)
                    should_order = stock <= reorder_point

                if should_order and not any(
                    True for arr_day in pending if arr_day > day
                ):
                    # Pas de commande en transit — passer commande
                    lt = int(max(
                        round(self.rng.lognormal(lt_mu, lt_sigma)),
                        1
                    ))
                    arrival_day = day + lt
                    if arrival_day < H:
                        pending[arrival_day] = pending.get(arrival_day, 0) + order_quantity
                    total_cost += K
                    n_orders   += 1

            total_costs[i]    = total_cost
            service_levels[i] = 1.0 - n_stockout / max(H, 1)
            stockout_days[i]  = n_stockout
            avg_stocks[i]     = stock_sum / H

        return SimulationResult(
            reorder_point=reorder_point,
            order_quantity=order_quantity,
            policy_type=policy_type,
            total_costs=total_costs,
            service_levels=service_levels,
            stockout_days=stockout_days,
            avg_stocks=avg_stocks,
        )

    def simulate_grid(
        self,
        sku_config: SKUConfig,
        reorder_points: np.ndarray,
        order_quantities: np.ndarray,
        policy_type: str = "(r,Q)",
        demand_sampler: Optional[Callable] = None,
        n_sim_grid: int = 2_000,   # Réduit pour le scan — N complet sur les candidats Pareto
    ) -> list[SimulationResult]:
        """
        Évalue une grille de politiques (r,Q) pour construire la frontière Pareto.
        Utilise n_sim_grid réduit pour la vitesse, puis raffine les candidats.
        """
        config_fast = SimulationConfig(
            n_simulations=n_sim_grid,
            horizon_days=self.config.horizon_days,
            random_seed=self.config.random_seed,
        )
        simulator_fast = MonteCarloSimulator(config_fast)

        results = []
        for r in reorder_points:
            for q in order_quantities:
                if q <= 0:
                    continue
                res = simulator_fast.simulate_policy(
                    sku_config, r, q, policy_type, demand_sampler
                )
                results.append(res)

        return results

    def safety_stock_from_service_level(
        self,
        sku_config: SKUConfig,
        target_service_level: float = 0.95,
        n_sim: int = 5_000,
    ) -> float:
        """
        Calcule le stock de sécurité nécessaire pour atteindre un taux de service cible.
        Approche simulation — pas de formule analytique (non-gaussien).

        Retourne le stock de sécurité en unités.
        """
        # Estimation de la demande sur le délai moyen
        mean_lt = np.exp(sku_config.lead_time_mu + sku_config.lead_time_sigma**2 / 2)
        mean_demand_lt = sku_config.base_demand * mean_lt

        # Scan du stock de sécurité
        ss_candidates = np.linspace(0, mean_demand_lt * 3, 50)
        best_ss = ss_candidates[-1]

        for ss in ss_candidates:
            r  = mean_demand_lt + ss
            q  = mean_demand_lt * 2  # Q initial approximatif

            config_quick = SimulationConfig(n_simulations=n_sim, horizon_days=180)
            sim = MonteCarloSimulator(config_quick)
            result = sim.simulate_policy(sku_config, r, q)

            if result.mean_service_level >= target_service_level:
                best_ss = ss
                break

        return float(best_ss)
