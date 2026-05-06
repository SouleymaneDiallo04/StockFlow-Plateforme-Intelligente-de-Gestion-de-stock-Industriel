"""
ml/data/schemas.py
------------------
Dataclasses définissant les structures de données partagées entre tous les modules ML.
Aucune dépendance Django — ce module doit rester importable hors contexte web.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


# ── Enumerations ──────────────────────────────────────────────────────────────

class DemandProfile(str, Enum):
    FAST_MOVER  = "fast_mover"   # Forte rotation, demande quasi-continue
    SLOW_MOVER  = "slow_mover"   # Faible rotation, épisodes sporadiques
    SEASONAL    = "seasonal"     # Saisonnalité dominante
    TRENDING    = "trending"     # Tendance longue (haussière ou baissière)
    LUMPY       = "lumpy"        # Demande groupée, irrégulière — modèle Croston


class ProductCategory(str, Enum):
    COMPOSANTS      = "Composants électroniques"
    MATIERES        = "Matières premières"
    CONSOMMABLES    = "Consommables"
    EQUIPEMENTS     = "Équipements"


class ForecastModel(str, Enum):
    SARIMAX  = "SARIMAX"
    PROPHET  = "Prophet"
    LGBM     = "LightGBM"
    TFT      = "TFT"


# ── SKU Configuration ─────────────────────────────────────────────────────────

@dataclass
class SKUConfig:
    """
    Paramètres complets d'un SKU pour la génération synthétique.
    Chaque paramètre est physiquement interprétable — pas de magic numbers.
    """
    sku_id:          str
    name:            str
    category:        ProductCategory
    profile:         DemandProfile

    # Niveau de base de la demande journalière (unités/jour)
    base_demand:     float

    # Tendance — unités/jour/jour (peut être négatif)
    trend_slope:     float = 0.0

    # Courbure de tendance — permet tendances sub/sur-linéaires
    trend_quadratic: float = 0.0

    # Amplitude relative de la saisonnalité hebdomadaire [0, 1]
    weekly_amplitude:  float = 0.0

    # Amplitude relative de la saisonnalité mensuelle [0, 1]
    monthly_amplitude: float = 0.0

    # Phase hebdomadaire (radians) — décale le pic de demande dans la semaine
    weekly_phase:    float = 0.0

    # Écart-type relatif du bruit multiplicatif — ex: 0.15 = ±15% de bruit
    noise_sigma:     float = 0.15

    # Probabilité journalière d'un événement exogène (promo, panne, etc.)
    event_probability: float = 0.02

    # Amplitude des événements en multiple de la demande de base
    event_amplitude:   float = 2.5

    # Paramètres délai fournisseur — distribution log-normale
    lead_time_mu:    float = 2.0   # log-moyenne (correspond à ~7-8 jours)
    lead_time_sigma: float = 0.4   # log-écart-type

    # Structure de coûts
    unit_cost:       float = 100.0
    holding_cost_rate: float = 0.002  # fraction de unit_cost par unité par jour
    shortage_cost_rate: float = 0.015 # fraction de unit_cost — asymétrique x7.5
    ordering_cost:   float = 50.0

    # Stock initial
    initial_stock:   int = 50

    # Pour profil intermittent (lumpy/slow_mover)
    intermittency_prob: float = 0.3  # Probabilité qu'un jour ait demande > 0


@dataclass
class GeneratedSKU:
    """Résultat complet de la génération pour un SKU."""
    config:          SKUConfig
    dates:           np.ndarray      # array de dates
    demand:          np.ndarray      # demande journalière
    lead_times:      np.ndarray      # délais fournisseurs simulés
    events:          np.ndarray      # indicateur événements (0/1)
    trend_component:     np.ndarray
    seasonal_weekly:     np.ndarray
    seasonal_monthly:    np.ndarray
    noise_component:     np.ndarray

    @property
    def n_days(self) -> int:
        return len(self.demand)

    @property
    def mean_demand(self) -> float:
        return float(np.mean(self.demand))

    @property
    def cv_demand(self) -> float:
        """Coefficient de variation — mesure de la volatilité."""
        return float(np.std(self.demand) / np.mean(self.demand))

    @property
    def intermittency_rate(self) -> float:
        """Part des jours avec demande nulle."""
        return float(np.mean(self.demand == 0))


# ── Forecast Results ──────────────────────────────────────────────────────────

@dataclass
class ForecastResult:
    """
    Résultat standardisé pour tous les modèles de prévision.
    Le format uniforme permet la comparaison directe.
    """
    model_name:       str
    sku_id:           str
    forecast_dates:   np.ndarray
    point_forecast:   np.ndarray   # Prévision ponctuelle
    lower_bound:      np.ndarray   # Borne inférieure intervalle confiance
    upper_bound:      np.ndarray   # Borne supérieure intervalle confiance
    confidence_level: float        # Ex: 0.90 pour IC à 90%

    # Métriques d'évaluation — remplies après évaluation out-of-sample
    mase:             Optional[float] = None
    mae:              Optional[float] = None
    coverage_rate:    Optional[float] = None   # Validation calibration IC
    training_time_s:  Optional[float] = None


@dataclass
class ModelComparison:
    """Comparaison multi-modèles sur un SKU."""
    sku_id:   str
    results:  list[ForecastResult]
    best_model_by_mase: Optional[str] = None

    def ranked_by_mase(self) -> list[ForecastResult]:
        return sorted(
            [r for r in self.results if r.mase is not None],
            key=lambda r: r.mase
        )


# ── Optimization Results ──────────────────────────────────────────────────────

@dataclass
class StockPolicy:
    """
    Politique de réapprovisionnement optimale pour un SKU.
    Supporte les politiques (r,Q) et (s,S).
    """
    sku_id:              str
    policy_type:         str          # "(r,Q)" ou "(s,S)"

    # Point de commande — déclenche une commande quand stock ≤ reorder_point
    reorder_point:       float

    # Quantité commandée (politique r,Q) ou niveau de remplissage (politique s,S)
    order_quantity:      float

    # Stock de sécurité — amortisseur contre la variabilité
    safety_stock:        float

    # Métriques économiques
    expected_annual_cost:    float
    service_level:           float    # P(pas de rupture) sur l'horizon
    expected_stockout_days:  float    # Jours de rupture attendus par an
    expected_avg_stock:      float    # Stock moyen — capital immobilisé

    # Lien Lean — réduction vs politique naïve (Wilson statique)
    capital_reduction_pct:   Optional[float] = None
    stockout_reduction_pct:  Optional[float] = None


@dataclass
class ParetoPoint:
    """Un point sur la frontière de Pareto coût vs risque de rupture."""
    reorder_point:    float
    order_quantity:   float
    total_cost:       float
    stockout_prob:    float
    service_level:    float
    safety_stock:     float


# ── SPC Results ───────────────────────────────────────────────────────────────

class WERule(str, Enum):
    """8 règles Western Electric — numérotation standard."""
    RULE_1 = "R1"   # 1 point au-delà de 3σ
    RULE_2 = "R2"   # 9 points consécutifs du même côté de la moyenne
    RULE_3 = "R3"   # 6 points consécutifs en tendance monotone
    RULE_4 = "R4"   # 14 points alternant haut/bas
    RULE_5 = "R5"   # 2 des 3 derniers points au-delà de 2σ (même côté)
    RULE_6 = "R6"   # 4 des 5 derniers points au-delà de 1σ (même côté)
    RULE_7 = "R7"   # 15 points consécutifs dans la zone C (< 1σ)
    RULE_8 = "R8"   # 8 points consécutifs hors zone C (> 1σ, alternant)


@dataclass
class ControlLimits:
    """Limites de contrôle calculées pour une carte de contrôle."""
    center_line:  float    # CL — moyenne du processus
    ucl:          float    # Upper Control Limit (+3σ)
    lcl:          float    # Lower Control Limit (-3σ)
    uwa:          float    # Upper Warning A (+2σ)
    lwa:          float    # Lower Warning A (-2σ)
    uwb:          float    # Upper Warning B (+1σ)
    lwb:          float    # Lower Warning B (-1σ)
    sigma:        float    # Écart-type estimé du processus


@dataclass
class SPCSignal:
    """Signal hors-contrôle détecté sur une carte de contrôle."""
    rule:         WERule
    point_index:  int       # Index dans la série temporelle
    date:         str
    value:        float
    description:  str       # Interprétation métier du signal
    severity:     str       # "critical" | "warning" | "info"


@dataclass
class ControlChartResult:
    """Résultat complet d'une carte de contrôle."""
    sku_id:          str
    chart_type:      str          # "xbar_r", "p_chart", "cusum", "ewma"
    metric_name:     str
    dates:           np.ndarray
    values:          np.ndarray
    limits:          ControlLimits
    signals:         list[SPCSignal]
    in_control:      bool         # Processus globalement sous contrôle ?
    process_capability: Optional[float] = None  # Cp/Cpk si applicable


@dataclass
class SPCReport:
    """Rapport SPC complet pour un SKU — agrège toutes les cartes."""
    sku_id:           str
    generated_at:     str
    charts:           list[ControlChartResult]
    overall_status:   str         # "in_control" | "warning" | "out_of_control"
    critical_signals: list[SPCSignal]
    recommendations:  list[str]
