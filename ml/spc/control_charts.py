"""
ml/spc/control_charts.py
-------------------------
Implémentation des cartes de contrôle statistique (SPC/MSP).

Cartes implémentées :
1. X-bar / R  : carte classique pour données continues groupées
2. I-MR       : carte individus / étendues mobiles (données non groupées — notre cas)
3. p-chart    : proportion de défauts (taux de rupture journalier)
4. CUSUM      : Cumulative Sum — détection de petits décalages de moyenne
5. EWMA       : Exponentially Weighted Moving Average — alternative au CUSUM

Référence : Montgomery, D.C. (2020). "Introduction to Statistical Quality Control", 8e éd.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from ..data.schemas import ControlLimits, ControlChartResult, SPCSignal, WERule
from .western_electric import WesternElectricRules


# ── Constantes de construction des cartes de contrôle ────────────────────────
# Facteurs tabulés (Montgomery, Table VI) pour sous-groupes de taille n=1 (I-MR)
D3_n2 = 0       # Limite inférieure pour carte R, n=2
D4_n2 = 3.267   # Limite supérieure pour carte R, n=2
d2_n2 = 1.128   # Facteur d2 pour estimation de sigma depuis étendue moyenne, n=2


class IMRChart:
    """
    Carte Individus-Étendues Mobiles (I-MR).

    Appropriée pour les données continues non groupées avec
    une observation par période — notre cas (demande journalière).

    σ est estimé depuis les étendues mobiles (MR) plutôt que l'écart-type
    global — cette estimation est moins sensible aux décalages de moyenne
    qui pourraient gonfler artificiellement l'estimation de σ.
    """

    def compute(
        self,
        series: np.ndarray,
        dates: np.ndarray,
        metric_name: str = "Demande",
        sku_id: str = "",
        phase: str = "I",  # Phase I : établissement limites | Phase II : surveillance
    ) -> ControlChartResult:

        values = series.astype(float)
        n = len(values)

        if n < 10:
            raise ValueError(f"Minimum 10 observations requis pour I-MR, reçu {n}")

        # ── Estimation de σ depuis les étendues mobiles ───────────────────────
        moving_ranges = np.abs(np.diff(values))
        mr_bar = np.mean(moving_ranges)
        sigma_estimated = mr_bar / d2_n2  # Estimateur non-biaisé de σ

        # ── Limites carte I (individus) ───────────────────────────────────────
        center = np.mean(values)
        ucl = center + 3 * sigma_estimated
        lcl = center - 3 * sigma_estimated  # Peut être négatif — on ne tronque pas ici

        limits = ControlLimits(
            center_line=round(center, 4),
            ucl=round(ucl, 4),
            lcl=round(max(lcl, 0), 4),  # Demande non négative
            uwa=round(center + 2 * sigma_estimated, 4),
            lwa=round(center - 2 * sigma_estimated, 4),
            uwb=round(center + 1 * sigma_estimated, 4),
            lwb=round(center - 1 * sigma_estimated, 4),
            sigma=round(sigma_estimated, 4),
        )

        # ── Détection des signaux Western Electric ────────────────────────────
        we_rules = WesternElectricRules()
        signals = we_rules.detect_all(values, dates, limits)

        in_control = len([s for s in signals if s.severity == 'critical']) == 0

        return ControlChartResult(
            sku_id=sku_id,
            chart_type="I-MR",
            metric_name=metric_name,
            dates=dates,
            values=values,
            limits=limits,
            signals=signals,
            in_control=in_control,
        )


class PChart:
    """
    Carte p — proportion de jours en rupture (stock = 0).

    Utilisée pour surveiller le taux de rupture sur une fenêtre glissante.
    La taille de sous-groupe variable est gérée par les limites de contrôle
    calculées individuellement pour chaque point (Laney p'-chart si surdispersion).
    """

    def __init__(self, window_days: int = 30):
        self.window_days = window_days

    def compute(
        self,
        stock_series: np.ndarray,
        dates: np.ndarray,
        sku_id: str = "",
    ) -> ControlChartResult:

        # Proportions sur fenêtre glissante
        n = len(stock_series)
        if n < self.window_days * 2:
            raise ValueError(f"Série trop courte pour p-chart avec fenêtre {self.window_days}")

        stockout_flags = (stock_series == 0).astype(float)

        # Calcul des proportions par fenêtre
        proportions = []
        chart_dates = []
        for i in range(self.window_days, n):
            window = stockout_flags[i - self.window_days:i]
            proportions.append(float(np.mean(window)))
            chart_dates.append(dates[i])

        p_arr   = np.array(proportions)
        d_arr   = np.array(chart_dates)
        p_bar   = float(np.mean(p_arr))
        n_sub   = float(self.window_days)

        # Limites 3-sigma pour proportion
        sigma_p = np.sqrt(p_bar * (1 - p_bar) / n_sub)
        ucl = min(p_bar + 3 * sigma_p, 1.0)
        lcl = max(p_bar - 3 * sigma_p, 0.0)

        limits = ControlLimits(
            center_line=round(p_bar, 6),
            ucl=round(ucl, 6),
            lcl=round(lcl, 6),
            uwa=round(min(p_bar + 2 * sigma_p, 1.0), 6),
            lwa=round(max(p_bar - 2 * sigma_p, 0.0), 6),
            uwb=round(min(p_bar + 1 * sigma_p, 1.0), 6),
            lwb=round(max(p_bar - 1 * sigma_p, 0.0), 6),
            sigma=round(sigma_p, 6),
        )

        we_rules = WesternElectricRules()
        signals  = we_rules.detect_all(p_arr, d_arr, limits)
        in_control = len([s for s in signals if s.severity == 'critical']) == 0

        return ControlChartResult(
            sku_id=sku_id,
            chart_type="p-chart",
            metric_name="Taux de rupture (%)",
            dates=d_arr,
            values=p_arr * 100,
            limits=ControlLimits(
                center_line=round(p_bar * 100, 4),
                ucl=round(ucl * 100, 4),
                lcl=round(lcl * 100, 4),
                uwa=round(limits.uwa * 100, 4),
                lwa=round(limits.lwa * 100, 4),
                uwb=round(limits.uwb * 100, 4),
                lwb=round(limits.lwb * 100, 4),
                sigma=round(sigma_p * 100, 4),
            ),
            signals=signals,
            in_control=in_control,
        )


class CUSUMChart:
    """
    Carte CUSUM (Cumulative Sum).

    Détecte des décalages de moyenne de l'ordre de 1-2σ que
    la carte I-MR (Shewhart) manque systématiquement.

    Paramétrage standard (Page, 1954; Montgomery, 2020) :
    - k = 0.5σ (slack value) — optimisé pour détecter un décalage de 1σ
    - h = 5σ (seuil de décision)

    ARL₀ ≈ 465 (longueur moyenne de run sous contrôle) — acceptable.
    """

    def __init__(self, k_sigma: float = 0.5, h_sigma: float = 5.0):
        self.k_sigma = k_sigma
        self.h_sigma = h_sigma

    def compute(
        self,
        series: np.ndarray,
        dates: np.ndarray,
        metric_name: str = "CUSUM",
        sku_id: str = "",
    ) -> ControlChartResult:

        values = series.astype(float)
        mu0    = np.mean(values)
        sigma  = np.std(values, ddof=1)

        if sigma < 1e-10:
            sigma = 1.0  # Série constante — CUSUM non applicable

        k = self.k_sigma * sigma
        h = self.h_sigma * sigma

        # CUSUM bilatéral
        c_plus  = np.zeros(len(values))
        c_minus = np.zeros(len(values))

        for i in range(1, len(values)):
            c_plus[i]  = max(0, c_plus[i-1]  + (values[i] - mu0) - k)
            c_minus[i] = max(0, c_minus[i-1] - (values[i] - mu0) - k)

        # Détection des dépassements du seuil h
        signals = []
        we_rules = WesternElectricRules()

        for i in range(len(values)):
            date_str = str(dates[i])[:10] if dates is not None else str(i)
            if c_plus[i] > h:
                signals.append(SPCSignal(
                    rule=WERule.RULE_1,
                    point_index=i,
                    date=date_str,
                    value=round(float(values[i]), 4),
                    description=f"CUSUM+ dépasse le seuil h={h:.2f} — dérive positive de la moyenne détectée",
                    severity='critical',
                ))
            elif c_minus[i] > h:
                signals.append(SPCSignal(
                    rule=WERule.RULE_1,
                    point_index=i,
                    date=date_str,
                    value=round(float(values[i]), 4),
                    description=f"CUSUM− dépasse le seuil h={h:.2f} — dérive négative de la moyenne détectée",
                    severity='critical',
                ))

        # Exposer C+ comme valeur principale de la carte
        cusum_display = c_plus - c_minus

        limits = ControlLimits(
            center_line=0.0,
            ucl=round(h, 4),
            lcl=round(-h, 4),
            uwa=round(h * 0.67, 4),
            lwa=round(-h * 0.67, 4),
            uwb=round(h * 0.33, 4),
            lwb=round(-h * 0.33, 4),
            sigma=round(sigma, 4),
        )

        return ControlChartResult(
            sku_id=sku_id,
            chart_type="CUSUM",
            metric_name=metric_name,
            dates=dates,
            values=cusum_display,
            limits=limits,
            signals=signals,
            in_control=len(signals) == 0,
        )


class EWMAChart:
    """
    Carte EWMA (Exponentially Weighted Moving Average).

    λ ∈ (0,1] : poids des observations récentes.
    λ = 0.1-0.3 : détection optimale des petits décalages.
    λ = 1.0 : équivalent à la carte Shewhart.

    Avantage vs CUSUM : plus intuitif (lissage exponentiel connu des praticiens).
    Inconvénient : légèrement moins puissant que CUSUM pour les décalages de précisément 1σ.
    """

    def __init__(self, lambda_: float = 0.2, L: float = 3.0):
        self.lambda_ = lambda_
        self.L = L

    def compute(
        self,
        series: np.ndarray,
        dates: np.ndarray,
        metric_name: str = "EWMA",
        sku_id: str = "",
    ) -> ControlChartResult:

        values = series.astype(float)
        mu0    = np.mean(values)
        sigma  = np.std(values, ddof=1)
        lam    = self.lambda_

        # Calcul EWMA
        ewma = np.zeros(len(values))
        ewma[0] = values[0]
        for i in range(1, len(values)):
            ewma[i] = lam * values[i] + (1 - lam) * ewma[i-1]

        # Limites variables (dépendent de i) — approximées par limite asymptotique
        # L limite asymptotique : ±L × σ × sqrt(λ/(2-λ))
        sigma_ewma = sigma * np.sqrt(lam / (2 - lam))

        ucl = mu0 + self.L * sigma_ewma
        lcl = mu0 - self.L * sigma_ewma

        limits = ControlLimits(
            center_line=round(mu0, 4),
            ucl=round(ucl, 4),
            lcl=round(max(lcl, 0), 4),
            uwa=round(mu0 + 2/3 * self.L * sigma_ewma, 4),
            lwa=round(mu0 - 2/3 * self.L * sigma_ewma, 4),
            uwb=round(mu0 + 1/3 * self.L * sigma_ewma, 4),
            lwb=round(mu0 - 1/3 * self.L * sigma_ewma, 4),
            sigma=round(sigma_ewma, 4),
        )

        signals = []
        for i, (v, d) in enumerate(zip(ewma, dates)):
            date_str = str(d)[:10]
            if v > ucl:
                signals.append(SPCSignal(
                    rule=WERule.RULE_1,
                    point_index=i,
                    date=date_str,
                    value=round(float(v), 4),
                    description="EWMA au-dessus de UCL — augmentation de la demande moyenne",
                    severity='critical',
                ))
            elif v < lcl:
                signals.append(SPCSignal(
                    rule=WERule.RULE_1,
                    point_index=i,
                    date=date_str,
                    value=round(float(v), 4),
                    description="EWMA en-dessous de LCL — diminution de la demande moyenne",
                    severity='critical',
                ))

        return ControlChartResult(
            sku_id=sku_id,
            chart_type="EWMA",
            metric_name=metric_name,
            dates=dates,
            values=ewma,
            limits=limits,
            signals=signals,
            in_control=len(signals) == 0,
        )
