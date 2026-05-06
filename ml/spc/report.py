"""
ml/spc/report.py
-----------------
Génération automatique du rapport SPC pour un SKU.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
import numpy as np

from .control_charts import IMRChart, PChart, CUSUMChart, EWMAChart
from ..data.schemas import SPCReport, SPCSignal, ControlChartResult


class SPCReportGenerator:

    def generate(
        self,
        sku_id:       str,
        demand:       np.ndarray,
        dates:        np.ndarray,
        stock_series: Optional[np.ndarray] = None,
    ) -> SPCReport:
        """
        Génère le rapport SPC complet pour un SKU.

        Cartes générées :
        1. I-MR sur la demande journalière
        2. CUSUM sur la demande — détection dérives lentes
        3. EWMA sur la demande — lissage adaptatif
        4. p-chart sur le taux de rupture (si stock_series fourni)
        """
        charts: list[ControlChartResult] = []
        errors: list[str] = []

        # ── Carte I-MR ────────────────────────────────────────────────────────
        try:
            imr = IMRChart()
            charts.append(imr.compute(demand, dates, "Demande journalière", sku_id))
        except Exception as e:
            errors.append(f"I-MR : {e}")

        # ── CUSUM ─────────────────────────────────────────────────────────────
        try:
            cusum = CUSUMChart()
            charts.append(cusum.compute(demand, dates, "CUSUM Demande", sku_id))
        except Exception as e:
            errors.append(f"CUSUM : {e}")

        # ── EWMA ──────────────────────────────────────────────────────────────
        try:
            ewma = EWMAChart(lambda_=0.2)
            charts.append(ewma.compute(demand, dates, "EWMA Demande", sku_id))
        except Exception as e:
            errors.append(f"EWMA : {e}")

        # ── p-chart (si données stock disponibles) ────────────────────────────
        if stock_series is not None and len(stock_series) == len(demand):
            try:
                pchart = PChart(window_days=30)
                charts.append(pchart.compute(stock_series, dates, sku_id))
            except Exception as e:
                errors.append(f"p-chart : {e}")

        # ── Agrégation des signaux critiques ──────────────────────────────────
        all_critical = [
            s for c in charts
            for s in c.signals
            if s.severity == 'critical'
        ]

        # ── Statut global ──────────────────────────────────────────────────────
        n_critical = len(all_critical)
        n_warning  = sum(1 for c in charts for s in c.signals if s.severity == 'warning')

        if n_critical > 3:
            status = "out_of_control"
        elif n_critical > 0 or n_warning > 5:
            status = "warning"
        else:
            status = "in_control"

        # ── Recommandations ───────────────────────────────────────────────────
        recommendations = self._build_recommendations(charts, status, sku_id)

        return SPCReport(
            sku_id=sku_id,
            generated_at=datetime.now().isoformat(),
            charts=charts,
            overall_status=status,
            critical_signals=all_critical[:10],  # Top 10
            recommendations=recommendations,
        )

    def _build_recommendations(
        self,
        charts: list[ControlChartResult],
        status: str,
        sku_id: str,
    ) -> list[str]:
        recs = []

        if status == "out_of_control":
            recs.append(
                f"URGENT — Le processus de demande de {sku_id} est hors contrôle statistique. "
                "Identifier et éliminer les causes assignables avant toute action sur les seuils de stock."
            )

        # Analyser les types de signaux dominants
        signal_rules = []
        for chart in charts:
            for s in chart.signals:
                signal_rules.append(s.rule.value)

        from collections import Counter
        rule_counts = Counter(signal_rules)

        if rule_counts.get('R2', 0) > 0:
            recs.append(
                "Dérive de la moyenne détectée (Règle 2). "
                "Recalibrer le modèle de prévision avec les données récentes. "
                "En contexte Lean : vérifier si un événement externe (nouveau client, "
                "saisonnalité non capturée) explique le décalage."
            )

        if rule_counts.get('R3', 0) > 0:
            recs.append(
                "Tendance longue détectée (Règle 3). "
                "Réviser le point de commande à la hausse si tendance croissante. "
                "Action Lean : activer le flux tiré sur ce SKU."
            )

        if rule_counts.get('R1', 0) > 2:
            recs.append(
                "Plusieurs points extrêmes détectés (Règle 1). "
                "Vérifier les données sources — possible erreur de saisie ou événement exceptionnel. "
                "Si réel : mettre à jour le modèle avec données récentes."
            )

        if not recs:
            recs.append(
                f"Processus de demande de {sku_id} sous contrôle statistique. "
                "Aucune action corrective requise. Maintenir la surveillance périodique."
            )

        return recs

    def summary_table(self, reports: list[SPCReport]) -> list[dict]:
        """Tableau synthétique multi-SKUs pour le dashboard."""
        rows = []
        for r in reports:
            n_critical = len(r.critical_signals)
            n_warning  = sum(
                1 for c in r.charts for s in c.signals if s.severity == 'warning'
            )
            rows.append({
                'sku_id':     r.sku_id,
                'status':     r.overall_status,
                'n_charts':   len(r.charts),
                'n_critical': n_critical,
                'n_warning':  n_warning,
                'generated':  r.generated_at[:10],
            })
        return sorted(rows, key=lambda x: (
            0 if x['status'] == 'out_of_control' else
            1 if x['status'] == 'warning' else 2
        ))
