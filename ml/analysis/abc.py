"""
ml/analysis/abc.py
------------------
Analyse ABC et XYZ du portefeuille de SKUs.

ABC : classification par valeur consommée (80/15/5 — loi de Pareto).
XYZ : classification par variabilité de la demande (CV).

Matrice ABC-XYZ :
  AX : fort chiffre d'affaires, demande stable    → stock minimal, approvisionnement précis
  AY : fort CA, demande variable                  → stock de sécurité modéré
  AZ : fort CA, demande erratique                 → attention particulière, commandes fréquentes
  BX/BY/BZ : CA moyen                             → gestion standard
  CX/CY/CZ : CA faible                            → réduire le nombre de références ou MOQ élevé

Référence : Wildemann (1984), repris dans Chopra & Meindl (2016).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class ABCResult:
    sku_id:         str
    abc_class:      str       # A, B ou C
    xyz_class:      str       # X, Y ou Z
    matrix_class:   str       # AX, AY, AZ, BX, etc.
    annual_value:   float     # Demande annuelle × coût unitaire
    value_pct:      float     # % de la valeur totale du portefeuille
    cumulative_pct: float     # % cumulé (pour courbe de Pareto)
    cv:             float     # Coefficient de variation de la demande
    mean_demand:    float
    total_demand:   float
    recommendation: str


class ABCXYZAnalyzer:
    """
    Classifie le portefeuille de SKUs selon la matrice ABC-XYZ.
    Fournit des recommandations de politique de gestion par classe.
    """

    # Seuils ABC standard (Pareto 80/15/5)
    ABC_THRESHOLDS = {'A': 0.80, 'B': 0.95}  # 0-80% = A, 80-95% = B, 95-100% = C

    # Seuils XYZ par coefficient de variation
    XYZ_THRESHOLDS = {'X': 0.25, 'Y': 0.75}  # CV < 0.25 = X, 0.25-0.75 = Y, >0.75 = Z

    RECOMMENDATIONS = {
        'AX': 'Flux tiré. Stock minimal. Approvisionnement juste-à-temps. Point de commande précis.',
        'AY': 'Stock de sécurité modéré. Prévoir la demande. Révision fréquente des paramètres.',
        'AZ': 'Gestion sur mesure. Commandes fréquentes. Approvisionnement d\'urgence possible.',
        'BX': 'Gestion standard. Paramètres stables. EOQ applicable.',
        'BY': 'Révision périodique des seuils. Stock de sécurité adapté à la variabilité.',
        'BZ': 'Surveiller les causes de variabilité. Envisager la réduction du nombre de fournisseurs.',
        'CX': 'Consolider les commandes. Évaluer si la référence est nécessaire.',
        'CY': 'Revoir la politique tarifaire. Regrouper avec d\'autres CY.',
        'CZ': 'Candidat à l\'élimination ou au make-on-order. Coût de stockage > valeur.',
    }

    def analyze(
        self,
        demand_df: pd.DataFrame,
        unit_costs: dict[str, float],
        sku_col: str = 'sku_id',
        demand_col: str = 'demand',
        date_col: str = 'date',
    ) -> list[ABCResult]:
        """
        Effectue l'analyse ABC-XYZ complète.

        Args:
            demand_df:  DataFrame long avec colonnes sku_id, date, demand.
            unit_costs: Dictionnaire {sku_id: coût_unitaire}.
        """
        results = []

        # Agréger par SKU
        grouped = demand_df.groupby(sku_col)[demand_col]
        stats = grouped.agg(['sum', 'mean', 'std']).reset_index()
        stats.columns = [sku_col, 'total_demand', 'mean_demand', 'std_demand']
        stats['cv'] = stats['std_demand'] / stats['mean_demand'].replace(0, np.nan)
        stats['cv'] = stats['cv'].fillna(0)

        # Valeur consommée annuelle = total_demand × coût_unitaire
        stats['unit_cost'] = stats[sku_col].map(lambda x: unit_costs.get(x, 1.0))
        n_days = demand_df[date_col].nunique() if date_col in demand_df.columns else 730
        annualization = 365 / max(n_days, 1)
        stats['annual_value'] = stats['total_demand'] * stats['unit_cost'] * annualization

        # Tri par valeur décroissante
        stats = stats.sort_values('annual_value', ascending=False).reset_index(drop=True)
        total_value = stats['annual_value'].sum()
        stats['value_pct']      = stats['annual_value'] / max(total_value, 1e-10)
        stats['cumulative_pct'] = stats['value_pct'].cumsum()

        # Classification ABC
        def abc_class(cum_pct):
            if cum_pct <= self.ABC_THRESHOLDS['A']:
                return 'A'
            elif cum_pct <= self.ABC_THRESHOLDS['B']:
                return 'B'
            return 'C'

        stats['abc'] = stats['cumulative_pct'].apply(abc_class)

        # Classification XYZ
        def xyz_class(cv):
            if cv <= self.XYZ_THRESHOLDS['X']:
                return 'X'
            elif cv <= self.XYZ_THRESHOLDS['Y']:
                return 'Y'
            return 'Z'

        stats['xyz'] = stats['cv'].apply(xyz_class)
        stats['matrix'] = stats['abc'] + stats['xyz']

        for _, row in stats.iterrows():
            rec = self.RECOMMENDATIONS.get(row['matrix'], 'Gestion standard.')
            results.append(ABCResult(
                sku_id        = row[sku_col],
                abc_class     = row['abc'],
                xyz_class     = row['xyz'],
                matrix_class  = row['matrix'],
                annual_value  = round(float(row['annual_value']), 2),
                value_pct     = round(float(row['value_pct']) * 100, 2),
                cumulative_pct= round(float(row['cumulative_pct']) * 100, 2),
                cv            = round(float(row['cv']), 4),
                mean_demand   = round(float(row['mean_demand']), 3),
                total_demand  = round(float(row['total_demand']), 1),
                recommendation= rec,
            ))

        return results

    def to_dataframe(self, results: list[ABCResult]) -> pd.DataFrame:
        rows = [
            {
                'SKU':           r.sku_id,
                'Classe ABC':    r.abc_class,
                'Classe XYZ':    r.xyz_class,
                'Matrice':       r.matrix_class,
                'Valeur annuelle (DH)': r.annual_value,
                'Part (%)':      r.value_pct,
                'Cumulé (%)':    r.cumulative_pct,
                'CV':            r.cv,
                'Recommandation': r.recommendation,
            }
            for r in results
        ]
        return pd.DataFrame(rows)

    def pareto_data(self, results: list[ABCResult]) -> dict:
        """Données pour le graphique de Pareto."""
        return {
            'sku_ids':        [r.sku_id for r in results],
            'values':         [r.annual_value for r in results],
            'cumulative_pct': [r.cumulative_pct for r in results],
            'abc_classes':    [r.abc_class for r in results],
            'matrix_classes': [r.matrix_class for r in results],
        }

    def portfolio_summary(self, results: list[ABCResult]) -> dict:
        """Résumé du portefeuille par classe."""
        summary = {}
        for cls in ['A', 'B', 'C']:
            subset = [r for r in results if r.abc_class == cls]
            if subset:
                summary[cls] = {
                    'n_skus':       len(subset),
                    'total_value':  round(sum(r.annual_value for r in subset), 2),
                    'value_pct':    round(sum(r.value_pct for r in subset), 1),
                    'xyz_breakdown': {
                        'X': sum(1 for r in subset if r.xyz_class == 'X'),
                        'Y': sum(1 for r in subset if r.xyz_class == 'Y'),
                        'Z': sum(1 for r in subset if r.xyz_class == 'Z'),
                    }
                }
        return summary
