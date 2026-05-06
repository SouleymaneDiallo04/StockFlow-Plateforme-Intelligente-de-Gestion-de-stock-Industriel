"""
ml/spc/western_electric.py
---------------------------
Implémentation des 8 règles Western Electric (WECO).

Source : Western Electric Company (1956). "Statistical Quality Control Handbook."
         Montgomery (2020), Section 5.3.

Rappel des 8 règles :
R1 : 1 point au-delà de ±3σ (Zone A)
R2 : 9 points consécutifs du même côté de la ligne centrale
R3 : 6 points consécutifs en tendance monotone (croissante ou décroissante)
R4 : 14 points consécutifs alternant haut/bas
R5 : 2 des 3 derniers points dans la même Zone A (au-delà de ±2σ)
R6 : 4 des 5 derniers points dans la même Zone B ou au-delà (au-delà de ±1σ)
R7 : 15 points consécutifs dans la Zone C (entre ±1σ)  — "hugging"
R8 : 8 points consécutifs en dehors de la Zone C (au-delà de ±1σ, des deux côtés)

Note sur le taux de fausses alarmes :
Chaque règle a un ARL₀ propre. Appliquer les 8 règles simultanément augmente
le taux de fausses alarmes. En pratique industrielle, R1+R2+R3 couvrent ~95%
des situations. Les règles R4-R8 sont signalées comme "warning" plutôt que "critical".
"""
from __future__ import annotations

import numpy as np

from ..data.schemas import ControlLimits, SPCSignal, WERule


# Sévérité par règle — calibrée sur l'ARL₀ de chaque règle
RULE_SEVERITY = {
    WERule.RULE_1: 'critical',   # ARL₀ ≈ 370 — signal fort
    WERule.RULE_2: 'critical',   # ARL₀ ≈ 250 — signal fort
    WERule.RULE_3: 'critical',   # ARL₀ ≈ 200 — signal fort
    WERule.RULE_4: 'warning',    # ARL₀ ≈ 50  — bruit plus fréquent
    WERule.RULE_5: 'warning',    # ARL₀ ≈ 90
    WERule.RULE_6: 'warning',    # ARL₀ ≈ 56
    WERule.RULE_7: 'info',       # ARL₀ ≈ 33  — très fréquent, souvent faux positif
    WERule.RULE_8: 'info',       # ARL₀ ≈ 50
}

RULE_DESCRIPTIONS = {
    WERule.RULE_1: "1 point au-delà de 3σ — cause assignable probable",
    WERule.RULE_2: "9 points consécutifs du même côté — dérive de la moyenne",
    WERule.RULE_3: "6 points en tendance monotone — tendance longue détectée",
    WERule.RULE_4: "14 points alternant haut/bas — oscillation systématique",
    WERule.RULE_5: "2 des 3 derniers points en Zone A — approche de la limite",
    WERule.RULE_6: "4 des 5 derniers points en Zone B — dispersion accrue",
    WERule.RULE_7: "15 points consécutifs en Zone C — stratification possible",
    WERule.RULE_8: "8 points consécutifs hors Zone C — mélange de distributions",
}


class WesternElectricRules:
    """
    Détecte les signaux hors-contrôle selon les 8 règles WECO.
    """

    def detect_all(
        self,
        values: np.ndarray,
        dates: np.ndarray,
        limits: ControlLimits,
        active_rules: list[WERule] = None,
    ) -> list[SPCSignal]:
        """
        Applique toutes les règles actives et retourne les signaux détectés.
        Par défaut : toutes les 8 règles.
        """
        if active_rules is None:
            active_rules = list(WERule)

        all_signals = []
        rule_methods = {
            WERule.RULE_1: self._rule_1,
            WERule.RULE_2: self._rule_2,
            WERule.RULE_3: self._rule_3,
            WERule.RULE_4: self._rule_4,
            WERule.RULE_5: self._rule_5,
            WERule.RULE_6: self._rule_6,
            WERule.RULE_7: self._rule_7,
            WERule.RULE_8: self._rule_8,
        }

        for rule in active_rules:
            if rule in rule_methods:
                signals = rule_methods[rule](values, dates, limits)
                all_signals.extend(signals)

        # Dédupliquer par (index, règle) et trier chronologiquement
        seen = set()
        unique_signals = []
        for s in sorted(all_signals, key=lambda x: x.point_index):
            key = (s.point_index, s.rule)
            if key not in seen:
                seen.add(key)
                unique_signals.append(s)

        return unique_signals

    def _make_signal(
        self,
        rule: WERule,
        index: int,
        dates: np.ndarray,
        value: float,
        extra_note: str = "",
    ) -> SPCSignal:
        date_str = str(dates[index])[:10] if dates is not None and index < len(dates) else str(index)
        base_desc = RULE_DESCRIPTIONS[rule]
        description = f"{base_desc}{'. ' + extra_note if extra_note else ''}"
        return SPCSignal(
            rule=rule,
            point_index=index,
            date=date_str,
            value=round(float(value), 4),
            description=description,
            severity=RULE_SEVERITY[rule],
        )

    def _rule_1(self, v, d, lim) -> list[SPCSignal]:
        """1 point au-delà de ±3σ."""
        signals = []
        for i, val in enumerate(v):
            if val > lim.ucl or val < lim.lcl:
                side = "supérieure" if val > lim.ucl else "inférieure"
                signals.append(self._make_signal(WERule.RULE_1, i, d, val,
                    f"Limite {side} franchie ({val:.2f} vs {lim.ucl:.2f}/{lim.lcl:.2f})"))
        return signals

    def _rule_2(self, v, d, lim) -> list[SPCSignal]:
        """9 points consécutifs du même côté de la ligne centrale."""
        signals = []
        above = (v > lim.center_line).astype(int)
        below = (v < lim.center_line).astype(int)
        for arr, side in [(above, "au-dessus"), (below, "en-dessous")]:
            run = 0
            for i, a in enumerate(arr):
                run = run + 1 if a else 0
                if run == 9:
                    signals.append(self._make_signal(WERule.RULE_2, i, d, v[i],
                        f"9 points consécutifs {side} de CL — dérive à surveiller"))
        return signals

    def _rule_3(self, v, d, lim) -> list[SPCSignal]:
        """6 points consécutifs en tendance monotone."""
        signals = []
        for i in range(5, len(v)):
        # Vérifier tendance sur les 6 derniers points (i-5 à i)
            window = v[i-5:i+1]
            diffs  = np.diff(window)
            if np.all(diffs > 0) or np.all(diffs < 0):
                direction = "croissante" if diffs[0] > 0 else "décroissante"
                signals.append(self._make_signal(WERule.RULE_3, i, d, v[i],
                    f"Tendance {direction} sur 6 points — vérifier cause assignable"))
        return signals

    def _rule_4(self, v, d, lim) -> list[SPCSignal]:
        """14 points alternant haut/bas."""
        signals = []
        if len(v) < 14:
            return signals
        for i in range(13, len(v)):
            window = v[i-13:i+1]
            diffs  = np.diff(window)
            alternating = np.all(diffs[::2] > 0) and np.all(diffs[1::2] < 0)
            alternating |= np.all(diffs[::2] < 0) and np.all(diffs[1::2] > 0)
            if alternating:
                signals.append(self._make_signal(WERule.RULE_4, i, d, v[i],
                    "Oscillation systématique — possible sur-contrôle ou perturbation cyclique"))
        return signals

    def _rule_5(self, v, d, lim) -> list[SPCSignal]:
        """2 des 3 derniers points en Zone A (au-delà de ±2σ), même côté."""
        signals = []
        if len(v) < 3:
            return signals
        for i in range(2, len(v)):
            window = v[i-2:i+1]
            above_2s = window > lim.uwa
            below_2s = window < lim.lwa
            if np.sum(above_2s) >= 2 or np.sum(below_2s) >= 2:
                signals.append(self._make_signal(WERule.RULE_5, i, d, v[i],
                    "2/3 points en Zone A — processus potentiellement hors contrôle"))
        return signals

    def _rule_6(self, v, d, lim) -> list[SPCSignal]:
        """4 des 5 derniers points en Zone B ou au-delà (±1σ), même côté."""
        signals = []
        if len(v) < 5:
            return signals
        for i in range(4, len(v)):
            window = v[i-4:i+1]
            above_1s = window > lim.uwb
            below_1s = window < lim.lwb
            if np.sum(above_1s) >= 4 or np.sum(below_1s) >= 4:
                signals.append(self._make_signal(WERule.RULE_6, i, d, v[i],
                    "4/5 points en Zone B — dispersion du processus accrue"))
        return signals

    def _rule_7(self, v, d, lim) -> list[SPCSignal]:
        """15 points consécutifs en Zone C (entre ±1σ) — stratification."""
        signals = []
        in_c = ((v >= lim.lwb) & (v <= lim.uwb)).astype(int)
        run = 0
        for i, c in enumerate(in_c):
            run = run + 1 if c else 0
            if run == 15:
                signals.append(self._make_signal(WERule.RULE_7, i, d, v[i],
                    "Stratification possible — données issues de plusieurs distributions"))
        return signals

    def _rule_8(self, v, d, lim) -> list[SPCSignal]:
        """8 points consécutifs hors Zone C (au-delà de ±1σ), des deux côtés."""
        signals = []
        out_c = ((v > lim.uwb) | (v < lim.lwb)).astype(int)
        run = 0
        for i, o in enumerate(out_c):
            run = run + 1 if o else 0
            if run == 8:
                signals.append(self._make_signal(WERule.RULE_8, i, d, v[i],
                    "Mélange de distributions — vérifier segmentation des données"))
        return signals
