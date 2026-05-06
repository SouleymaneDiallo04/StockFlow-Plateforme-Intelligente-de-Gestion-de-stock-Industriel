"""
ml/data/generator.py
--------------------
Générateur de données de demande industrielle synthétiques.

Choix méthodologiques documentés :
- Bruit multiplicatif (pas additif) : la variance de la demande croît avec son niveau,
  propriété empiriquement validée sur les données de vente industrielles (Syntetos et al., 2005).
- Log-normale pour les délais : délais non-négatifs avec queue droite asymétrique —
  les retards extrêmes existent, les avances sont bornées (Stalk & Hout, 1990).
- Modèle de Croston pour les profils intermittents : sépare la probabilité d'occurrence
  de l'amplitude — standard en gestion de pièces de rechange industrielles.
- Événements exogènes injectés aléatoirement : pics de demande, ruptures fournisseur,
  promotions — rend les séries non-stationnaires de façon réaliste.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from .schemas import (
    SKUConfig, GeneratedSKU, DemandProfile, ProductCategory
)


# ── Seed catalog — 50 SKUs industriels réalistes ─────────────────────────────

def build_sku_catalog() -> list[SKUConfig]:
    """
    Construit le catalogue complet de 50 SKUs répartis sur 4 catégories.
    Les paramètres sont calibrés pour reproduire des comportements industriels réels.
    """
    catalog = []
    rng = np.random.default_rng(42)  # Reproductibilité

    # ── Composants électroniques (12 SKUs) ────────────────────────────────────
    # Forte rotation, saisonnalité faible, tendance liée aux cycles produits
    components = [
        ("COMP-001", "Résistance 10kΩ (lot 1000)",   80.0,  0.005,  0.12, 0.08),
        ("COMP-002", "Condensateur 100µF",            45.0,  0.003,  0.15, 0.10),
        ("COMP-003", "Microcontrôleur STM32F4",        8.0,  0.010,  0.20, 0.15),
        ("COMP-004", "Transistor NPN BC547",          120.0, 0.002,  0.10, 0.06),
        ("COMP-005", "Diode Zener 5.1V",              95.0,  0.001,  0.12, 0.07),
        ("COMP-006", "MOSFET IRF540",                 30.0, -0.002,  0.18, 0.12),
        ("COMP-007", "Régulateur LM7805",             55.0,  0.004,  0.14, 0.09),
        ("COMP-008", "Capteur temp DS18B20",           12.0,  0.015,  0.22, 0.18),
        ("COMP-009", "Module ESP32-WROOM",              6.0,  0.020,  0.25, 0.20),
        ("COMP-010", "Relais 12V 10A",                22.0,  0.003,  0.16, 0.11),
        ("COMP-011", "Afficheur LCD 16x2",              9.0,  0.008,  0.19, 0.14),
        ("COMP-012", "Convertisseur DC-DC 5A",         14.0,  0.012,  0.21, 0.16),
    ]
    for sku_id, name, base, trend, noise, monthly_amp in components:
        catalog.append(SKUConfig(
            sku_id=sku_id, name=name,
            category=ProductCategory.COMPOSANTS,
            profile=DemandProfile.FAST_MOVER,
            base_demand=base, trend_slope=trend,
            weekly_amplitude=0.05, monthly_amplitude=monthly_amp,
            weekly_phase=float(rng.uniform(0, 2*np.pi)),
            noise_sigma=noise,
            event_probability=0.025, event_amplitude=2.0,
            lead_time_mu=1.8, lead_time_sigma=0.35,
            unit_cost=float(rng.uniform(5, 150)),
            holding_cost_rate=0.002, shortage_cost_rate=0.018,
            ordering_cost=float(rng.uniform(20, 80)),
            initial_stock=int(base * 10),
        ))

    # ── Matières premières (12 SKUs) ──────────────────────────────────────────
    # Saisonnalité forte, tendance longue, demande intermittente possible
    raw_materials = [
        ("MAT-001", "Acier inox 316L (kg)",      500.0,  0.10,  0.20, 0.30),
        ("MAT-002", "Aluminium 6061 (kg)",        320.0,  0.05,  0.18, 0.25),
        ("MAT-003", "Cuivre électrolytique (kg)", 180.0, -0.03,  0.22, 0.28),
        ("MAT-004", "PVC granulés (kg)",          400.0,  0.08,  0.15, 0.35),
        ("MAT-005", "Résine époxy (L)",           150.0,  0.04,  0.25, 0.20),
        ("MAT-006", "Huile hydraulique (L)",      280.0, -0.01,  0.17, 0.22),
        ("MAT-007", "Peinture primaire (L)",      120.0,  0.06,  0.20, 0.30),
        ("MAT-008", "Caoutchouc naturel (kg)",    200.0,  0.03,  0.23, 0.27),
        ("MAT-009", "Acide sulfurique tech (L)",   90.0,  0.00,  0.19, 0.18),
        ("MAT-010", "Fibres carbone (m²)",         35.0,  0.15,  0.28, 0.15),
        ("MAT-011", "Titane grade 5 (kg)",         15.0,  0.12,  0.30, 0.12),
        ("MAT-012", "PTFE feuille (m²)",           25.0,  0.07,  0.24, 0.20),
    ]
    for sku_id, name, base, trend, noise, seasonal_amp in raw_materials:
        catalog.append(SKUConfig(
            sku_id=sku_id, name=name,
            category=ProductCategory.MATIERES,
            profile=DemandProfile.SEASONAL,
            base_demand=base, trend_slope=trend,
            weekly_amplitude=0.08, monthly_amplitude=seasonal_amp,
            weekly_phase=float(rng.uniform(0, 2*np.pi)),
            noise_sigma=noise,
            event_probability=0.015, event_amplitude=3.0,
            lead_time_mu=2.5, lead_time_sigma=0.5,
            unit_cost=float(rng.uniform(50, 2000)),
            holding_cost_rate=0.0015, shortage_cost_rate=0.020,
            ordering_cost=float(rng.uniform(100, 500)),
            initial_stock=int(base * 5),
        ))

    # ── Consommables (14 SKUs) ────────────────────────────────────────────────
    # Mix fast mover et slow mover, saisonnalité modérée
    consumables = [
        ("CON-001", "Gants nitrile (boîte 100)",   200.0, 0.05,  0.12, True),
        ("CON-002", "Masques FFP2 (boîte 20)",     150.0, 0.08,  0.15, True),
        ("CON-003", "Lubrifiant WD-40 500ml",       60.0, 0.02,  0.18, True),
        ("CON-004", "Papier abrasif P120 (lot)",    45.0, 0.01,  0.14, True),
        ("CON-005", "Électrodes soudure E6013",     80.0, 0.03,  0.16, True),
        ("CON-006", "Filtre à air G4 (lot 10)",     30.0, 0.04,  0.20, False),
        ("CON-007", "Joint torique NBR (lot 50)",   25.0, 0.02,  0.22, False),
        ("CON-008", "Vis inox M6x20 (lot 500)",     70.0, 0.01,  0.11, True),
        ("CON-009", "Écrous frein M8 (lot 200)",    55.0, 0.02,  0.13, True),
        ("CON-010", "Câble électrique 2.5mm²(m)",   40.0, 0.03,  0.17, True),
        ("CON-011", "Scotch isolant (lot 10)",      90.0, 0.01,  0.10, True),
        ("CON-012", "Solvant nettoyant IPA 1L",     35.0, 0.02,  0.19, False),
        ("CON-013", "Colle époxy bicomposant",      20.0, 0.04,  0.25, False),
        ("CON-014", "Rouleau papier thermique",     65.0, 0.00,  0.14, True),
    ]
    for sku_id, name, base, trend, noise, is_fast in consumables:
        profile = DemandProfile.FAST_MOVER if is_fast else DemandProfile.SLOW_MOVER
        catalog.append(SKUConfig(
            sku_id=sku_id, name=name,
            category=ProductCategory.CONSOMMABLES,
            profile=profile,
            base_demand=base, trend_slope=trend,
            weekly_amplitude=0.12, monthly_amplitude=0.15,
            weekly_phase=float(rng.uniform(0, 2*np.pi)),
            noise_sigma=noise,
            event_probability=0.02, event_amplitude=2.2,
            lead_time_mu=1.5, lead_time_sigma=0.3,
            unit_cost=float(rng.uniform(2, 80)),
            holding_cost_rate=0.003, shortage_cost_rate=0.012,
            ordering_cost=float(rng.uniform(10, 50)),
            initial_stock=int(base * 8),
            intermittency_prob=0.2 if not is_fast else 0.05,
        ))

    # ── Équipements (12 SKUs) — pièces de rechange industrielles ─────────────
    # Demande lumpy, intermittente — cas Croston classique
    equipment = [
        ("EQP-001", "Roulement SKF 6204",           8.0, 0.01, 0.35),
        ("EQP-002", "Courroie trapézoïdale A45",     5.0, 0.00, 0.40),
        ("EQP-003", "Joint spi 25x40x7",            12.0, 0.02, 0.30),
        ("EQP-004", "Pompe centrifuge 0.5kW",         2.0, 0.01, 0.50),
        ("EQP-005", "Vanne solénoïde 24V",            3.0, 0.02, 0.45),
        ("EQP-006", "Capteur inductif PNP",           6.0, 0.03, 0.38),
        ("EQP-007", "Motoréducteur 40tr/min",         1.5, 0.01, 0.55),
        ("EQP-008", "Contacteur LC1D25",              4.0, 0.02, 0.42),
        ("EQP-009", "Disjoncteur magnéto 10A",        7.0, 0.01, 0.36),
        ("EQP-010", "Variateur de fréquence 2.2kW",   1.0, 0.02, 0.60),
        ("EQP-011", "Encodeur incrémental 1000ppr",   3.5, 0.03, 0.48),
        ("EQP-012", "Pressostat 0-10bar",             2.5, 0.01, 0.52),
    ]
    for sku_id, name, base, trend, noise in equipment:
        catalog.append(SKUConfig(
            sku_id=sku_id, name=name,
            category=ProductCategory.EQUIPEMENTS,
            profile=DemandProfile.LUMPY,
            base_demand=base, trend_slope=trend,
            weekly_amplitude=0.02, monthly_amplitude=0.05,
            weekly_phase=float(rng.uniform(0, 2*np.pi)),
            noise_sigma=noise,
            event_probability=0.01, event_amplitude=5.0,
            lead_time_mu=3.0, lead_time_sigma=0.6,
            unit_cost=float(rng.uniform(50, 5000)),
            holding_cost_rate=0.001, shortage_cost_rate=0.030,
            ordering_cost=float(rng.uniform(50, 200)),
            initial_stock=int(base * 15),
            intermittency_prob=0.6,   # 60% des jours : demande nulle
        ))

    return catalog


# ── Core generation engine ────────────────────────────────────────────────────

class SyntheticDataGenerator:
    """
    Moteur de génération de données de demande industrielle synthétiques.

    Usage:
        gen = SyntheticDataGenerator(seed=42)
        dataset = gen.generate_all(n_days=730)
        df = gen.to_dataframe(dataset)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.catalog = build_sku_catalog()

    def generate_sku(
        self,
        config: SKUConfig,
        n_days: int = 730,
        start_date: str = "2023-01-01"
    ) -> GeneratedSKU:
        """
        Génère la série temporelle complète pour un SKU donné.

        La décomposition multiplicative est intentionnelle :
        demand = trend * seasonal * (1 + noise)
        → la variance absolue croît avec le niveau de la demande.
        """
        dates = pd.date_range(start=start_date, periods=n_days, freq='D')
        t = np.arange(n_days, dtype=float)

        # ── Composante tendance ───────────────────────────────────────────────
        trend = (
            config.base_demand
            + config.trend_slope * t
            + config.trend_quadratic * t**2
        )
        # Contrainte : la demande ne peut pas devenir négative
        trend = np.maximum(trend, 0.1)

        # ── Saisonnalité hebdomadaire (période 7) ─────────────────────────────
        # Fourier order 2 pour capturer l'asymétrie intra-semaine
        seasonal_weekly = (
            1
            + config.weekly_amplitude * np.sin(2 * np.pi * t / 7 + config.weekly_phase)
            + config.weekly_amplitude * 0.3 * np.sin(4 * np.pi * t / 7 + config.weekly_phase)
        )

        # ── Saisonnalité mensuelle (période 30.44) ────────────────────────────
        seasonal_monthly = (
            1
            + config.monthly_amplitude * np.sin(2 * np.pi * t / 30.44)
            + config.monthly_amplitude * 0.4 * np.sin(4 * np.pi * t / 30.44 + np.pi/4)
        )

        # ── Événements exogènes ───────────────────────────────────────────────
        # Bernoulli avec persistance : un événement dure 1-3 jours
        events = np.zeros(n_days)
        i = 0
        while i < n_days:
            if self.rng.random() < config.event_probability:
                duration = int(self.rng.integers(1, 4))
                amplitude = config.event_amplitude * self.rng.uniform(0.5, 1.5)
                end = min(i + duration, n_days)
                events[i:end] = amplitude - 1  # -1 car on multiplie par (1 + events)
                i += duration
            else:
                i += 1

        # ── Bruit multiplicatif ───────────────────────────────────────────────
        # epsilon ~ TruncNormal pour éviter les demandes négatives
        noise = self.rng.normal(0, config.noise_sigma, n_days)
        noise = np.clip(noise, -0.5, 2.0)  # Borne les outliers extrêmes

        # ── Assemblage multiplicatif ──────────────────────────────────────────
        demand = trend * seasonal_weekly * seasonal_monthly * (1 + events + noise)
        demand = np.maximum(demand, 0.0)

        # ── Profil intermittent (Croston) ─────────────────────────────────────
        # Pour lumpy/slow_mover : masquer aléatoirement des jours entiers
        if config.profile in (DemandProfile.LUMPY, DemandProfile.SLOW_MOVER):
            occurrence_mask = self.rng.random(n_days) < config.intermittency_prob
            demand = demand * occurrence_mask

        # Arrondi à l'entier — les unités sont des articles discrets
        demand = np.round(demand).astype(float)

        # ── Délais fournisseurs — Log-normale ─────────────────────────────────
        # Paramétrage : mu=2.0, sigma=0.4 → médiane ≈ 7.4 jours, P95 ≈ 14 jours
        n_orders = max(n_days // 7, 1)  # Estimation haute du nombre de commandes
        lead_times = self.rng.lognormal(
            config.lead_time_mu,
            config.lead_time_sigma,
            n_orders
        )
        lead_times = np.maximum(np.round(lead_times), 1).astype(int)

        return GeneratedSKU(
            config=config,
            dates=np.array(dates),
            demand=demand,
            lead_times=lead_times,
            events=(events > 0).astype(int),
            trend_component=trend,
            seasonal_weekly=seasonal_weekly,
            seasonal_monthly=seasonal_monthly,
            noise_component=noise,
        )

    def generate_all(
        self,
        n_days: int = 730,
        start_date: str = "2023-01-01"
    ) -> dict[str, GeneratedSKU]:
        """Génère l'ensemble du catalogue. Retourne un dict sku_id → GeneratedSKU."""
        dataset = {}
        for config in self.catalog:
            dataset[config.sku_id] = self.generate_sku(config, n_days, start_date)
        return dataset

    def to_dataframe(self, dataset: dict[str, GeneratedSKU]) -> pd.DataFrame:
        """
        Convertit le dataset en DataFrame long format.
        Format : (date, sku_id, demand, lead_time_mean, category, profile, ...)
        """
        records = []
        for sku_id, sku in dataset.items():
            for i, (date, d) in enumerate(zip(sku.dates, sku.demand)):
                records.append({
                    "date":            pd.Timestamp(date),
                    "sku_id":          sku_id,
                    "sku_name":        sku.config.name,
                    "category":        sku.config.category.value,
                    "profile":         sku.config.profile.value,
                    "demand":          d,
                    "event_flag":      int(sku.events[i]),
                    "trend":           sku.trend_component[i],
                    "seasonal_weekly": sku.seasonal_weekly[i],
                    "seasonal_monthly": sku.seasonal_monthly[i],
                })
        df = pd.DataFrame(records)
        df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
        return df

    def get_sku_dataframe(self, sku_id: str, dataset: dict[str, GeneratedSKU]) -> pd.DataFrame:
        """Retourne un DataFrame univarié pour un SKU — format attendu par les modèles."""
        sku = dataset[sku_id]
        df = pd.DataFrame({
            "date":    pd.to_datetime(sku.dates),
            "demand":  sku.demand,
            "event":   sku.events,
        })
        df = df.set_index("date")
        return df

    def save_dataset(self, dataset: dict[str, GeneratedSKU], output_dir: str) -> None:
        """Sauvegarde le dataset en CSV + métadonnées JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # DataFrame principal
        df = self.to_dataframe(dataset)
        df.to_csv(output_path / "demand_data.csv", index=False)

        # Métadonnées des SKUs
        metadata = {}
        for sku_id, sku in dataset.items():
            metadata[sku_id] = {
                "name":               sku.config.name,
                "category":           sku.config.category.value,
                "profile":            sku.config.profile.value,
                "base_demand":        sku.config.base_demand,
                "mean_realized":      round(sku.mean_demand, 3),
                "cv":                 round(sku.cv_demand, 3),
                "intermittency_rate": round(sku.intermittency_rate, 3),
                "n_days":             sku.n_days,
                "unit_cost":          sku.config.unit_cost,
                "lead_time_mu":       sku.config.lead_time_mu,
                "lead_time_sigma":    sku.config.lead_time_sigma,
            }

        with open(output_path / "sku_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_dataset(data_dir: str) -> tuple[pd.DataFrame, dict]:
        """Charge un dataset précédemment sauvegardé."""
        data_path = Path(data_dir)
        df = pd.read_csv(data_path / "demand_data.csv", parse_dates=["date"])
        with open(data_path / "sku_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
        return df, metadata
