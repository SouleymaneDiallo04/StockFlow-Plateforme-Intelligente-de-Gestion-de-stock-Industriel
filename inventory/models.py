"""
inventory/models.py
--------------------
Modèles Django pour StockFlow Intelligence.

Deux couches :
1. Gestion de stock opérationnelle : Category, Product (existants, enrichis)
2. Couche ML : MLJobResult, SKUMLProfile — stockent les résultats des analyses IA
"""
from __future__ import annotations

import uuid
from django.db import models
from django.contrib.auth.models import User


# ── Gestion de stock ──────────────────────────────────────────────────────────

class Category(models.Model):
    name        = models.CharField("Nom", max_length=120, unique=True)
    description = models.TextField("Description", blank=True)
    created_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering       = ['name']
        verbose_name   = "Catégorie"
        verbose_name_plural = "Catégories"

    def __str__(self):
        return self.name

    @property
    def product_count(self):
        return self.products.count()


class Product(models.Model):
    STATUS_ACTIVE   = 'active'
    STATUS_INACTIVE = 'inactive'
    STATUS_CHOICES  = [
        (STATUS_ACTIVE,   'Actif'),
        (STATUS_INACTIVE, 'Inactif'),
    ]

    reference        = models.CharField("Référence", max_length=50, unique=True, blank=True)
    name             = models.CharField("Nom du produit", max_length=150)
    description      = models.TextField("Description", blank=True)
    price            = models.DecimalField("Prix (DH)", max_digits=10, decimal_places=2)
    stock            = models.PositiveIntegerField("Quantité en stock", default=0)
    alert_threshold  = models.PositiveIntegerField("Seuil d'alerte stock", default=5)
    photo            = models.ImageField(upload_to='products/', blank=True, null=True)
    category         = models.ForeignKey(
        Category, on_delete=models.PROTECT,
        related_name='products', verbose_name="Catégorie"
    )
    status           = models.CharField(
        "Statut", max_length=10,
        choices=STATUS_CHOICES, default=STATUS_ACTIVE
    )
    # Lien vers le SKU synthétique du catalogue ML (optionnel)
    ml_sku_id        = models.CharField(
        "SKU ML associé", max_length=20, blank=True,
        help_text="Identifiant dans le catalogue ML synthétique (ex: COMP-001)"
    )
    created_at  = models.DateTimeField(auto_now_add=True)
    updated_at  = models.DateTimeField(auto_now=True)

    class Meta:
        ordering     = ['name']
        verbose_name = "Produit"
        verbose_name_plural = "Produits"

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.reference:
            self.reference = 'PRD-' + uuid.uuid4().hex[:8].upper()
        super().save(*args, **kwargs)

    @property
    def stock_status(self):
        if self.stock == 0:
            return 'out'
        elif self.stock <= self.alert_threshold:
            return 'low'
        return 'ok'

    @property
    def stock_badge_class(self):
        return {'out': 'danger', 'low': 'warning', 'ok': 'success'}.get(self.stock_status, 'secondary')

    @property
    def stock_label(self):
        return {'out': 'Rupture', 'low': 'Stock faible', 'ok': 'En stock'}.get(self.stock_status, '')


# ── Couche ML ─────────────────────────────────────────────────────────────────

class MLJobResult(models.Model):
    """
    Stocke l'état et le résultat de chaque tâche Celery ML.
    Pollé par le frontend via HTMX pour afficher la progression.
    """
    STATUS_PENDING = 'pending'
    STATUS_RUNNING = 'running'
    STATUS_SUCCESS = 'success'
    STATUS_FAILED  = 'failed'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'En attente'),
        (STATUS_RUNNING, 'En cours'),
        (STATUS_SUCCESS, 'Terminé'),
        (STATUS_FAILED,  'Échoué'),
    ]

    JOB_FORECAST   = 'forecast'
    JOB_OPTIMIZE   = 'optimize'
    JOB_SPC        = 'spc'
    JOB_GENERATE   = 'generate'
    JOB_ABC      = 'abc'
    JOB_TYPE_CHOICES = [
        (JOB_FORECAST, 'Prévision'),
        (JOB_OPTIMIZE, 'Optimisation'),
        (JOB_SPC,      'Contrôle SPC'),
        (JOB_GENERATE, 'Génération données'),
        ('abc', 'Analyse ABC-XYZ'),
    ]

    job_id      = models.CharField("ID tâche", max_length=64, unique=True, db_index=True)
    job_type    = models.CharField("Type", max_length=20, choices=JOB_TYPE_CHOICES)
    sku_id      = models.CharField("SKU", max_length=20, blank=True)
    status      = models.CharField("Statut", max_length=10, choices=STATUS_CHOICES, default=STATUS_PENDING)
    result_json = models.TextField("Résultat JSON", null=True, blank=True)
    error       = models.TextField("Erreur", blank=True)
    created_by  = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    created_at  = models.DateTimeField(auto_now_add=True)
    updated_at  = models.DateTimeField(auto_now=True)

    class Meta:
        ordering     = ['-created_at']
        verbose_name = "Tâche ML"
        verbose_name_plural = "Tâches ML"

    def __str__(self):
        return f"{self.job_type} | {self.sku_id} | {self.status}"

    @property
    def is_done(self):
        return self.status in (self.STATUS_SUCCESS, self.STATUS_FAILED)

    @property
    def result(self):
        import json
        if self.result_json:
            try:
                return json.loads(self.result_json)
            except Exception:
                return None
        return None


class SKUMLProfile(models.Model):
    """
    Cache des derniers résultats ML pour un SKU.
    Mis à jour à chaque tâche Celery réussie.
    Évite de re-parser les JSON à chaque requête dashboard.
    """
    sku_id = models.CharField("SKU ID", max_length=20, unique=True, db_index=True)

    # Forecasting
    best_forecast_model  = models.CharField("Meilleur modèle", max_length=20, blank=True)
    forecast_mase        = models.FloatField("MASE", null=True, blank=True)
    forecast_coverage    = models.FloatField("Coverage rate IC", null=True, blank=True)
    last_forecast_job    = models.ForeignKey(
        MLJobResult, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='forecast_profiles'
    )

    # Optimization
    reorder_point       = models.FloatField("Point de commande", null=True, blank=True)
    order_quantity      = models.FloatField("Quantité commandée", null=True, blank=True)
    safety_stock        = models.FloatField("Stock de sécurité", null=True, blank=True)
    service_level       = models.FloatField("Taux de service", null=True, blank=True)
    capital_reduction   = models.FloatField("Réduction capital (%)", null=True, blank=True)
    last_optimize_job   = models.ForeignKey(
        MLJobResult, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='optimize_profiles'
    )

    # SPC
    spc_status          = models.CharField("Statut SPC", max_length=20, blank=True)
    spc_n_critical      = models.IntegerField("Signaux critiques", default=0)
    last_spc_job        = models.ForeignKey(
        MLJobResult, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='spc_profiles'
    )

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering     = ['sku_id']
        verbose_name = "Profil ML SKU"

    def __str__(self):
        return f"ML Profile — {self.sku_id}"
