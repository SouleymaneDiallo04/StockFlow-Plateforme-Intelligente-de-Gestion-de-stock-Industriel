# 📦AI StockFlow 

<div align="center">

**Plateforme de Gestion de Stock Industriel Augmentée par l'Intelligence Artificielle**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Django](https://img.shields.io/badge/Django-5.x-092E20?style=flat-square&logo=django&logoColor=white)](https://djangoproject.com)
[![Celery](https://img.shields.io/badge/Celery-5.3-37814A?style=flat-square&logo=celery&logoColor=white)](https://docs.celeryq.dev)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![License](https://img.shields.io/badge/Licence-MIT-10b981?style=flat-square)](LICENSE)

*Projet réalisé par un étudiant ingénieur 4ème année — IA & Data Science · ENSAM Meknès*

</div>

---

## 🎯 Problème résolu

Les entreprises industrielles perdent **4% de leur chiffre d'affaires en ruptures de stock** tout en immobilisant **20 à 30% de leur capital en surstock inutile** — simultanément. Cause racine : des politiques de réapprovisionnement statiques héritées de la formule Wilson (1913), appliquées sans adaptation à une demande industrielle volatile, saisonnière et intermittente.

**StockFlow Intelligence** remplace ces seuils statiques par un moteur analytique continu combinant :

- 🔮 Prévision multi-modèles avec intervalles de confiance garantis par conformal prediction
- 🎯 Optimisation des politiques de stock par simulation Monte Carlo avec frontière de Pareto
- 📊 Contrôle Statistique des Processus (SPC) sur les flux de demande (8 règles WECO)
- 🏷️ Classification ABC-XYZ du portefeuille avec recommandations de politique par classe
- 🌍 Enrichissement par données externes (météo, calendrier industriel marocain)
- 🔬 Tracking MLflow de toutes les expériences de prévision et d'optimisation

> ⚠️ Ce projet n'est **pas** une application CRUD avec un dashboard. Chaque module est ancré dans une méthodologie publiée et peer-reviewed : conformal prediction (Romano et al., 2019), modèle de Croston, CUSUM de Page (1954), règles Western Electric (Montgomery, 2020).

---

## 🏗️ Architecture

```
stockflow/
├── gestion_stock/                  # Configuration Django + Celery
│   ├── settings.py                 # CELERY_BROKER, ML_DATA_DIR, MESSAGE_TAGS
│   ├── urls.py                     # Routing principal
│   └── celery.py                   # App Celery + autodiscover tasks
│
├── inventory/                      # Couche web — zéro import ML
│   ├── models.py                   # Product, Category, MLJobResult, SKUMLProfile
│   ├── views.py                    # CRUD + 10 vues ML + 2 endpoints API polling
│   ├── urls.py                     # 25+ routes dont /api/ml/job/<id>/status/
│   ├── forms.py                    # CategoryForm, ProductForm, RegisterForm
│   ├── permissions.py              # is_superadmin(), is_admin_or_superadmin()
│   └── context_processors.py      # user_role injecté dans tous les templates
│
└── ml/                             # Moteur ML — zéro dépendance Django
    ├── data/
    │   ├── generator.py            # Générateur synthétique paramétrique (50 SKUs, 5 profils)
    │   ├── validator.py            # Tests ADF, KPSS, Ljung-Box, ACF, distributions
    │   └── schemas.py              # Dataclasses typées pour toutes les structures ML
    │
    ├── forecasting/
    │   ├── base.py                 # BaseForecaster abstrait + walk-forward cross-validation
    │   ├── sarima.py               # SARIMAX avec sélection auto d'ordre (Hyndman-Khandakar)
    │   ├── prophet_model.py        # Meta Prophet avec saisonnalité mensuelle custom
    │   ├── lgbm_model.py           # LightGBM : 25+ features + régression quantile (3 modèles)
    │   ├── tft_model.py            # Temporal Fusion Transformer (PyTorch Forecasting)
    │   ├── conformal.py            # Calibration des IC par conformal prediction (EnbPI)
    │   └── evaluator.py            # MASE, sMAPE, Coverage Rate, Winkler Score
    │
    ├── optimization/
    │   ├── monte_carlo.py          # Simulation de 10 000 trajectoires d'inventaire
    │   ├── pareto.py               # Extraction frontière Pareto + analyse Lean
    │   └── policy.py              # Orchestrateur politiques (r,Q) et (s,S)
    │
    ├── spc/
    │   ├── control_charts.py       # I-MR, p-chart, CUSUM, EWMA
    │   ├── western_electric.py     # 8 règles WECO avec sévérité calibrée sur ARL₀
    │   └── report.py               # Générateur de rapport SPC automatique
    │
    ├── analysis/
    │   ├── abc.py                  # Classification ABC-XYZ (Pareto 80/15/5 + CV)
    │   ├── threshold_optimizer.py  # Mise à jour automatique de Product.alert_threshold
    │   ├── mlflow_tracker.py       # Tracking MLflow avec dégradation gracieuse
    │   └── external_data.py        # API Open-Meteo + calendrier marocain + features météo
    │
    └── tasks.py                    # 5 tâches Celery : generate, forecast, optimize, spc, abc
```

**Décision architecturale clé :** le moteur ML n'a aucun import Django. La couche web appelle le ML via des tâches Celery et lit les résultats depuis `MLJobResult` (JSON en base). Un data scientist peut exécuter n'importe quel module ML en script Python standalone. La couche web peut être remplacée sans toucher une seule ligne ML.

---

## 🚀 Fonctionnalités implémentées

### 🔐 Authentification & Contrôle d'accès par rôles

- **3 rôles** : Superadmin (CRUD complet + génération dataset), Admin (CRUD produits + tous les modules ML), Viewer (lecture seule + tous les modules ML)
- Décorateurs `@user_passes_test` sur toutes les vues sensibles
- Inscription avec choix de rôle, affectation automatique au groupe Django
- Context processor `user_role` injecté globalement : `{{ is_superadmin }}`, `{{ can_manage_products }}`
- Page login avec remplissage automatique des comptes de démonstration (un clic)

---

### 📦 Gestion opérationnelle du stock

- CRUD complet Produits et Catégories avec upload photo (Pillow)
- Référence unique auto-générée (`PRD-XXXXXXXX`, UUID tronqué)
- Seuil d'alerte configurable par produit, mis à jour automatiquement après optimisation ML
- Badges stock dynamiques : `En stock` / `Stock faible` / `Rupture` calculés depuis `stock` vs `alert_threshold`
- Champ `ml_sku_id` qui lie chaque produit opérationnel à son SKU dans le catalogue ML — boucle de feedback ML → opérations
- Export CSV avec BOM UTF-8 (compatible Excel), filtres courants préservés dans l'URL d'export
- Recherche/filtre/tri/pagination **HTMX** sans rechargement de page
- **Autocomplete** produits via endpoint JSON `/products/autocomplete/` (debounce 220ms)
- Double confirmation par checkbox avant toute suppression
- Suppression de catégorie bloquée si elle contient des produits — message explicite

---

### 📈 Module 1 — Prévision de demande

#### Modèles implémentés

| Modèle | Bibliothèque | Détail d'implémentation |
|--------|-------------|------------------------|
| **SARIMAX** | `pmdarima` | Sélection auto ordre `(p,d,q)(P,D,Q,s)` via Hyndman-Khandakar (critère AIC), transformation log1p, régresseurs exogènes |
| **Prophet** | `prophet` | Saisonnalité mensuelle custom (période 30.44j, Fourier order 5), mode multiplicatif, régresseurs événements |
| **LightGBM** | `lightgbm` | 25+ features temporelles, régression quantile (3 modèles séparés : point/bas/haut), prévision récursive multi-step |
| **TFT** | `pytorch-forecasting` | Variable Selection Networks, multi-head attention, sorties quantiles simultanées, interprétabilité native |

#### Feature Engineering (LightGBM)
```python
# Lags
lag_1, lag_2, lag_3, lag_7, lag_14, lag_21, lag_28

# Rolling statistics (shift=1 pour éviter la fuite de données)
roll_mean_7/14/30,  roll_std_7/14/30,  roll_max_7/14/30

# Encodage cyclique (sin/cos — évite les discontinuités entre périodes)
sin_dow, cos_dow        # jour de la semaine
sin_week, cos_week      # semaine de l'année
sin_month, cos_month    # mois
sin_dom, cos_dom        # jour du mois

# Calendrier industriel marocain
is_holiday, is_ramadan, is_end_of_month, is_start_of_month

# Événements exogènes
event_flag              # pic de demande, rupture fournisseur
```

#### Protocole d'évaluation
- **Walk-forward cross-validation** (3 folds) — seule méthode valide pour les séries temporelles. Le k-fold standard viole l'ordre causal.
- **MASE** comme métrique principale : scale-free, comparable entre SKUs d'échelles différentes. MASE < 1 = meilleur que le modèle naïf saisonnier.
- **Coverage Rate** : validation empirique de la calibration des IC. Un IC à 90% doit contenir ~90% des valeurs réalisées.
- **Winkler Score** : pénalise simultanément les IC trop larges et les non-couvertures (standard M5 et GEFCom).
- **sMAPE** : robuste aux valeurs proches de zéro (fréquentes en demande industrielle).

#### Conformal Prediction (EnbPI)
- Calibration des intervalles **sans hypothèse distributionnelle** (Romano et al., 2019 + Xu & Xie 2021)
- Applicable à tous les modèles de base (SARIMAX, Prophet, LightGBM, TFT)
- Résidus de calibration sur fenêtre glissante récente uniquement (adaptation EnbPI à la non-échangeabilité temporelle)
- Coverage garanti au niveau nominal sur l'ensemble de calibration

---

### 🎯 Module 2 — Optimisation des politiques de réapprovisionnement

#### Simulation Monte Carlo
- **10 000 scénarios** par politique candidate
- Demande journalière tirée depuis une log-normale calibrée sur les paramètres historiques du SKU
- Délais fournisseurs tirés depuis une log-normale (μ=2.0, σ=0.4 → médiane ≈ 7.4j, P95 ≈ 14j) — distribution empiriquement validée pour les délais logistiques
- Simulation complète de l'inventaire : réception commandes, satisfaction demande, tracking ruptures, comptabilisation coûts
- Modèle de coût : `coût_stockage × stock_moyen + coût_commande × nb_commandes + coût_rupture × demande_non_servie`
- **Asymétrie shortage/holding** : coût de rupture = 7.5× coût de stockage (ratio documenté en supply chain)
- Politiques **(r,Q)** et **(s,S)** supportées

#### Frontière de Pareto
- Scan sur grille 15×15 dans l'espace `(point_de_commande, quantité_commandée)`
- **Deux objectifs antagonistes** : coût total annuel ↓ et probabilité de rupture ↓
- Extraction des solutions non-dominées (optimalité Pareto stricte)
- Visualisation Plotly interactive avec annotation du point optimal
- Sélection du point optimal au taux de service cible configurable (80–99%)

#### Analyse Lean Six Sigma
- Réduction du capital immobilisé vs baseline Wilson EOQ (muda de surstock quantifié en DH)
- Économie annuelle totale (DH/an)
- Amélioration du taux de service (points de pourcentage)
- Points de référence Three Sigma (99.73%) et Six Sigma (99.9997%)
- **Mise à jour automatique** de `Product.alert_threshold` en base Django après chaque optimisation réussie

---

### 📊 Module 3 — Contrôle Statistique des Processus (SPC/MSP)

#### Cartes de contrôle

| Carte | Méthode | Ce qu'elle détecte |
|-------|---------|-------------------|
| **I-MR** | σ estimé depuis les étendues mobiles (facteur d2=1.128) | Anomalies ponctuelles, dérive de la dispersion |
| **p-chart** | Proportion ruptures sur fenêtre glissante 30j | Dérive de la qualité de service |
| **CUSUM** | Page (1954), k=0.5σ, h=5σ, ARL₀≈465 | Décalages de moyenne 1–2σ manqués par Shewhart |
| **EWMA** | λ=0.2, L=3.0 | Dérives progressives, complémentaire au CUSUM |

#### 8 Règles Western Electric (toutes implémentées)

| Règle | Description | Sévérité | ARL₀ |
|-------|------------|----------|------|
| R1 | 1 point au-delà de ±3σ | 🔴 Critique | ~370 |
| R2 | 9 pts consécutifs même côté de la CL | 🔴 Critique | ~250 |
| R3 | 6 pts consécutifs en tendance monotone | 🔴 Critique | ~200 |
| R4 | 14 pts alternant haut/bas | 🟡 Alerte | ~50 |
| R5 | 2 des 3 derniers pts en Zone A | 🟡 Alerte | ~90 |
| R6 | 4 des 5 derniers pts en Zone B+ | 🟡 Alerte | ~56 |
| R7 | 15 pts consécutifs en Zone C | 🔵 Info | ~33 |
| R8 | 8 pts consécutifs hors Zone C | 🔵 Info | ~50 |

- Chaque signal déclenché inclut son **interprétation métier** + **action corrective recommandée**
- Zones A/B/C affichées en bandes colorées dans Plotly
- Points hors contrôle marqués avec symbole ❌, points d'alerte en ambre
- Rapport SPC automatique avec statut global : `in_control` / `warning` / `out_of_control`

---

### 🏷️ Module 4 — Analyse ABC-XYZ

- **ABC** par valeur consommée annuelle (Pareto 80/15/5)
- **XYZ** par coefficient de variation (X: CV<0.25, Y: 0.25–0.75, Z: >0.75)
- **9 classes** de la matrice avec politique de gestion spécifique par classe :

| Classe | Politique recommandée |
|--------|----------------------|
| **AX** | Flux tiré. Stock minimal. Approvisionnement JIT. |
| **AY** | Stock de sécurité modéré. Révision fréquente des paramètres. |
| **AZ** | Gestion sur mesure. Commandes fréquentes. Approvisionnement d'urgence. |
| **BX/BY/BZ** | Gestion standard. EOQ applicable. |
| **CX/CY** | Consolider les commandes. Évaluer la nécessité de la référence. |
| **CZ** | Candidat à l'élimination ou au make-on-order. CoQ > valeur. |

- Courbe de Pareto interactive (barres + % cumulé double axe) en Plotly
- Matrice ABC-XYZ en barplot coloré
- Tableau de classement complet filtrable par classe
- Résumé portefeuille : nombre de SKUs, part de valeur, répartition XYZ par classe

---

### 🌍 Données externes

- **Open-Meteo API** (gratuite, sans clé) : météo historique pour 5 villes industrielles marocaines (Casablanca, Rabat, Meknès, Fès, Tanger)
- Température max/min, précipitations, indicateurs `is_extreme_heat`, `is_cold`, `is_rainy`
- **Calendrier marocain** : jours fériés officiels, périodes de Ramadan 2023–2024, effets fin/début de mois
- Encodage cyclique sin/cos de toutes les features calendaires pour éviter les discontinuités de frontière
- Fusion automatique avec les séries de demande pour enrichir les régresseurs SARIMAX et Prophet
- **Dégradation gracieuse** : données météo synthétiques réalistes générées si l'API est indisponible

---

### 🔬 MLflow Experiment Tracking

- Enregistre chaque run de prévision : modèle, SKU, profil, hyperparamètres, MASE, Coverage, Winkler, temps d'entraînement
- Enregistre chaque run d'optimisation : type de politique, point de commande, quantité, taux de service, coût annuel, réduction de capital
- `get_best_model(sku_id)` récupère le meilleur modèle historique pour un SKU donné
- **Dégradation gracieuse complète** : si MLflow n'est pas installé, toutes les méthodes sont no-ops. Aucune exception levée.

---

### 🗄️ Générateur de données synthétiques

**50 SKUs × 730 jours** avec 5 profils de demande :

| Profil | Catégorie (SKUs) | Caractéristiques |
|--------|----------------|------------------|
| `fast_mover` | Composants électroniques (12) | Demande continue, faible intermittence |
| `seasonal` | Matières premières (12) | Forte saisonnalité mensuelle, tendances longues |
| `fast/slow_mover` | Consommables (14) | Mix, variabilité modérée |
| `lumpy` | Équipements/Pièces de rechange (12) | Intermittence 60%, modèle de Croston |

**Modèle mathématique (bruit multiplicatif — variance croît avec le niveau) :**
```
demand(t) = trend(t) × seasonal_weekly(t) × seasonal_monthly(t) × (1 + events(t) + noise(t))

trend(t)              = base + β·t + γ·t²
seasonal_weekly(t)    = 1 + A·sin(2πt/7 + φ) + 0.3A·sin(4πt/7 + φ)   [Fourier order 2]
seasonal_monthly(t)   = 1 + B·sin(2πt/30.44) + 0.4B·sin(4πt/30.44 + π/4)
noise(t)              ~ TruncNormal(0, σ_rel)    ← bruit MULTIPLICATIF
events(t)             ~ Bernoulli(p_event) avec persistance 1–3 jours
lead_time             ~ LogNormal(μ_lt, σ_lt)    ← délais non-négatifs, queue droite
```

**Validation statistique automatique :**
- Tests ADF + KPSS (hypothèses complémentaires sur la stationnarité)
- Test de Ljung-Box aux lags 7, 14, 30
- Vérification du pic ACF saisonnier
- Vérification CV, asymétrie, taux d'intermittence
- Récupération des paramètres log-normaux des délais

---

### ⚡ Architecture asynchrone (Celery + Redis)

| Tâche | Nom Celery | Limite | Output |
|-------|-----------|--------|--------|
| Génération dataset | `ml.generate_dataset` | 3 min | CSV + métadonnées JSON |
| Prévision multi-modèles | `ml.forecast_sku` | 10 min | Tableau comparatif + toutes prévisions |
| Optimisation politique | `ml.optimize_policy` | 6 min | Frontière Pareto + analyse Lean |
| Rapport SPC | `ml.spc_report` | 2 min | 4 cartes + signaux + recommandations |
| Analyse ABC-XYZ | `ml.abc_analysis` | 2 min | Classement complet + données Pareto |

HTMX interroge `/api/ml/job/<id>/status/` toutes les 2 secondes. Résultats stockés en JSON dans `MLJobResult` (SQLite), agrégés vers `SKUMLProfile` pour l'affichage dashboard.

---

## 🖥️ Interface utilisateur

- **Thème sombre/clair** avec toggle, persisté en `localStorage`
- Sidebar fixe avec navigation conditionnelle selon le rôle
- **Toast notifications** auto-dismiss (4.5s) pour tous les messages Django
- **Compteurs animés** sur les KPIs du dashboard (ease-out cubique)
- **Plotly** pour toutes les visualisations ML : prévisions avec bandes IC, frontière Pareto annotée, cartes de contrôle avec zones A/B/C, courbe Pareto ABC
- **Chart.js** pour le dashboard opérationnel (doughnut répartition catégories)
- **HTMX** pour la recherche/filtre/tri/pagination produits sans rechargement
- Autocomplete produits avec debounce 220ms
- Design responsive — sidebar collapsable sur mobile

---

## 🔧 Stack technique — Justifications des choix

| Technologie | Version | Pourquoi ce choix, pas un autre |
|-------------|---------|--------------------------------|
| **Django** | 5.x | ORM, admin, auth, migrations intégrés. FastAPI aurait nécessité de tout reconstruire pour un gain marginal. |
| **Celery + Redis** | 5.3 / 7 | Workers asynchrones vrais avec retry, time limits, routing. Django-Q est plus simple mais moins crédible en production. Les threads background bloqueraient le serveur WSGI. |
| **pmdarima** | 2.x | `auto_arima` avec algorithme Hyndman-Khandakar stepwise. La recherche manuelle sur grille complète est O(n⁵) pour un résultat équivalent. |
| **Prophet** | 1.1 | Meilleur pour les données manquantes, les ruptures structurelles et les effets de jours fériés. statsmodels SARIMA nécessite une série complète. |
| **LightGBM** | 4.x | Plus rapide qu'XGBoost sur features tabulaires, régression quantile native, parallélisme `n_jobs=-1`. Les alternatives neurales (N-BEATS) sont moins interprétables. |
| **PyTorch Forecasting** | 1.x | Seule bibliothèque avec implémentation TFT de production incluant Variable Selection Networks et visualisation d'attention. |
| **Conformal Prediction** | custom | Sans distribution. Toutes les alternatives paramétriques (bayésien, bootstrap) requièrent des hypothèses qui échouent sur les résidus industriels. |
| **NumPy/SciPy** | — | Monte Carlo 10 000 scénarios nécessite des ops vectorisées. pandas serait 5× plus lent dans la boucle interne de simulation. |
| **MLflow** | 2.x | Standard en ML engineering. Comparaison d'expériences, logging de paramètres, registre de modèles. Weights & Biases nécessite un compte externe. |
| **HTMX** | 1.9 | Mises à jour partielles de page sans framework JS. React/Vue aurait triplé la complexité frontend sans gain dans ce cas d'usage. |
| **Plotly** | 2.27 | Zoom interactif, hover, annotations sur les cartes de contrôle et les frontières de Pareto. Matplotlib est statique. Chart.js manque de la granularité nécessaire. |
| **SQLite** | — | Suffisant pour la démonstration. Le passage à PostgreSQL se fait en modifiant uniquement `DATABASES` dans `settings.py`. |

---

## ⚙️ Installation

```bash
# 1. Cloner et environnement
git clone https://github.com/VOTRE_USERNAME/stockflow-intelligence.git
cd stockflow-intelligence
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate

# 2. Dépendances
pip install -r requirements.txt

# 3. Base de données
python manage.py migrate
python manage.py seed_data         # Crée groupes + comptes de test + 10 produits démo

# 4. Services (terminaux séparés)
redis-server
celery -A gestion_stock worker --loglevel=info

# 5. Lancer
python manage.py runserver
```

### Optionnel

```bash
# Modèle TFT
pip install pytorch-forecasting pytorch-lightning torch

# Interface MLflow
pip install mlflow
mlflow ui                          # → http://localhost:5000
```

**Première utilisation ML :** se connecter en `super1` → Intelligence → Générer dataset (730 jours) → tous les modules ML deviennent opérationnels.

---

## 👤 Comptes de test

| Utilisateur | Mot de passe | Rôle | Permissions |
|------------|-------------|------|-------------|
| `super1` | `super1pass` | Superadmin | CRUD complet + tous modules ML + génération dataset |
| `admin1` | `admin1pass` | Admin | CRUD produits + tous modules ML |
| `user1` | `user1pass` | Viewer | Lecture seule + tous modules ML |

---

## 📐 Limitations méthodologiques documentées

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| Données synthétiques — pas de corrélations inter-SKUs | Effets de substitution non capturés | Le générateur accepte un CSV réel en remplacement |
| Violation de l'échangeabilité pour EnbPI | Couverture garantie approximative | Calibration sur fenêtre glissante récente réduit le biais |
| Frontière Pareto sur grille 15×15 | Densité de la frontière limitée | NSGA-II améliorerait ; la grille est suffisante en démonstration |
| Monte Carlo avec demandes journalières indépendantes | Sous-estime le risque en cas de corrélation sérielle | Acceptable pour les SKUs à demande i.i.d. ; documenté |

---

## 📚 Références

- Montgomery, D.C. (2020). *Introduction to Statistical Quality Control*, 8e éd. Wiley.
- Lim et al. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *IJF*, 37(4).
- Romano, Y., Patterson, E., Candès, E. (2019). Conformalized Quantile Regression. *NeurIPS*, 32.
- Xu, C., Xie, Y. (2021). Conformal Prediction Interval for Dynamic Time-Series. *ICML*.
- Syntetos, A.A., Boylan, J.E. (2005). The accuracy of intermittent demand estimates. *IJF*, 21(2).
- Page, E.S. (1954). Continuous inspection schemes. *Biometrika*, 41(1–2).
- Chopra, S., Meindl, P. (2016). *Supply Chain Management*, 6e éd. Pearson.

---

## 🎓 À propos

**Auteur :** Étudiant ingénieur 4ème année — IA & Data Science : Systèmes Industriels  
**Établissement :** ENSAM Meknès — Génie Industriel (Lean Six Sigma, MSP, Démarche Qualité)  

Ce projet démontre que l'écart entre la méthodologie ML académique et les outils opérationnels industriels peut être comblé par un seul ingénieur avec suffisamment de connaissance du domaine et deux mois de travail. La formation Lean Six Sigma n'est pas décorative — elle structure chaque choix de modélisation, du ratio asymétrique rupture/stockage aux points de référence Three Sigma et Six Sigma.

---

<div align="center">

**Si ce projet vous est utile, une ⭐ sur GitHub est appréciée.**

</div>
