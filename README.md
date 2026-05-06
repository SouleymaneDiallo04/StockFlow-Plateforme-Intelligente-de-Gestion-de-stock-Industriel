# StockFlow Intelligence

**Système de gestion de stock industriel augmenté par l'intelligence artificielle**

---

## Synthèse

StockFlow Intelligence est une plateforme web conçue pour répondre à un problème documenté dans l'industrie manufacturière : les politiques de réapprovisionnement statiques — héritées des modèles Wilson des années 1910 — génèrent simultanément des excès de stock immobilisé et des ruptures évitables. La littérature estime entre 20 et 30% du capital en stock comme du gaspillage pur ; les ruptures représentent en moyenne 4% du chiffre d'affaires en pertes de production non récupérables.

L'application intègre trois modules d'analyse quantitative distincts, construits sur une architecture Django/Celery découplée du moteur ML, et visualisés via une interface SaaS moderne.

---

## Modules d'intelligence

### Module 1 — Prévision de demande avec incertitude quantifiée

Quatre modèles sont entraînés et comparés sur chaque SKU en walk-forward cross-validation, seule méthode valide pour les séries temporelles (le k-fold standard viole l'ordre causal).

**SARIMAX** — modèle autorégressif saisonnier avec variables exogènes. L'ordre `(p,d,q)(P,D,Q,s)` est sélectionné automatiquement via l'algorithme de Hyndman-Khandakar (critère AIC). Une transformation log stabilise la variance — cohérente avec le bruit multiplicatif du générateur de données.

**Prophet (Meta)** — décomposition additive avec saisonnalités customisées (hebdomadaire et mensuelle). Avantage principal : robustesse aux valeurs manquantes, fréquentes dans les données industrielles réelles, et estimation bayésienne via Stan produisant des intervalles mieux calibrés sur séries courtes.

**LightGBM** — gradient boosting sur features temporelles construites manuellement : lags (1, 2, 3, 7, 14, 21, 28), rolling statistics (mean/std/max sur 7, 14, 30 jours), encodage cyclique sin/cos des périodes. Trois modèles séparés (point, quantile bas, quantile haut) via régression quantile — plus honnête sur données non gaussiennes que les intervalles paramétriques.

**Temporal Fusion Transformer** — architecture Lim et al. (2021) via PyTorch Forecasting. Variable Selection Networks et multi-head attention fournissent une explicabilité native : le modèle expose les features et les pas de temps passés qui pilotent chaque prévision. Entraîné de façon asynchrone via Celery.

**Métriques d'évaluation retenues** :

- MASE (Mean Absolute Scaled Error) — métrique principale. Scale-free, comparable entre SKUs d'échelles différentes. MASE < 1 indique une supériorité sur le modèle naïf saisonnier.
- Coverage Rate — proportion des valeurs réelles contenues dans l'intervalle de confiance. Valide la calibration : un IC à 90% doit contenir ~90% des observations.
- Winkler Score — pénalise simultanément les IC trop larges et les non-couvertures. Standard des compétitions M5 et GEFCom.

Les intervalles de confiance sont ajustés par **conformal prediction** (méthode EnbPI, Xu & Xie 2021) — distribution-free, coverage garanti sans hypothèse gaussienne.

---

### Module 2 — Optimisation dynamique des politiques de réapprovisionnement

Ce module replace le calcul Wilson/EOQ (hypothèses : demande constante, délai certain, coûts fixes) par une approche simulation-optimisation qui respecte la réalité stochastique des processus industriels.

**Simulation Monte Carlo** — 10 000 scénarios par politique évaluée. Chaque scénario tire la demande journalière depuis une log-normale calibrée sur les paramètres du SKU, et les délais fournisseurs depuis une log-normale (distribution empiriquement validée pour les délais logistiques : non-négative, asymétrie droite). Les politiques (r,Q) et (s,S) sont supportées.

**Frontière de Pareto** — deux objectifs antagonistes sont optimisés simultanément :
- Minimiser le coût total (holding + ordering + shortage)
- Minimiser la probabilité de rupture sur l'horizon

La frontière de Pareto expose l'ensemble des politiques non-dominées. Le décideur choisit son point selon son appétit au risque, pas selon une formule fixe. Chaque point de la frontière expose : point de commande r, quantité Q, taux de service attendu, coût annuel estimé, stock de sécurité implicite.

**Lien Lean Six Sigma** — la politique optimale est comparée à la baseline Wilson naïve sur :
- Réduction du capital immobilisé en stock de sécurité (élimination du muda de surstock)
- Amélioration du taux de service (approche flux tiré)
- Économie annuelle totale en coûts de gestion de stock

Le taux de service à 99,73% correspond à la définition Three Sigma ; 99,99966% à Six Sigma — des points de référence intentionnellement ancrés dans la culture qualité industrielle.

---

### Module 3 — Contrôle Statistique des Processus (SPC/MSP)

Application des cartes de contrôle statistique à la surveillance de la demande et des ruptures, domaine où les ERP standards n'interviennent pas. Le SPC traite le stock comme un processus sous contrôle statistique, pas comme un compteur.

**Cartes implémentées** :

- Carte I-MR (Individus — Étendues Mobiles) : adaptée aux données continues non groupées, une observation par jour. L'estimation de σ depuis les étendues mobiles est moins sensible aux décalages de moyenne que l'écart-type global — propriété importante pour la Phase I.
- Carte p : taux de rupture sur fenêtre glissante de 30 jours. Surveille la proportion de jours en rupture plutôt que le niveau absolu.
- CUSUM (Cumulative Sum, Page 1954) : détecte des décalages de moyenne de l'ordre de 1 à 2σ que la carte Shewhart manque systématiquement. Paramétrage standard : k = 0,5σ, h = 5σ (ARL₀ ≈ 465).
- EWMA (Exponentially Weighted Moving Average) : λ = 0,2, sensible aux petites dérives progressives. Complémentaire au CUSUM, plus intuitif pour les praticiens.

**8 règles Western Electric** complètes (WECO 1956, Montgomery 2020) avec sévérité calibrée sur l'ARL₀ de chaque règle : les règles R1, R2, R3 (ARL₀ élevé) sont signalées comme critiques ; R4 à R8 comme alertes ou informations pour limiter les fausses alarmes.

Chaque signal déclenché est accompagné de son interprétation métier et d'une recommandation d'action — le rapport SPC n'est pas une liste de flags mais un outil de décision opérationnel.

---

## Données synthétiques — justification méthodologique

L'absence de données propriétaires industrielles est une contrainte fréquente en R&D appliquée. Le générateur de données est lui-même un livrable : il reproduit les propriétés statistiques documentées des séries de demande industrielle.

**Propriétés modélisées** :
- Bruit multiplicatif (variance croît avec le niveau de demande — Syntetos et al. 2005)
- Saisonnalité multiple par Fourier (hebdomadaire et mensuelle)
- Tendances linéaires et sub-linéaires par profil
- Événements exogènes avec persistance (promotions, pannes fournisseurs)
- Intermittence par modèle de Croston (pièces de rechange)
- Délais log-normaux (non-négatifs, queue droite asymétrique)
- 5 profils de demande : fast mover, slow mover, seasonal, trending, lumpy

**Validation statistique automatique** : tests ADF, KPSS, Ljung-Box, ACF saisonnière, vérification des propriétés distributionnelles. Le taux de validation est reporté dans le dashboard.

Ce choix transforme la contrainte en avantage : il est possible de tester la robustesse des modèles sous différents régimes de demande, ce qu'une seule source de données réelles ne permettrait pas.

---

## Architecture technique

```
stockflow/
├── gestion_stock/          Django project config + Celery
├── inventory/              App web (CRUD, vues ML, API polling)
├── ml/
│   ├── data/               Générateur synthétique + validateur statistique
│   ├── forecasting/        SARIMAX, Prophet, LightGBM, TFT, conformal prediction, évaluateur
│   ├── optimization/       Monte Carlo, frontière Pareto, politiques (r,Q)/(s,S)
│   ├── spc/                Cartes de contrôle, 8 règles Western Electric, rapport automatique
│   └── tasks.py            Tâches Celery asynchrones
├── templates/              Interface SaaS (dark/light, HTMX, Plotly)
├── static/                 CSS custom + JS (toasts, animations, autocomplete)
└── requirements.txt
```

**Stack** : Django 5, Celery + Redis, statsmodels, pmdarima, Prophet, LightGBM, PyTorch Forecasting, Plotly, HTMX, Bootstrap 5.

Le moteur ML est découplé du service web : les calculs longs (simulation Monte Carlo, entraînement TFT) s'exécutent dans des workers Celery séparés. Le frontend poll l'état via HTMX sans bloquer l'interface.

---

## Installation et démarrage

**Prérequis** : Python 3.10+, Redis (pour Celery)

```bash
# Cloner et installer
git clone <repo>
cd stockflow
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt

# Base de données et données de démonstration
python manage.py migrate
python manage.py seed_data

# Démarrer Redis (dans un terminal séparé)
redis-server

# Démarrer le worker Celery (dans un terminal séparé)
celery -A gestion_stock worker --loglevel=info

# Démarrer le serveur Django
python manage.py runserver
```

Accéder à `http://127.0.0.1:8000/`

**Comptes de test** :

| Identifiant | Mot de passe | Rôle       | Accès                                |
|-------------|--------------|------------|--------------------------------------|
| super1      | super1pass   | Superadmin | CRUD complet + Intelligence IA       |
| admin1      | admin1pass   | Admin      | CRUD produits + Intelligence IA      |
| user1       | user1pass    | Viewer     | Consultation + Intelligence IA       |

**Générer le dataset ML** : connecté en superadmin, aller dans Intelligence > Générer dataset. Cette étape est nécessaire avant de lancer les analyses ML.

**TFT (optionnel)** :
```bash
pip install pytorch-forecasting pytorch-lightning torch
```
Cocher la case TFT dans l'interface de prévision après installation.

---

## Limites et perspectives

**Limites documentées** :

- Les données synthétiques ne capturent pas les corrélations croisées entre SKUs (effets de substitution, demandes liées).
- La conformal prediction suppose l'échangeabilité des résidus, hypothèse approximative pour les séries temporelles. L'EnbPI atténue ce problème mais ne l'élimine pas.
- La simulation Monte Carlo suppose l'indépendance des demandes journalières — corrélation sérielle ignorée pour simplifier.
- L'optimisation de la frontière Pareto est effectuée sur grille (15×15 points) — une méthode d'optimisation continue (NSGA-II) fournirait une frontière plus dense.

**Extensions naturelles** :

- Intégration de données réelles via connecteur ERP (SAP, Odoo) ou import CSV
- Optimisation multi-SKU avec contraintes de budget et d'espace de stockage
- Modèle de simulation des délais fournisseurs corrélés (chaîne d'approvisionnement)
- Alertes automatiques par email/webhook sur signaux SPC critiques
- API REST pour intégration dans des systèmes existants

---

## Références

- Montgomery, D.C. (2020). *Introduction to Statistical Quality Control*, 8e éd. Wiley.
- Lim, B. et al. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*.
- Romano, Y., Patterson, E., Candès, E. (2019). Conformalized Quantile Regression. *NeurIPS*.
- Xu, C., Xie, Y. (2021). Conformal Prediction Interval for Dynamic Time-Series. *ICML*.
- Syntetos, A.A., Boylan, J.E. (2005). The accuracy of intermittent demand estimates. *International Journal of Forecasting*.
- Page, E.S. (1954). Continuous inspection schemes. *Biometrika*, 41(1–2).

---

*Projet réalisé dans le cadre d'une formation Ingénieur 4e année — Intelligence Artificielle et Data Science : Systèmes Industriels — ENSAM Meknès.*
