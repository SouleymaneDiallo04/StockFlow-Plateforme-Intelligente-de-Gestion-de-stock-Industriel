# 📦 StockFlow

<div align="center">

**AI-Powered Industrial Inventory Management Platform**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Django](https://img.shields.io/badge/Django-5.x-092E20?style=flat-square&logo=django&logoColor=white)](https://djangoproject.com)
[![Celery](https://img.shields.io/badge/Celery-5.3-37814A?style=flat-square&logo=celery&logoColor=white)](https://docs.celeryq.dev)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-10b981?style=flat-square)](LICENSE)

*Built by a 4th-year AI & Data Science Engineering student — ENSAM Meknès*

</div>

---

## 🎯 What & Why

Industrial companies lose **4% of revenue to stockouts** while simultaneously immobilizing **20–30% of working capital in excess safety stock**. The root cause: static replenishment policies based on Wilson's EOQ (1913), applied unchanged to volatile, seasonal, intermittent industrial demand.

**StockFlow Intelligence** replaces static thresholds with a continuously-updated analytical engine combining:
- Multi-model demand forecasting with statistically guaranteed prediction intervals
- Monte Carlo simulation-based policy optimization with Pareto frontier visualization
- Statistical Process Control on demand streams (8 WECO rules, I-MR, CUSUM, EWMA)
- ABC-XYZ portfolio classification with automated management policy recommendations

> 💡 This is not a CRUD app with a chart. Every module is grounded in peer-reviewed methodology — conformal prediction, Croston's model, Page's CUSUM, Montgomery's SPC — and implemented from scratch.

---

## 🏗️ Architecture

```
stockflow/
├── gestion_stock/              # Django project config + Celery setup
│   ├── settings.py
│   ├── urls.py
│   └── celery.py
│
├── inventory/                  # Web layer — fully decoupled from ML
│   ├── models.py               # Product, Category, MLJobResult, SKUMLProfile
│   ├── views.py                # CRUD + ML API polling endpoints
│   ├── urls.py                 # 20+ routes including ML API
│   ├── forms.py
│   ├── permissions.py          # Role-based access (superadmin / admin / viewer)
│   └── context_processors.py
│
└── ml/                         # ML engine — zero Django dependency
    ├── data/
    │   ├── generator.py        # Parametric synthetic data generator (50 SKUs, 5 profiles)
    │   ├── validator.py        # ADF, KPSS, Ljung-Box, ACF statistical validation
    │   └── schemas.py          # Typed dataclasses for all ML data structures
    │
    ├── forecasting/
    │   ├── base.py             # Abstract BaseForecaster + walk-forward cross-validation
    │   ├── sarima.py           # SARIMAX with auto order selection (Hyndman-Khandakar)
    │   ├── prophet_model.py    # Meta Prophet with custom monthly seasonality
    │   ├── lgbm_model.py       # LightGBM with 25+ temporal features + quantile regression
    │   ├── tft_model.py        # Temporal Fusion Transformer (PyTorch Forecasting)
    │   ├── conformal.py        # Conformal prediction interval calibration (EnbPI)
    │   └── evaluator.py        # MASE, sMAPE, Coverage Rate, Winkler Score
    │
    ├── optimization/
    │   ├── monte_carlo.py      # 10,000-scenario inventory trajectory simulation
    │   ├── pareto.py           # Pareto frontier extraction + Lean analysis
    │   └── policy.py           # (r,Q) and (s,S) policy orchestrator
    │
    ├── spc/
    │   ├── control_charts.py   # I-MR, p-chart, CUSUM, EWMA
    │   ├── western_electric.py # All 8 WECO rules with ARL₀-calibrated severity
    │   └── report.py           # Automated SPC report generator
    │
    ├── analysis/
    │   ├── abc.py              # ABC-XYZ classification (Pareto 80/15/5 + CV)
    │   ├── threshold_optimizer.py  # Auto-update Product.alert_threshold post-optimization
    │   ├── mlflow_tracker.py   # MLflow experiment tracking (graceful degradation)
    │   └── external_data.py    # Weather API (Open-Meteo) + Moroccan calendar features
    │
    └── tasks.py                # 5 Celery tasks: generate, forecast, optimize, spc, abc
```

**Key architectural decision:** ML engine has zero Django imports. The web layer calls ML through Celery tasks and reads results from `MLJobResult` (JSON in SQLite). A data scientist can run any ML module as a standalone Python script. The web layer can be replaced without touching a single ML file.

---

## 🚀 Features Implemented

### 🔐 Authentication & Role-Based Access Control
- 3 roles: **Superadmin** (full CRUD + dataset generation), **Admin** (product CRUD + all ML modules), **Viewer** (read-only + all ML modules)
- Django Groups + `@user_passes_test` decorators on all sensitive views
- Registration with role selection, auto-assigned Django group

### 📦 Operational Stock Management
- Full CRUD on Products and Categories with image upload (Pillow)
- Auto-generated unique reference (UUID-based, `PRD-XXXXXXXX`)
- Per-product configurable alert threshold (updated automatically by ML optimization)
- Stock badges: `En stock` / `Stock faible` / `Rupture` computed from `stock` vs `alert_threshold`
- Link between operational `Product` and ML `SKU ID` for ML-to-operations feedback loop
- CSV export with UTF-8 BOM (Excel-compatible), filters preserved in export
- HTMX-powered search/filter/sort without full page reload
- Product autocomplete via `/products/autocomplete/` JSON endpoint (debounced, 220ms)
- Confirmation double-check (checkbox) before any destructive action
- Category deletion blocked if products exist — explicit error message

### 📈 Module 1 — Demand Forecasting

#### Models
| Model | Library | Key Implementation Detail |
|---|---|---|
| SARIMAX | `pmdarima` | Auto order selection via Hyndman-Khandakar (AIC), log1p transform, exogenous regressors |
| Prophet | `prophet` | Custom 30.44-day monthly seasonality, multiplicative mode, event regressors |
| LightGBM | `lightgbm` | 25+ features, quantile regression (3 separate models), recursive multi-step |
| TFT | `pytorch-forecasting` | Variable Selection Networks, multi-head attention, simultaneous quantile output |

#### Feature Engineering (LightGBM)
- Lags: `t-1, t-2, t-3, t-7, t-14, t-21, t-28`
- Rolling: mean/std/max over 7, 14, 30 days
- Cyclical encoding: `sin/cos` of day-of-week, week-of-year, month, day-of-month
- Binary: `is_monday`, `is_friday`, `is_weekend`, `is_month_start`, `is_month_end`
- External: `event_flag`, `is_holiday`, `is_ramadan`, `week_sin/cos`

#### Evaluation Protocol
- **Walk-forward cross-validation** (3 folds) — the only valid method for time series
- **MASE** as primary metric (scale-free, interpretable vs naive seasonal benchmark)
- **Coverage Rate** — empirical calibration check of prediction intervals
- **Winkler Score** — simultaneous penalty on width and coverage failures (M5 standard)
- **sMAPE** — symmetric, robust to near-zero values

#### Conformal Prediction (EnbPI)
- Distribution-free interval calibration (Romano et al., 2019 + Xu & Xie 2021)
- No Gaussian assumption required
- Calibration residuals from recent window only (EnbPI adaptation for time series exchangeability)
- Coverage guaranteed at nominal level on calibration set

### 🎯 Module 2 — Policy Optimization

#### Monte Carlo Simulation
- **10,000 scenarios** per policy candidate
- Demand sampled from log-normal (calibrated on historical μ, σ)
- Lead times from log-normal (μ=2.0, σ=0.4 → median ≈ 7.4 days, P95 ≈ 14 days)
- Full inventory simulation: daily demand, pending orders, stockout tracking
- Cost model: `holding_cost × stock + ordering_cost × n_orders + shortage_cost × unmet_demand`
- Shortage cost asymmetry: 7.5× holding cost (empirically documented in supply chain literature)

#### Pareto Frontier
- 15×15 grid search over `(reorder_point, order_quantity)` space
- Two antagonistic objectives: total cost ↓ and stockout probability ↓
- Non-dominated solution extraction (true Pareto optimality)
- Interactive Plotly scatter — hover shows full policy parameters
- Optimal point selection at target service level (configurable 80–99%)

#### Lean Six Sigma Analysis
- Capital reduction vs Wilson EOQ baseline (muda de surstock quantified)
- Annual cost savings (DH)
- Service level improvement (percentage points)
- Three Sigma (99.73%) and Six Sigma (99.9997%) reference points
- Automatic update of `Product.alert_threshold` post-optimization

### 📊 Module 3 — Statistical Process Control

#### Control Charts
| Chart | Method | Detects |
|---|---|---|
| I-MR | σ from moving ranges (d2 factor) | Point anomalies, process spread |
| p-chart | 30-day rolling stockout proportion | Service quality drift |
| CUSUM | Page (1954), k=0.5σ, h=5σ | Small mean shifts (1–2σ) missed by Shewhart |
| EWMA | λ=0.2, L=3.0 | Progressive drift, complementary to CUSUM |

#### 8 Western Electric Rules (all implemented)
| Rule | Description | Severity | ARL₀ |
|---|---|---|---|
| R1 | 1 point beyond ±3σ | 🔴 Critical | ~370 |
| R2 | 9 consecutive points same side | 🔴 Critical | ~250 |
| R3 | 6 consecutive monotone trend | 🔴 Critical | ~200 |
| R4 | 14 alternating up/down | 🟡 Warning | ~50 |
| R5 | 2/3 last points in Zone A | 🟡 Warning | ~90 |
| R6 | 4/5 last points in Zone B+ | 🟡 Warning | ~56 |
| R7 | 15 consecutive in Zone C | 🔵 Info | ~33 |
| R8 | 8 consecutive outside Zone C | 🔵 Info | ~50 |

- Every signal includes business-language interpretation + corrective action recommendation
- Control limit zones (A/B/C) rendered as shaded bands in Plotly
- Out-of-control points marked with ❌ symbol, warning points in amber

### 📊 Module 4 — ABC-XYZ Portfolio Classification
- **ABC** by annual consumed value (Pareto 80/15/5)
- **XYZ** by coefficient of variation (X: CV<0.25, Y: 0.25–0.75, Z: >0.75)
- 9-class matrix with specific replenishment policy per class (AX → JIT, CZ → make-on-order)
- Interactive Pareto curve (bars + cumulative % dual-axis)
- Filterable ranking table by class
- Portfolio summary: SKU count, value share, XYZ breakdown per class

### 🌍 External Data Integration
- **Open-Meteo API** (free, no API key): historical weather for 5 Moroccan cities
- Calendar features: Moroccan public holidays, Ramadan periods, end-of-month effects
- Cyclical encoding of all calendar features (sin/cos) to avoid boundary discontinuities
- Graceful fallback to synthetic weather if API unavailable

### 🔬 MLflow Experiment Tracking
- Tracks every forecasting run: model name, SKU, hyperparameters, MASE, Coverage, Winkler, training time
- Tracks every optimization run: policy parameters, service level, annual cost, capital reduction
- `get_best_model(sku_id)` retrieves best historical model for any SKU
- Full graceful degradation: zero breaking changes if MLflow not installed

### 🗄️ Synthetic Data Generator
50 SKUs × 730 days with 5 demand profiles:

| Profile | SKUs | Characteristics |
|---|---|---|
| `fast_mover` | Composants (12) | Stable, continuous, low intermittency |
| `seasonal` | Matières (12) | Strong monthly seasonality, long trends |
| `fast/slow_mover` | Consommables (14) | Mixed, moderate variability |
| `lumpy` | Équipements (12) | Intermittency 60%, Croston model, spare parts |

**Mathematical model:**
```
demand(t) = trend(t) × seasonal_weekly(t) × seasonal_monthly(t) × (1 + events(t) + noise(t))

trend(t)            = base + β·t + γ·t²
seasonal_weekly(t)  = 1 + A·sin(2πt/7 + φ) + 0.3A·sin(4πt/7 + φ)
seasonal_monthly(t) = 1 + B·sin(2πt/30.44) + 0.4B·sin(4πt/30.44 + π/4)
noise(t)            = multiplicative, ~ TruncNormal(0, σ_rel)    ← variance grows with level
lead_time           ~ LogNormal(μ_lt, σ_lt)                      ← right-skewed delays
```

**Statistical validation (automatic):**
- ADF + KPSS stationarity tests (complementary hypotheses)
- Ljung-Box autocorrelation test at lags 7, 14, 30
- ACF seasonal peak verification
- CV, skewness, intermittency rate checks
- Log-normal parameter recovery for lead times

### ⚡ Async Task Architecture
| Task | Celery name | Time limit | Output |
|---|---|---|---|
| Dataset generation | `ml.generate_dataset` | 3 min | CSV + JSON metadata |
| Multi-model forecast | `ml.forecast_sku` | 10 min | Comparison table + all forecasts |
| Policy optimization | `ml.optimize_policy` | 6 min | Pareto frontier + Lean analysis |
| SPC report | `ml.spc_report` | 2 min | 4 charts + signals + recommendations |
| ABC-XYZ analysis | `ml.abc_analysis` | 2 min | Full ranking + Pareto data |

HTMX polls `/api/ml/job/<id>/status/` every 2s. No JavaScript framework needed. Job results stored as JSON in `MLJobResult` (SQLite), aggregated to `SKUMLProfile` for dashboard display.

---

## 🖥️ UI / UX

- **Dark/Light theme** toggle, persisted in `localStorage`
- **Fixed sidebar** with role-conditional navigation
- **Toast notifications** (auto-dismiss 4.5s) for all Django messages
- **Animated counters** on dashboard stats (ease-out cubic)
- **Plotly** for all ML visualizations: forecast charts with IC bands, Pareto frontier, control charts with zone shading, ABC Pareto curve
- **Chart.js** for operational dashboard (doughnut by category)
- **HTMX** for product search/filter/sort/pagination without page reload
- **Product autocomplete** (debounced 220ms, JSON endpoint)
- Responsive — collapsible sidebar on mobile

---

## 🔧 Tech Stack — Justified

| Technology | Version | Why This, Not Something Else |
|---|---|---|
| **Django** | 5.x | Batteries-included ORM, admin, auth, migrations. FastAPI would require rebuilding all of this. |
| **Celery + Redis** | 5.3 / 7 | True async workers with retry, time limits, task routing. Django-Q is simpler but less production-credible. Background threads would block the WSGI server. |
| **pmdarima** | 2.x | `auto_arima` with stepwise Hyndman-Khandakar. Manual grid search over (p,d,q,P,D,Q) is O(n⁵) and produces equivalent results with 10× the compute. |
| **Prophet** | 1.1 | Best-in-class for missing values, structural breaks, and holiday effects. statsmodels SARIMA requires complete series. |
| **LightGBM** | 4.x | Faster than XGBoost on tabular features, native quantile regression, `n_jobs=-1` parallelism. Neural alternatives (N-BEATS) are harder to interpret. |
| **PyTorch Forecasting** | 1.x | Only library with TFT production implementation including Variable Selection Networks and attention visualization. |
| **Conformal Prediction** | custom | Distribution-free. All parametric alternatives (Bayesian, bootstrap) require assumptions that fail on industrial residuals. |
| **NumPy/SciPy** | — | Monte Carlo at 10,000 scenarios needs vectorized ops. pandas would be 5× slower for inner simulation loop. |
| **MLflow** | 2.x | Standard in ML engineering. Experiment comparison, parameter logging, model registry. W&B requires an account. |
| **HTMX** | 1.9 | Partial page updates with 0 JavaScript. React/Vue would triple the frontend complexity for no gain in this use case. |
| **Plotly** | 2.27 | Interactive zoom, hover, annotation on control charts and Pareto frontiers. Matplotlib is static. Chart.js lacks the financial-grade interactivity needed. |
| **SQLite** | — | Sufficient for a demonstration. Connection string in `settings.py` is the only change needed for PostgreSQL. |

---

## ⚙️ Installation

```bash
# 1. Clone & environment
git clone https://github.com/YOUR_USERNAME/stockflow-intelligence.git
cd stockflow-intelligence
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Dependencies
pip install -r requirements.txt

# 3. Database
python manage.py migrate
python manage.py seed_data       # Groups + test accounts + 10 demo products

# 4. Services (separate terminals)
redis-server
celery -A gestion_stock worker --loglevel=info

# 5. Run
python manage.py runserver
```

### Optional

```bash
# TFT model
pip install pytorch-forecasting pytorch-lightning torch

# MLflow tracking UI
pip install mlflow
mlflow ui                        # → http://localhost:5000
```

---

## 👤 Test Accounts

| User | Password | Role | Permissions |
|---|---|---|---|
| `super1` | `super1pass` | Superadmin | Full CRUD + All ML + Dataset generation |
| `admin1` | `admin1pass` | Admin | Product CRUD + All ML modules |
| `user1` | `user1pass` | Viewer | Read-only + All ML modules |

**First run:** log in as `super1` → Intelligence → Generate Dataset (730 days) → all ML modules become operational.

---

## 📐 Methodological Limitations (documented honestly)

| Limitation | Impact | Mitigation |
|---|---|---|
| Synthetic data — no cross-SKU correlations | Substitution effects not captured | Modular generator accepts real CSV |
| EnbPI exchangeability violation | Coverage guarantee is approximate | Recent-window calibration reduces bias |
| Pareto grid (15×15) | Frontier density limited | NSGA-II would improve; grid is sufficient for demo |
| MC assumes daily demand independence | Underestimates correlated risk | Acceptable for i.i.d. SKUs; noted in docs |

---

## 📚 References

- Montgomery, D.C. (2020). *Introduction to Statistical Quality Control*, 8th ed. Wiley.
- Lim et al. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *IJF*.
- Romano, Y., Patterson, E., Candès, E. (2019). Conformalized Quantile Regression. *NeurIPS*.
- Xu, C., Xie, Y. (2021). Conformal Prediction Interval for Dynamic Time-Series. *ICML*.
- Syntetos, A.A., Boylan, J.E. (2005). The accuracy of intermittent demand estimates. *IJF*.
- Page, E.S. (1954). Continuous inspection schemes. *Biometrika*.
- Chopra, S., Meindl, P. (2016). *Supply Chain Management*, 6th ed. Pearson.

---

## 🎓 About

**Author:** 4th-year engineering student — AI & Data Science: Industrial Systems  
**Institution:** ENSAM Meknès — Génie Industriel (Lean Six Sigma, MSP, Quality Management)  

This project demonstrates that the gap between academic ML methodology and production-grade industrial tooling can be closed by a single engineer with sufficient domain knowledge and two months. The Lean Six Sigma background is not decorative — it shapes every modeling choice, from the asymmetric shortage/holding cost ratio to the Six Sigma service level reference points.

---

<div align="center">

**If this project is useful to you, a ⭐ on GitHub goes a long way.**

</div>
