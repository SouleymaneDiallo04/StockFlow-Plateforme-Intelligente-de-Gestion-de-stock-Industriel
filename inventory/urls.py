from django.urls import path
from . import views

urlpatterns = [
    # ── Dashboard ──────────────────────────────────────────────────────────────
    path('', views.dashboard, name='dashboard'),

    # ── Products ───────────────────────────────────────────────────────────────
    path('products/',                  views.product_list,   name='product_list'),
    path('products/add/',              views.product_create, name='product_create'),
    path('products/<int:pk>/',         views.product_detail, name='product_detail'),
    path('products/<int:pk>/edit/',    views.product_update, name='product_update'),
    path('products/<int:pk>/delete/',  views.product_delete, name='product_delete'),
    path('products/autocomplete/',     views.product_search_autocomplete, name='product_autocomplete'),

    # ── Categories ─────────────────────────────────────────────────────────────
    path('categories/',                    views.category_list,   name='category_list'),
    path('categories/add/',                views.category_create, name='category_create'),
    path('categories/<int:pk>/edit/',      views.category_update, name='category_update'),
    path('categories/<int:pk>/delete/',    views.category_delete, name='category_delete'),

    # ── ML Intelligence ────────────────────────────────────────────────────────
    path('intelligence/',                           views.ml_dashboard,      name='ml_dashboard'),
    path('intelligence/forecast/',                  views.ml_forecast,       name='ml_forecast'),
    path('intelligence/forecast/run/',              views.ml_forecast_run,   name='ml_forecast_run'),
    path('intelligence/optimize/',                  views.ml_optimize,       name='ml_optimize'),
    path('intelligence/optimize/run/',              views.ml_optimize_run,   name='ml_optimize_run'),
    path('intelligence/spc/',                       views.ml_spc,            name='ml_spc'),
    path('intelligence/spc/run/',                   views.ml_spc_run,        name='ml_spc_run'),
    path('intelligence/generate/',                  views.ml_generate_data,  name='ml_generate'),

    path('intelligence/abc/',                      views.ml_abc,            name='ml_abc'),
    path('intelligence/abc/run/',                   views.ml_abc_run,        name='ml_abc_run'),

    # ── API polling (HTMX) ────────────────────────────────────────────────────
    path('api/ml/job/<str:job_id>/status/',         views.ml_job_status,     name='ml_job_status'),
    path('api/ml/job/<str:job_id>/result/',         views.ml_job_result,     name='ml_job_result'),
]
