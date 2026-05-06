"""
inventory/views.py
------------------
Vues Django pour StockFlow Intelligence.
"""
from __future__ import annotations

import csv
import json
import uuid
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth import login
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q, Count, F
from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_POST

from .models import Category, Product, MLJobResult, SKUMLProfile
from .forms import CategoryForm, ProductForm, RegisterForm
from .permissions import is_superadmin, is_admin_or_superadmin


# ── Auth ──────────────────────────────────────────────────────────────────────

def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    form = RegisterForm(request.POST or None)
    if form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, f"Bienvenue {user.username} !")
        return redirect('dashboard')
    return render(request, 'registration/register.html', {'form': form})


# ── Dashboard opérationnel ────────────────────────────────────────────────────

@login_required
def dashboard(request):
    products = Product.objects.select_related('category')
    total_products    = products.count()
    total_categories  = Category.objects.count()
    out_of_stock      = products.filter(stock=0).count()
    low_stock         = products.filter(stock__gt=0, stock__lte=F('alert_threshold')).count()

    chart_data = list(
        Category.objects.annotate(count=Count('products'))
        .values('name', 'count').order_by('-count')
    )
    recent_products = products.order_by('-created_at')[:6]
    alerts = products.filter(
        Q(stock=0) | Q(stock__gt=0, stock__lte=F('alert_threshold'))
    ).order_by('stock')[:8]

    # Récents jobs ML
    recent_jobs = MLJobResult.objects.filter(
        status=MLJobResult.STATUS_SUCCESS
    ).order_by('-updated_at')[:5]

    return render(request, 'inventory/dashboard.html', {
        'total_products':   total_products,
        'total_categories': total_categories,
        'out_of_stock':     out_of_stock,
        'low_stock':        low_stock,
        'chart_data':       chart_data,
        'recent_products':  recent_products,
        'alerts':           alerts,
        'recent_jobs':      recent_jobs,
    })


# ── Products ──────────────────────────────────────────────────────────────────

@login_required
def product_list(request):
    query        = request.GET.get('q', '').strip()
    category_id  = request.GET.get('category', '')
    status_filter = request.GET.get('status', '')
    sort         = request.GET.get('sort', 'name')
    stock_filter = request.GET.get('stock', '')

    products = Product.objects.select_related('category')

    if query:
        products = products.filter(
            Q(name__icontains=query) | Q(description__icontains=query) | Q(reference__icontains=query)
        )
    if category_id:
        products = products.filter(category_id=category_id)
    if status_filter:
        products = products.filter(status=status_filter)
    if stock_filter == 'out':
        products = products.filter(stock=0)
    elif stock_filter == 'low':
        products = products.filter(stock__gt=0, stock__lte=F('alert_threshold'))

    valid_sorts = ['name', '-name', 'price', '-price', 'stock', '-stock', '-created_at', 'created_at']
    if sort in valid_sorts:
        products = products.order_by(sort)

    if request.GET.get('export') == 'csv':
        return _export_csv(products)

    total_results = products.count()
    paginator = Paginator(products, 9)
    page_obj  = paginator.get_page(request.GET.get('page'))
    categories = Category.objects.all()

    ctx = {
        'page_obj': page_obj, 'categories': categories,
        'query': query, 'category_id': category_id,
        'sort': sort, 'status_filter': status_filter,
        'stock_filter': stock_filter, 'total_results': total_results,
    }

    if request.headers.get('HX-Request'):
        return render(request, 'inventory/partials/product_grid.html', ctx)
    return render(request, 'inventory/product_list.html', ctx)


def _export_csv(queryset):
    response = HttpResponse(content_type='text/csv; charset=utf-8')
    response['Content-Disposition'] = 'attachment; filename="stockflow_export.csv"'
    response.write('\ufeff')
    writer = csv.writer(response, delimiter=';')
    writer.writerow(['Référence', 'Nom', 'Catégorie', 'Prix (DH)', 'Stock',
                     'Seuil alerte', 'Statut', 'SKU ML', 'Créé le'])
    for p in queryset:
        writer.writerow([
            p.reference, p.name, p.category.name, p.price,
            p.stock, p.alert_threshold, p.get_status_display(),
            p.ml_sku_id or '', p.created_at.strftime('%d/%m/%Y'),
        ])
    return response


@login_required
def product_detail(request, pk):
    product = get_object_or_404(Product, pk=pk)
    ml_profile = None
    if product.ml_sku_id:
        ml_profile = SKUMLProfile.objects.filter(sku_id=product.ml_sku_id).first()
    return render(request, 'inventory/product_detail.html', {
        'product': product, 'ml_profile': ml_profile,
    })


@login_required
@user_passes_test(is_admin_or_superadmin)
def product_create(request):
    form = ProductForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        form.save()
        messages.success(request, "Produit ajouté avec succès.")
        return redirect('product_list')
    return render(request, 'inventory/product_form.html', {
        'form': form, 'action': 'Ajouter un produit', 'is_create': True,
    })


@login_required
@user_passes_test(is_admin_or_superadmin)
def product_update(request, pk):
    product = get_object_or_404(Product, pk=pk)
    form = ProductForm(request.POST or None, request.FILES or None, instance=product)
    if form.is_valid():
        form.save()
        messages.success(request, "Produit modifié avec succès.")
        return redirect('product_detail', pk=pk)
    return render(request, 'inventory/product_form.html', {
        'form': form, 'action': f'Modifier : {product.name}', 'product': product,
    })


@login_required
@user_passes_test(is_admin_or_superadmin)
def product_delete(request, pk):
    product = get_object_or_404(Product, pk=pk)
    if request.method == 'POST':
        product.delete()
        messages.warning(request, f"Produit « {product.name} » supprimé.")
        return redirect('product_list')
    return render(request, 'inventory/product_confirm_delete.html', {
        'object': product, 'type': 'produit',
    })


@login_required
def product_search_autocomplete(request):
    q = request.GET.get('q', '').strip()
    results = []
    if len(q) >= 2:
        products = Product.objects.filter(
            Q(name__icontains=q) | Q(reference__icontains=q)
        ).values('pk', 'name', 'reference')[:8]
        results = list(products)
    return JsonResponse({'results': results})


# ── Categories ────────────────────────────────────────────────────────────────

@login_required
@user_passes_test(is_superadmin)
def category_list(request):
    categories = Category.objects.annotate(count=Count('products'))
    return render(request, 'inventory/category_list.html', {'categories': categories})


@login_required
@user_passes_test(is_superadmin)
def category_create(request):
    form = CategoryForm(request.POST or None)
    if form.is_valid():
        form.save()
        messages.success(request, "Catégorie ajoutée.")
        return redirect('category_list')
    return render(request, 'inventory/category_form.html', {
        'form': form, 'action': 'Ajouter une catégorie',
    })


@login_required
@user_passes_test(is_superadmin)
def category_update(request, pk):
    category = get_object_or_404(Category, pk=pk)
    form = CategoryForm(request.POST or None, instance=category)
    if form.is_valid():
        form.save()
        messages.success(request, "Catégorie modifiée.")
        return redirect('category_list')
    return render(request, 'inventory/category_form.html', {
        'form': form, 'action': f'Modifier : {category.name}', 'category': category,
    })


@login_required
@user_passes_test(is_superadmin)
def category_delete(request, pk):
    category = get_object_or_404(Category, pk=pk)
    if request.method == 'POST':
        if category.products.exists():
            messages.error(request,
                f"Impossible de supprimer « {category.name} » : "
                f"elle contient {category.products.count()} produit(s).")
            return redirect('category_list')
        category.delete()
        messages.warning(request, f"Catégorie « {category.name} » supprimée.")
        return redirect('category_list')
    return render(request, 'inventory/product_confirm_delete.html', {
        'object': category, 'type': 'catégorie',
    })


# ── ML Intelligence — vues principales ───────────────────────────────────────

@login_required
def ml_dashboard(request):
    """Dashboard Intelligence : vue d'ensemble des analyses ML."""
    sku_profiles  = SKUMLProfile.objects.all().order_by('sku_id')
    recent_jobs   = MLJobResult.objects.order_by('-created_at')[:10]
    pending_jobs  = MLJobResult.objects.filter(
        status__in=[MLJobResult.STATUS_PENDING, MLJobResult.STATUS_RUNNING]
    ).count()

    # KPIs agrégés
    avg_service = None
    total_capital_reduction = None
    out_of_control_count = SKUMLProfile.objects.filter(spc_status='out_of_control').count()

    profiles_with_opt = sku_profiles.filter(service_level__isnull=False)
    if profiles_with_opt.exists():
        from django.db.models import Avg, Sum
        avg_service = profiles_with_opt.aggregate(avg=Avg('service_level'))['avg']

    # Catalogue ML disponible
    try:
        from ml.data.generator import build_sku_catalog
        catalog = build_sku_catalog()
        catalog_size = len(catalog)
        categories_ml = list({c.category.value for c in catalog})
    except Exception:
        catalog_size = 0
        categories_ml = []

    return render(request, 'inventory/ml_dashboard.html', {
        'sku_profiles':       sku_profiles,
        'recent_jobs':        recent_jobs,
        'pending_jobs':       pending_jobs,
        'avg_service':        round(avg_service * 100, 1) if avg_service else None,
        'out_of_control':     out_of_control_count,
        'catalog_size':       catalog_size,
        'categories_ml':      categories_ml,
    })


@login_required
def ml_forecast(request):
    """Page de prévision de demande."""
    try:
        from ml.data.generator import build_sku_catalog
        catalog = [(c.sku_id, c.name, c.category.value, c.profile.value)
                   for c in build_sku_catalog()]
    except Exception:
        catalog = []

    recent_jobs = MLJobResult.objects.filter(
        job_type=MLJobResult.JOB_FORECAST
    ).order_by('-created_at')[:5]

    return render(request, 'inventory/ml_forecast.html', {
        'catalog': catalog, 'recent_jobs': recent_jobs,
    })


@login_required
@require_POST
def ml_forecast_run(request):
    """Lance la tâche Celery de prévision."""
    sku_id  = request.POST.get('sku_id', '').strip()
    horizon = int(request.POST.get('horizon', 30))
    models  = request.POST.getlist('models') or ['SARIMAX', 'Prophet', 'LightGBM']

    if not sku_id:
        return JsonResponse({'error': 'SKU requis'}, status=400)

    job_id = str(uuid.uuid4())[:16]
    job = MLJobResult.objects.create(
        job_id=job_id, job_type=MLJobResult.JOB_FORECAST,
        sku_id=sku_id, status=MLJobResult.STATUS_PENDING,
        created_by=request.user,
    )

    try:
        from ml.tasks import forecast_sku_task
        forecast_sku_task.delay(
            sku_id=sku_id, horizon=horizon, models=models, job_id=job_id
        )
    except Exception as e:
        job.status = MLJobResult.STATUS_FAILED
        job.error  = str(e)
        job.save()
        return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'job_id': job_id, 'status': 'pending'})


@login_required
def ml_optimize(request):
    """Page d'optimisation des politiques de stock."""
    try:
        from ml.data.generator import build_sku_catalog
        catalog = [(c.sku_id, c.name, c.category.value,
                    round(c.unit_cost, 0), c.profile.value)
                   for c in build_sku_catalog()]
    except Exception:
        catalog = []

    recent_jobs = MLJobResult.objects.filter(
        job_type=MLJobResult.JOB_OPTIMIZE
    ).order_by('-created_at')[:5]

    return render(request, 'inventory/ml_optimize.html', {
        'catalog': catalog, 'recent_jobs': recent_jobs,
    })


@login_required
@require_POST
def ml_optimize_run(request):
    """Lance la tâche Celery d'optimisation."""
    sku_id   = request.POST.get('sku_id', '').strip()
    sl_target = float(request.POST.get('service_level', 0.95))

    if not sku_id:
        return JsonResponse({'error': 'SKU requis'}, status=400)

    job_id = str(uuid.uuid4())[:16]
    job = MLJobResult.objects.create(
        job_id=job_id, job_type=MLJobResult.JOB_OPTIMIZE,
        sku_id=sku_id, status=MLJobResult.STATUS_PENDING,
        created_by=request.user,
    )

    try:
        from ml.tasks import optimize_policy_task
        optimize_policy_task.delay(
            sku_id=sku_id, target_service_level=sl_target, job_id=job_id
        )
    except Exception as e:
        job.status = MLJobResult.STATUS_FAILED
        job.error  = str(e)
        job.save()
        return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'job_id': job_id, 'status': 'pending'})


@login_required
def ml_spc(request):
    """Page de contrôle statistique SPC."""
    try:
        from ml.data.generator import build_sku_catalog
        catalog = [(c.sku_id, c.name, c.category.value) for c in build_sku_catalog()]
    except Exception:
        catalog = []

    recent_jobs = MLJobResult.objects.filter(
        job_type=MLJobResult.JOB_SPC
    ).order_by('-created_at')[:5]

    profiles = SKUMLProfile.objects.filter(
        spc_status__in=['out_of_control', 'warning']
    ).order_by('spc_status', '-spc_n_critical')[:10]

    return render(request, 'inventory/ml_spc.html', {
        'catalog': catalog, 'recent_jobs': recent_jobs, 'alert_profiles': profiles,
    })


@login_required
@require_POST
def ml_spc_run(request):
    """Lance la tâche Celery SPC."""
    sku_id = request.POST.get('sku_id', '').strip()
    if not sku_id:
        return JsonResponse({'error': 'SKU requis'}, status=400)

    job_id = str(uuid.uuid4())[:16]
    MLJobResult.objects.create(
        job_id=job_id, job_type=MLJobResult.JOB_SPC,
        sku_id=sku_id, status=MLJobResult.STATUS_PENDING,
        created_by=request.user,
    )

    try:
        from ml.tasks import spc_report_task
        spc_report_task.delay(sku_id=sku_id, job_id=job_id)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'job_id': job_id, 'status': 'pending'})


@login_required
@user_passes_test(is_superadmin)
def ml_generate_data(request):
    """Lance la génération du dataset synthétique (superadmin uniquement)."""
    if request.method == 'POST':
        n_days = int(request.POST.get('n_days', 730))
        job_id = str(uuid.uuid4())[:16]
        MLJobResult.objects.create(
            job_id=job_id, job_type=MLJobResult.JOB_GENERATE,
            status=MLJobResult.STATUS_PENDING, created_by=request.user,
        )
        try:
            from ml.tasks import generate_dataset_task
            generate_dataset_task.delay(n_days=n_days, job_id=job_id)
            messages.success(request, f"Génération lancée (job {job_id}).")
        except Exception as e:
            messages.error(request, f"Erreur : {e}")
        return redirect('ml_dashboard')
    return render(request, 'inventory/ml_generate.html')


# ── API ML polling ────────────────────────────────────────────────────────────

@login_required
def ml_job_status(request, job_id):
    """
    Endpoint HTMX — retourne le statut d'une tâche.
    Retourne du HTML partiel si HX-Request, JSON sinon.
    """
    job = get_object_or_404(MLJobResult, job_id=job_id)

    if request.headers.get('HX-Request'):
        return render(request, 'inventory/partials/job_status.html', {'job': job})

    return JsonResponse({
        'job_id':  job.job_id,
        'status':  job.status,
        'is_done': job.is_done,
        'error':   job.error,
    })


@login_required
def ml_job_result(request, job_id):
    """Retourne le résultat complet d'une tâche (JSON)."""
    job = get_object_or_404(MLJobResult, job_id=job_id)

    if job.status != MLJobResult.STATUS_SUCCESS:
        return JsonResponse({'error': 'Résultat non disponible', 'status': job.status}, status=400)

    result = job.result
    if result is None:
        return JsonResponse({'error': 'Résultat vide'}, status=500)

    # Mettre à jour le profil ML du SKU après succès
    if job.sku_id:
        _update_sku_profile(job, result)

    return JsonResponse(result)


def _update_sku_profile(job: MLJobResult, result: dict) -> None:
    """Met à jour SKUMLProfile après une tâche réussie."""
    try:
        profile, _ = SKUMLProfile.objects.get_or_create(sku_id=job.sku_id)

        if job.job_type == MLJobResult.JOB_FORECAST:
            profile.best_forecast_model = result.get('best_model', '')
            comp = result.get('comparison_table', [])
            if comp:
                profile.forecast_mase     = comp[0].get('MASE')
                profile.forecast_coverage = comp[0].get('Coverage (%)', 0) / 100
            profile.last_forecast_job = job

        elif job.job_type == MLJobResult.JOB_OPTIMIZE:
            profile.reorder_point     = result.get('reorder_point')
            profile.order_quantity    = result.get('order_quantity')
            profile.safety_stock      = result.get('safety_stock')
            profile.service_level     = result.get('service_level')
            lean = result.get('lean_analysis', {})
            profile.capital_reduction = lean.get('reduction_capital_pct')
            profile.last_optimize_job = job

        elif job.job_type == MLJobResult.JOB_SPC:
            profile.spc_status    = result.get('overall_status', '')
            profile.spc_n_critical = len(result.get('critical_signals', []))
            profile.last_spc_job  = job

        profile.save()
    except Exception:
        pass


# ── ML ABC-XYZ ────────────────────────────────────────────────────────────────

@login_required
def ml_abc(request):
    """Page analyse ABC-XYZ."""
    recent_jobs = MLJobResult.objects.filter(
        job_type='abc'
    ).order_by('-created_at')[:3]
    return render(request, 'inventory/ml_abc.html', {'recent_jobs': recent_jobs})


@login_required
@require_POST
def ml_abc_run(request):
    """Lance la tâche Celery ABC."""
    job_id = str(uuid.uuid4())[:16]
    MLJobResult.objects.create(
        job_id=job_id, job_type='abc',
        status=MLJobResult.STATUS_PENDING, created_by=request.user,
    )
    try:
        from ml.tasks import abc_analysis_task
        abc_analysis_task.delay(job_id=job_id)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'job_id': job_id, 'status': 'pending'})
