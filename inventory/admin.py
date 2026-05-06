from django.contrib import admin
from .models import Category, Product, MLJobResult, SKUMLProfile


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display  = ('name', 'product_count', 'created_at')
    search_fields = ('name',)


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display   = ('reference', 'name', 'category', 'price', 'stock', 'stock_status', 'status', 'ml_sku_id')
    search_fields  = ('name', 'reference', 'ml_sku_id')
    list_filter    = ('category', 'status')
    readonly_fields = ('reference', 'created_at', 'updated_at')


@admin.register(MLJobResult)
class MLJobResultAdmin(admin.ModelAdmin):
    list_display  = ('job_id', 'job_type', 'sku_id', 'status', 'created_by', 'updated_at')
    list_filter   = ('job_type', 'status')
    readonly_fields = ('job_id', 'created_at', 'updated_at', 'result_json')
    search_fields = ('job_id', 'sku_id')


@admin.register(SKUMLProfile)
class SKUMLProfileAdmin(admin.ModelAdmin):
    list_display = ('sku_id', 'best_forecast_model', 'forecast_mase', 'service_level', 'spc_status', 'updated_at')
    list_filter  = ('spc_status', 'best_forecast_model')
