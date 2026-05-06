"""
gestion_stock/celery.py
------------------------
Configuration Celery pour StockFlow Intelligence.
"""
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gestion_stock.settings')

app = Celery('stockflow')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks(['ml'])

@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
