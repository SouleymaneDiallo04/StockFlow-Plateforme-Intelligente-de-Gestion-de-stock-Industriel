"""
ml/tasks.py - Tâches Celery StockFlow Intelligence
"""
from __future__ import annotations
import json, traceback
from datetime import datetime
from typing import Optional
from celery import shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

def _save(job_id, status, result=None, error=""):
    try:
        from inventory.models import MLJobResult
        MLJobResult.objects.update_or_create(
            job_id=job_id,
            defaults={'status': status,
                      'result_json': json.dumps(result, default=str) if result else None,
                      'error': error, 'updated_at': datetime.now()})
    except Exception as e:
        logger.error(f"Cannot save job {job_id}: {e}")

@shared_task(bind=True, time_limit=180, name='ml.generate_dataset')
def generate_dataset_task(self, n_days=730, job_id=""):
    try:
        _save(job_id, 'running')
        import os
        from django.conf import settings
        from ml.data.generator import SyntheticDataGenerator
        from ml.data.validator import DatasetValidator
        gen = SyntheticDataGenerator(seed=42)
        dataset = gen.generate_all(n_days=n_days)
        output_dir = os.path.join(settings.MEDIA_ROOT, 'ml_data')
        gen.save_dataset(dataset, output_dir)
        validator = DatasetValidator()
        val = validator.validate_dataset(dataset)
        summary = validator.dataset_summary(val)
        result = {'n_skus': len(dataset), 'n_days': n_days,
                  'output_dir': output_dir, 'validation': summary,
                  'generated_at': datetime.now().isoformat()}
        _save(job_id, 'success', result)
        return result
    except Exception as e:
        _save(job_id, 'failed', error=str(e))
        raise

@shared_task(bind=True, time_limit=600, name='ml.forecast_sku')
def forecast_sku_task(self, sku_id, horizon=30, models=None, job_id=""):
    if models is None:
        models = ['SARIMAX', 'Prophet', 'LightGBM']
    try:
        _save(job_id, 'running')
        import numpy as np
        import pandas as pd
        from ml.data.generator import SyntheticDataGenerator, build_sku_catalog
        from ml.forecasting.evaluator import ForecastEvaluator
        from ml.analysis.mlflow_tracker import tracker
        from ml.analysis.external_data import build_calendar_features

        gen = SyntheticDataGenerator(seed=42)
        dataset = gen.generate_all(n_days=730)
        if sku_id not in dataset:
            raise ValueError(f"SKU {sku_id} non trouve")

        sku = dataset[sku_id]
        dates = pd.DatetimeIndex(pd.to_datetime(sku.dates))
        series = pd.Series(sku.demand, index=dates)
        exog = pd.DataFrame({'event_flag': sku.events}, index=dates)
        cal = build_calendar_features(dates)
        for col in ['is_holiday', 'is_ramadan', 'is_end_of_month', 'week_sin', 'week_cos']:
            if col in cal.columns:
                exog[col] = cal[col]

        split = int(len(series) * 0.8)
        train, test = series.iloc[:split], series.iloc[split:]
        exog_train, exog_test = exog.iloc[:split], exog.iloc[split:]

        results = []
        catalog = {c.sku_id: c for c in build_sku_catalog()}
        sku_cfg = catalog.get(sku_id)

        for model_name in models:
            try:
                if model_name == 'SARIMAX':
                    from ml.forecasting.sarima import SARIMAXForecaster
                    m = SARIMAXForecaster()
                elif model_name == 'Prophet':
                    from ml.forecasting.prophet_model import ProphetForecaster
                    m = ProphetForecaster()
                elif model_name == 'LightGBM':
                    from ml.forecasting.lgbm_model import LGBMForecaster
                    m = LGBMForecaster()
                elif model_name == 'TFT':
                    from ml.forecasting.tft_model import TFTForecaster
                    m = TFTForecaster(max_epochs=30)
                else:
                    continue
                m.fit(train, exog_train)
                r = m.predict(len(test), future_exog=exog_test, sku_id=sku_id)
                results.append(r)
                if sku_cfg:
                    tracker.log_forecast_run(
                        model_name=model_name, sku_id=sku_id,
                        sku_profile=sku_cfg.profile.value,
                        params={'horizon': horizon},
                        metrics={'mase': r.mase or 0, 'mae': r.mae or 0,
                                 'coverage_rate': r.coverage_rate or 0},
                        training_time_s=r.training_time_s)
                logger.info(f"[{job_id}] {model_name} OK")
            except Exception as e:
                logger.warning(f"[{job_id}] {model_name} failed: {e}")

        evaluator = ForecastEvaluator()
        comp_df = evaluator.comparison_dataframe(results, test.values, train.values)

        serialized = []
        for r in results:
            serialized.append({
                'model_name': r.model_name,
                'forecast_dates': [str(d)[:10] for d in r.forecast_dates],
                'point_forecast': r.point_forecast.tolist(),
                'lower_bound': r.lower_bound.tolist(),
                'upper_bound': r.upper_bound.tolist(),
                'mase': r.mase, 'mae': r.mae, 'coverage_rate': r.coverage_rate,
            })

        result = {
            'sku_id': sku_id, 'horizon': horizon,
            'n_train': len(train), 'n_test': len(test),
            'models_run': [r['model_name'] for r in serialized],
            'best_model': comp_df.iloc[0]['model'] if not comp_df.empty else None,
            'comparison_table': comp_df.to_dict('records') if not comp_df.empty else [],
            'forecasts': serialized,
            'train_dates': [str(d)[:10] for d in train.index],
            'train_values': train.values.tolist(),
            'test_dates': [str(d)[:10] for d in test.index],
            'test_values': test.values.tolist(),
            'mlflow_enabled': tracker.enabled,
        }
        _save(job_id, 'success', result)
        return result
    except Exception as e:
        _save(job_id, 'failed', error=str(e))
        logger.error(traceback.format_exc())
        raise

@shared_task(bind=True, time_limit=360, name='ml.optimize_policy')
def optimize_policy_task(self, sku_id, target_service_level=0.95, job_id=""):
    try:
        _save(job_id, 'running')
        from ml.data.generator import build_sku_catalog
        from ml.optimization.policy import StockPolicyOptimizer
        from ml.analysis.threshold_optimizer import ThresholdOptimizer
        from ml.analysis.mlflow_tracker import tracker

        catalog = {c.sku_id: c for c in build_sku_catalog()}
        if sku_id not in catalog:
            raise ValueError(f"SKU {sku_id} non trouve")
        sku_config = catalog[sku_id]
        optimizer = StockPolicyOptimizer(n_simulations=8_000, horizon_days=365)
        policy, frontier, lean = optimizer.optimize(sku_config, target_service_level)

        thresh_opt = ThresholdOptimizer()
        rec = thresh_opt.from_policy(
            sku_id=sku_id, reorder_point=policy.reorder_point,
            safety_stock=policy.safety_stock, service_level=policy.service_level,
            current_threshold=sku_config.initial_stock)
        threshold_updated = thresh_opt.apply_to_product(rec)

        tracker.log_optimization_run(
            sku_id=sku_id, policy_type=policy.policy_type,
            reorder_point=policy.reorder_point, order_quantity=policy.order_quantity,
            service_level=policy.service_level, annual_cost=policy.expected_annual_cost,
            capital_reduction_pct=policy.capital_reduction_pct, n_simulations=8_000)

        result = {
            'sku_id': sku_id, 'policy_type': policy.policy_type,
            'reorder_point': round(policy.reorder_point, 2),
            'order_quantity': round(policy.order_quantity, 2),
            'safety_stock': round(policy.safety_stock, 2),
            'service_level': round(policy.service_level, 4),
            'expected_annual_cost': round(policy.expected_annual_cost, 2),
            'expected_stockout_days': round(policy.expected_stockout_days, 2),
            'expected_avg_stock': round(policy.expected_avg_stock, 2),
            'lean_analysis': lean,
            'threshold_recommendation': {
                'current': rec.current_threshold,
                'recommended': rec.recommended_threshold,
                'delta': rec.delta, 'updated': threshold_updated,
                'rationale': rec.rationale,
            },
            'pareto_frontier': [
                {'reorder_point': round(p.reorder_point, 2),
                 'order_quantity': round(p.order_quantity, 2),
                 'total_cost': round(p.total_cost, 2),
                 'stockout_prob': round(p.stockout_prob, 4),
                 'service_level': round(p.service_level, 4),
                 'safety_stock': round(p.safety_stock, 2)}
                for p in frontier],
        }
        _save(job_id, 'success', result)
        return result
    except Exception as e:
        _save(job_id, 'failed', error=str(e))
        logger.error(traceback.format_exc())
        raise

@shared_task(bind=True, time_limit=120, name='ml.spc_report')
def spc_report_task(self, sku_id, job_id=""):
    try:
        _save(job_id, 'running')
        import numpy as np
        from ml.data.generator import SyntheticDataGenerator
        from ml.spc.report import SPCReportGenerator

        gen = SyntheticDataGenerator(seed=42)
        dataset = gen.generate_all(n_days=730)
        if sku_id not in dataset:
            raise ValueError(f"SKU {sku_id} non trouve")

        sku = dataset[sku_id]
        generator = SPCReportGenerator()
        report = generator.generate(
            sku_id=sku_id, demand=sku.demand,
            dates=np.array([str(d)[:10] for d in sku.dates]))

        result = {
            'sku_id': report.sku_id,
            'generated_at': report.generated_at,
            'overall_status': report.overall_status,
            'recommendations': report.recommendations,
            'charts': [
                {
                    'chart_type': c.chart_type, 'metric_name': c.metric_name,
                    'in_control': c.in_control, 'n_signals': len(c.signals),
                    'dates': [str(d)[:10] for d in c.dates[-90:]],
                    'values': np.round(c.values[-90:], 4).tolist(),
                    'limits': {
                        'cl': c.limits.center_line, 'ucl': c.limits.ucl,
                        'lcl': c.limits.lcl, 'uwa': c.limits.uwa,
                        'lwa': c.limits.lwa, 'uwb': c.limits.uwb, 'lwb': c.limits.lwb,
                    },
                    'signals': [
                        {'rule': s.rule.value, 'point_index': s.point_index,
                         'date': s.date, 'value': s.value,
                         'description': s.description, 'severity': s.severity}
                        for s in c.signals],
                }
                for c in report.charts],
            'critical_signals': [
                {'rule': s.rule.value, 'date': s.date,
                 'value': s.value, 'description': s.description}
                for s in report.critical_signals],
        }
        _save(job_id, 'success', result)
        return result
    except Exception as e:
        _save(job_id, 'failed', error=str(e))
        logger.error(traceback.format_exc())
        raise

@shared_task(bind=True, time_limit=120, name='ml.abc_analysis')
def abc_analysis_task(self, job_id=""):
    try:
        _save(job_id, 'running')
        from ml.data.generator import SyntheticDataGenerator, build_sku_catalog
        from ml.analysis.abc import ABCXYZAnalyzer

        gen = SyntheticDataGenerator(seed=42)
        dataset = gen.generate_all(n_days=730)
        df = gen.to_dataframe(dataset)
        catalog = {c.sku_id: c for c in build_sku_catalog()}
        unit_costs = {sid: catalog[sid].unit_cost for sid in catalog}

        analyzer = ABCXYZAnalyzer()
        results = analyzer.analyze(df, unit_costs)
        summary = analyzer.portfolio_summary(results)
        pareto = analyzer.pareto_data(results)

        result = {
            'n_skus': len(results), 'summary': summary, 'pareto': pareto,
            'rankings': [
                {'sku_id': r.sku_id, 'abc': r.abc_class, 'xyz': r.xyz_class,
                 'matrix': r.matrix_class, 'annual_value': r.annual_value,
                 'value_pct': r.value_pct, 'cumulative_pct': r.cumulative_pct,
                 'cv': r.cv, 'recommendation': r.recommendation}
                for r in results],
            'generated_at': datetime.now().isoformat(),
        }
        _save(job_id, 'success', result)
        return result
    except Exception as e:
        _save(job_id, 'failed', error=str(e))
        logger.error(traceback.format_exc())
        raise
