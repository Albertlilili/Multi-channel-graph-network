#!/usr/bin/env python3
"""高斯过程回归（GPR）预测聚酰亚胺玻璃化转变温度（Tg，5 折 CV + 遗传算法 GA 优化）。"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)

from feature_pipeline import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    FeatureConfig,
    extract_raw_features,
    fit_transform_feature_fold,
)
from ga_utils import (
    CV_FOLDS,
    GA_CROSSOVER_RATE,
    GA_GENERATIONS,
    GA_MUTATION_RATE,
    GA_POP_SIZE,
    prepare_descriptor_cv_folds,
    run_discrete_ga,
)
from model_utils import finalize_model, regression_metrics

warnings.filterwarnings('ignore', category=ConvergenceWarning)

TRAIN_PATH = Path(DEFAULT_TRAIN_PATH)
TEST_PATH = Path(DEFAULT_TEST_PATH)
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'gpr'

RANDOM_STATE = 42

GA_SEARCH_SPACE: dict[str, tuple] = {
    'kernel_family': ('Matern_1.5', 'Matern_2.5', 'RBF', 'RationalQuadratic'),
    'alpha': (1e-10, 1e-6, 1e-4, 1e-2, 1e-1),
    'n_restarts_optimizer': (0, 2, 5),
    'length_scale_init': (1.0, 10.0, 30.0),
}


def make_kernel(kernel_family: str, length_scale_init: float):
    ls = float(length_scale_init)
    amp = ConstantKernel(1.0, (1e-3, 1e3))
    noise = WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2))
    if kernel_family == 'Matern_1.5':
        base = Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e3), nu=1.5)
    elif kernel_family == 'Matern_2.5':
        base = Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e3), nu=2.5)
    elif kernel_family == 'RBF':
        base = RBF(length_scale=ls, length_scale_bounds=(1e-2, 1e3))
    elif kernel_family == 'RationalQuadratic':
        base = RationalQuadratic(length_scale=ls, alpha=1.0)
    else:
        raise ValueError(f'未知核函数：{kernel_family}')
    return amp * base + noise


def build_gpr(params: dict) -> GaussianProcessRegressor:
    kernel = make_kernel(params['kernel_family'], params['length_scale_init'])
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(params['alpha']),
        normalize_y=True,
        n_restarts_optimizer=int(params['n_restarts_optimizer']),
        random_state=RANDOM_STATE,
    )


def main() -> None:
    cfg = FeatureConfig(random_state=RANDOM_STATE)
    raw = extract_raw_features(TRAIN_PATH, TEST_PATH, cfg)
    folds = prepare_descriptor_cv_folds(raw, cfg, n_splits=CV_FOLDS, seed=RANDOM_STATE)

    def eval_fn(params: dict) -> dict:
        fold_rmses = []
        for f in folds:
            pipe = build_gpr(params)
            pipe.fit(f['X_train'], f['y_train'])
            pred = pipe.predict(f['X_val'])
            rmse = float(np.sqrt(np.mean((pred - f['y_val']) ** 2)))
            fold_rmses.append(rmse)
        return {'mean_cv_rmse': float(np.mean(fold_rmses)), 'fold_rmses': fold_rmses}

    best_params, _, ga_evals_df = run_discrete_ga(
        model_tag='GPR',
        search_space=GA_SEARCH_SPACE,
        eval_fn=eval_fn,
        output_dir=OUTPUT_DIR,
        pop_size=GA_POP_SIZE,
        generations=GA_GENERATIONS,
        crossover_rate=GA_CROSSOVER_RATE,
        mutation_rate=GA_MUTATION_RATE,
        seed=RANDOM_STATE,
    )
    print(f'\nGPR GA 最优超参数：{json.dumps(best_params, ensure_ascii=False)}')

    cv_rows = []
    for f in folds:
        pipe = build_gpr(best_params)
        pipe.fit(f['X_train'], f['y_train'])
        pred = pipe.predict(f['X_val'])
        m = regression_metrics(f['y_val'], pred)
        cv_rows.append({
            'fold': f['fold'] + 1,
            'val_R2': m['R2'],
            'val_RMSE_C': m['RMSE_C'],
            'val_MAE_C': m['MAE_C'],
        })
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(OUTPUT_DIR / 'cv_best_fold_metrics.csv', index=False, encoding='utf-8-sig')

    all_train_idx = np.arange(len(raw.y_train))
    final_features = fit_transform_feature_fold(
        raw, train_idx=all_train_idx, config=cfg, use_test=True, verbose=True
    )
    model = build_gpr(best_params)
    model.fit(final_features.X_train, final_features.y_train)
    train_pred = np.asarray(model.predict(final_features.X_train)).ravel()
    test_pred, test_std = model.predict(final_features.X_test, return_std=True)
    test_pred = np.asarray(test_pred).ravel()
    test_std = np.asarray(test_std).ravel()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / 'gpr_model.joblib')
    fitted_kernel = str(model.kernel_)
    print(f'拟合后的核函数：{fitted_kernel}')

    finalize_model(
        'GPR', OUTPUT_DIR,
        final_features.y_train, train_pred,
        final_features.y_test, test_pred,
        final_features,
        extra_metadata={
            'ga_config': {
                'cv_folds': CV_FOLDS,
                'population_size': GA_POP_SIZE,
                'generations': GA_GENERATIONS,
                'crossover_rate': GA_CROSSOVER_RATE,
                'mutation_rate': GA_MUTATION_RATE,
                'fitness_metric': '5-fold CV mean RMSE (C)',
                'unique_configurations_evaluated': int(len(ga_evals_df)),
            },
            'ga_search_space': {k: list(v) for k, v in GA_SEARCH_SPACE.items()},
            'hyperparameters': best_params,
            'kernel_initial': str(make_kernel(
                best_params['kernel_family'], best_params['length_scale_init']
            )),
            'kernel_fitted': fitted_kernel,
            'normalize_y': True,
            'cv_5fold_summary': {
                'mean_R2': float(cv_df['val_R2'].mean()),
                'mean_RMSE_C': float(cv_df['val_RMSE_C'].mean()),
                'mean_MAE_C': float(cv_df['val_MAE_C'].mean()),
                'folds': cv_rows,
            },
            'test_prediction_std_mean_C': float(np.mean(test_std)),
            'test_prediction_std_max_C': float(np.max(test_std)),
            'test_used_for_optimization': False,
            'random_state': RANDOM_STATE,
        },
        y_test_std=test_std,
    )


if __name__ == '__main__':
    main()
