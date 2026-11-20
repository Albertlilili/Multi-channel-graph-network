from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVR

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

TRAIN_PATH = Path(DEFAULT_TRAIN_PATH)
TEST_PATH = Path(DEFAULT_TEST_PATH)
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'svr'

RANDOM_STATE = 42

GA_SEARCH_SPACE: dict[str, tuple] = {
    'kernel': ('rbf', 'poly'),
    'C': (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
    'gamma': ('scale', 'auto', 0.001, 0.003, 0.01, 0.03, 0.1),
    'epsilon': (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
}


def build_svr(params: dict) -> SVR:
    return SVR(
        kernel=str(params['kernel']),
        C=float(params['C']),
        gamma=params['gamma'],
        epsilon=float(params['epsilon']),
    )


def main() -> None:
    cfg = FeatureConfig(random_state=RANDOM_STATE)
    raw = extract_raw_features(TRAIN_PATH, TEST_PATH, cfg)
    folds = prepare_descriptor_cv_folds(raw, cfg, n_splits=CV_FOLDS, seed=RANDOM_STATE)

    def eval_fn(params: dict) -> dict:
        fold_rmses = []
        for f in folds:
            pipe = build_svr(params)
            pipe.fit(f['X_train'], f['y_train'])
            pred = pipe.predict(f['X_val'])
            rmse = float(np.sqrt(np.mean((pred - f['y_val']) ** 2)))
            fold_rmses.append(rmse)
        return {'mean_cv_rmse': float(np.mean(fold_rmses)), 'fold_rmses': fold_rmses}

    best_params, _, ga_evals_df = run_discrete_ga(
        model_tag='SVR',
        search_space=GA_SEARCH_SPACE,
        eval_fn=eval_fn,
        output_dir=OUTPUT_DIR,
        pop_size=GA_POP_SIZE,
        generations=GA_GENERATIONS,
        crossover_rate=GA_CROSSOVER_RATE,
        mutation_rate=GA_MUTATION_RATE,
        seed=RANDOM_STATE,
    )
    print(f'\nSVR GA 最优超参数：{json.dumps(best_params, ensure_ascii=False)}')

    # 5 折完整评估详情
    cv_rows = []
    for f in folds:
        pipe = build_svr(best_params)
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

    # 在完整训练集上重新拟合特征筛选与 SVR，并在独立测试集上一次性评估
    all_train_idx = np.arange(len(raw.y_train))
    final_features = fit_transform_feature_fold(
        raw, train_idx=all_train_idx, config=cfg, use_test=True, verbose=True
    )
    model = build_svr(best_params)
    model.fit(final_features.X_train, final_features.y_train)
    train_pred = model.predict(final_features.X_train)
    test_pred = model.predict(final_features.X_test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / 'svr_model.joblib')

    finalize_model(
        'SVR', OUTPUT_DIR,
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
            'cv_5fold_summary': {
                'mean_R2': float(cv_df['val_R2'].mean()),
                'mean_RMSE_C': float(cv_df['val_RMSE_C'].mean()),
                'mean_MAE_C': float(cv_df['val_MAE_C'].mean()),
                'folds': cv_rows,
            },
            'test_used_for_optimization': False,
            'random_state': RANDOM_STATE,
        },
    )


if __name__ == '__main__':
    main()
