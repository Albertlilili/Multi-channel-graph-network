#!/usr/bin/env python3
"""随机森林（RF）预测聚酰亚胺玻璃化转变温度（Tg）。

运行：
    /home/albert/anaconda3/envs/gnn/bin/python \
        /home/albert/MD_LAMMPS/GNN_OA/ML_Tg_prediction/rf_model.py

流程：RDKit 描述符 + Morgan 指纹 -> 训练集内筛选 ->
      5 折交叉验证随机搜索随机森林超参数 -> 训练集/测试集评估。
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV

from feature_pipeline import (
    DEFAULT_TEST_PATH, DEFAULT_TRAIN_PATH, FeatureConfig, build_features,
)
from model_utils import finalize_model


# ==========================================================
# 1. 路径与参数
# ==========================================================
TRAIN_PATH = Path(DEFAULT_TRAIN_PATH)
TEST_PATH = Path(DEFAULT_TEST_PATH)
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'rf'

RANDOM_STATE = 42
K_BEST = 300
SEARCH_N_ITER = 25
CV_FOLDS = 4
USE_SEARCH = True


def main() -> None:
    # -------- 1. 特征提取与筛选 --------
    features = build_features(
        TRAIN_PATH, TEST_PATH,
        FeatureConfig(k_best=K_BEST, random_state=RANDOM_STATE),
    )

    # -------- 2. 模型与超参数搜索 --------
    # 树模型不需要标准化。
    base_model = RandomForestRegressor(
        n_jobs=-1, random_state=RANDOM_STATE,
    )

    if USE_SEARCH:
        param_dist = {
            'n_estimators': [300, 600],
            'max_depth': [None, 15, 25, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 0.3, 0.5],
        }
        search = RandomizedSearchCV(
            base_model, param_distributions=param_dist, n_iter=SEARCH_N_ITER,
            scoring='r2', cv=KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=1, random_state=RANDOM_STATE, verbose=1, refit=True,
        )
        print('\n开始随机森林超参数随机搜索……')
        search.fit(features.X_train, features.y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        best_cv_r2 = float(search.best_score_)
        print(f'最佳 CV R² = {best_cv_r2:.4f}')
        print('最佳参数：', json.dumps(best_params, ensure_ascii=False))
    else:
        model = RandomForestRegressor(
            n_estimators=600, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, max_features='sqrt',
            n_jobs=-1, random_state=RANDOM_STATE,
        )
        model.fit(features.X_train, features.y_train)
        best_params = {
            'n_estimators': 600, 'max_depth': None,
            'min_samples_split': 2, 'min_samples_leaf': 1,
            'max_features': 'sqrt',
        }
        best_cv_r2 = None

    # -------- 3. 预测与保存 --------
    train_pred = model.predict(features.X_train)
    test_pred = model.predict(features.X_test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / 'rf_model.joblib')

    # 特征重要性（随机森林原生输出，便于解释模型）。
    importance = (
        pd.DataFrame({
            'feature': features.feature_names,
            'importance': model.feature_importances_,
        })
        .sort_values('importance', ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(
        OUTPUT_DIR / 'rf_feature_importance.csv', index=False, encoding='utf-8-sig'
    )

    finalize_model(
        'RF', OUTPUT_DIR,
        features.y_train, train_pred, features.y_test, test_pred,
        features,
        extra_metadata={
            'hyperparameters': best_params,
            'best_cv_r2': best_cv_r2,
            'random_state': RANDOM_STATE,
        },
    )


if __name__ == '__main__':
    main()
