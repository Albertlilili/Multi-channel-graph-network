#!/usr/bin/env python3
"""SVR 预测聚酰亚胺玻璃化转变温度（Tg）。

运行：
    /home/albert/anaconda3/envs/gnn/bin/python \
        /home/albert/MD_LAMMPS/GNN_OA/ML_Tg_prediction/svr_model.py

流程：RDKit 描述符 + Morgan 指纹 -> 训练集内筛选 -> 标准化 ->
      5 折交叉验证随机搜索 RBF-SVR 超参数 -> 训练集/测试集评估。
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from feature_pipeline import (
    DEFAULT_TEST_PATH, DEFAULT_TRAIN_PATH, FeatureConfig, build_features,
)
from model_utils import finalize_model


# ==========================================================
# 1. 路径与参数
# ==========================================================
TRAIN_PATH = Path(DEFAULT_TRAIN_PATH)
TEST_PATH = Path(DEFAULT_TEST_PATH)
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'svr'

RANDOM_STATE = 42
K_BEST = 300                 # 描述符 + 指纹筛选后保留的特征数
SEARCH_N_ITER = 40           # 随机搜索采样次数
CV_FOLDS = 5
USE_SEARCH = True            # 数据量小、追求复现时可设为 False 使用固定参数


def main() -> None:
    # -------- 1. 特征提取与筛选（仅训练集拟合筛选器） --------
    features = build_features(
        TRAIN_PATH, TEST_PATH,
        FeatureConfig(k_best=K_BEST, random_state=RANDOM_STATE),
    )

    # -------- 2. 模型与超参数搜索 --------
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf')),
    ])

    if USE_SEARCH:
        param_dist = {
            'svr__C': [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
            'svr__gamma': ['scale', 0.001, 0.003, 0.01, 0.03, 0.1],
            'svr__epsilon': [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        }
        search = RandomizedSearchCV(
            pipeline, param_distributions=param_dist, n_iter=SEARCH_N_ITER,
            scoring='r2', cv=KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=-1, random_state=RANDOM_STATE, verbose=1, refit=True,
        )
        print('\n开始 SVR 超参数随机搜索……')
        search.fit(features.X_train, features.y_train)
        model = search.best_estimator_
        best_params = {
            key.replace('svr__', ''): value
            for key, value in search.best_params_.items()
        }
        best_cv_r2 = float(search.best_score_)
        print(f'最佳 CV R² = {best_cv_r2:.4f}')
        print('最佳参数：', json.dumps(best_params, ensure_ascii=False))
    else:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=100.0, gamma='scale', epsilon=0.1)),
        ])
        model.fit(features.X_train, features.y_train)
        best_params = {'C': 100.0, 'gamma': 'scale', 'epsilon': 0.1}
        best_cv_r2 = None

    # -------- 3. 预测与保存 --------
    train_pred = model.predict(features.X_train)
    test_pred = model.predict(features.X_test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / 'svr_model.joblib')

    finalize_model(
        'SVR', OUTPUT_DIR,
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
