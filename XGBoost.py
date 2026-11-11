#!/usr/bin/env python3
"""XGBoost 预测聚酰亚胺玻璃化转变温度（Tg）。

运行：
    /home/albert/anaconda3/envs/gnn/bin/python \
        /home/albert/MD_LAMMPS/GNN_OA/ML_Tg_prediction/xgboost_model.py

超参数采用论文给出的 best_param：
    n_estimators=280, max_depth=4, learning_rate=0.1,
    min_child_weight=1, gamma=0.1

流程：RDKit 描述符 + Morgan 指纹 -> 训练集内筛选 ->
      按上述参数训练 XGBRegressor -> 训练集/测试集评估。
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor

from feature_pipeline import (
    DEFAULT_TEST_PATH, DEFAULT_TRAIN_PATH, FeatureConfig, build_features,
)
from model_utils import finalize_model


# ==========================================================
# 1. 路径与参数
# ==========================================================
TRAIN_PATH = Path(DEFAULT_TRAIN_PATH)
TEST_PATH = Path(DEFAULT_TEST_PATH)
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'xgboost'

RANDOM_STATE = 42
K_BEST = 300

BEST_PARAM = {
    'n_estimators': 280,
    'max_depth': 4,
    'learning_rate': 0.1,
    'min_child_weight': 1,
    'gamma': 0.1,
}


def main() -> None:
    # -------- 1. 特征提取与筛选 --------
    features = build_features(
        TRAIN_PATH, TEST_PATH,
        FeatureConfig(k_best=K_BEST, random_state=RANDOM_STATE),
    )

    # -------- 2. 按论文参数训练 --------
    # 树模型不需要特征标准化。
    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **BEST_PARAM,
    )
    print('\nXGBoost 参数：', json.dumps(BEST_PARAM, ensure_ascii=False))
    model.fit(features.X_train, features.y_train)

    # -------- 3. 预测与保存 --------
    train_pred = model.predict(features.X_train)
    test_pred = model.predict(features.X_test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / 'xgboost_model.joblib')

    importance = (
        pd.DataFrame({
            'feature': features.feature_names,
            'importance': model.feature_importances_,
        })
        .sort_values('importance', ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(
        OUTPUT_DIR / 'xgboost_feature_importance.csv',
        index=False, encoding='utf-8-sig',
    )

    finalize_model(
        'XGBoost', OUTPUT_DIR,
        features.y_train, train_pred, features.y_test, test_pred,
        features,
        extra_metadata={
            'hyperparameters': BEST_PARAM,
            'objective': 'reg:squarederror',
            'random_state': RANDOM_STATE,
        },
    )


if __name__ == '__main__':
    main()
