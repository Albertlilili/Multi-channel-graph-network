#!/usr/bin/env python3
"""高斯过程回归（GPR）预测聚酰亚胺玻璃化转变温度（Tg）。

运行：
    /home/albert/anaconda3/envs/gnn/bin/python \
        /home/albert/MD_LAMMPS/GNN_OA/ML_Tg_prediction/gpr_model.py

流程：RDKit 描述符 + Morgan 指纹 -> 训练集内筛选 -> 标准化 ->
      Matern 核 + WhiteKernel 的高斯过程回归（最大化边际似然） ->
      训练集/测试集评估，并输出测试集预测标准差。
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_pipeline import (
    DEFAULT_TEST_PATH, DEFAULT_TRAIN_PATH, FeatureConfig, build_features,
)
from model_utils import finalize_model


# ==========================================================
# 1. 路径与参数
# ==========================================================
TRAIN_PATH = Path(DEFAULT_TRAIN_PATH)
TEST_PATH = Path(DEFAULT_TEST_PATH)
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'gpr'

RANDOM_STATE = 42
# GPR 的计算量随特征维数增加而明显上升，这里用 200 维；
# 其余模型使用 300 维。若机器内存/时间充足可改为 300。
K_BEST = 200


def main() -> None:
    # -------- 1. 特征提取与筛选 --------
    features = build_features(
        TRAIN_PATH, TEST_PATH,
        FeatureConfig(k_best=K_BEST, random_state=RANDOM_STATE),
    )

    # -------- 2. 高斯过程模型 --------
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=10.0, length_scale_bounds=(1e-2, 1e3), nu=1.5)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2))
    )
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('gpr', GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=RANDOM_STATE,
        )),
    ])

    print('\n训练高斯过程回归……')
    model.fit(features.X_train, features.y_train)

    # -------- 3. 预测（测试集同时给出标准差） --------
    train_pred = model.predict(features.X_train)
    test_pred, test_std = model.predict(features.X_test, return_std=True)
    test_pred = np.asarray(test_pred).ravel()
    test_std = np.asarray(test_std).ravel()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / 'gpr_model.joblib')

    fitted_kernel = str(model.named_steps['gpr'].kernel_)
    print(f'拟合后的核函数：{fitted_kernel}')

    finalize_model(
        'GPR', OUTPUT_DIR,
        features.y_train, train_pred, features.y_test, test_pred,
        features,
        extra_metadata={
            'kernel_initial': str(kernel),
            'kernel_fitted': fitted_kernel,
            'normalize_y': True,
            'n_restarts_optimizer': 5,
            'random_state': RANDOM_STATE,
            'test_prediction_std_mean_C': float(np.mean(test_std)),
            'test_prediction_std_max_C': float(np.max(test_std)),
        },
        y_test_std=test_std,
    )


if __name__ == '__main__':
    main()
