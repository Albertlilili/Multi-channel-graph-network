#!/usr/bin/env python3
"""深度神经网络（DNN）预测聚酰亚胺玻璃化转变温度（Tg）。

运行：
    /home/albert/anaconda3/envs/gnn/bin/python \
        /home/albert/MD_LAMMPS/GNN_OA/ML_Tg_prediction/dnn_model.py

网络结构按论文参数设置：
    Dense(256, relu) -> Dropout(0.2)
    Dense(256, relu) -> Dropout(0.1)
    Dense(256, relu) -> Dropout(0.1)
    Dense(256, relu) -> Dropout(0.1)
    Dense(1, relu)
    Adam(lr=0.001), loss=mean_squared_error, epochs=350, batch_size=10

流程：RDKit 描述符 + Morgan 指纹 -> 训练集内筛选 ->
      X 标准化、y 归一化到 [0,1]（输出层 ReLU 要求非负目标）->
      训练 -> 反归一化后在原始 ℃ 尺度上评估。
"""
from __future__ import annotations

import os
from pathlib import Path

# 本机 WSL 的 TensorFlow GPU 句柄不可用，默认走 CPU。
# 需要 GPU 时先执行 export CUDA_VISIBLE_DEVICES=0 再运行本脚本。
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from feature_pipeline import (
    DEFAULT_TEST_PATH, DEFAULT_TRAIN_PATH, FeatureConfig, build_features,
)
from model_utils import finalize_model


# ==========================================================
# 1. 路径与参数
# ==========================================================
TRAIN_PATH = Path(DEFAULT_TRAIN_PATH)
TEST_PATH = Path(DEFAULT_TEST_PATH)
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'dnn'

RANDOM_STATE = 42
K_BEST = 300
EPOCHS = 350
BATCH_SIZE = 10
LEARNING_RATE = 0.001


def build_dnn(input_dim: int, output_bias: float) -> keras.Model:
    """按论文给出的层数与 Dropout 比例搭建网络。"""
    model = keras.Sequential(name='Tg_DNN')
    model.add(keras.Input(shape=(input_dim,), name='features'))

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.1))

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.1))

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.1))

    # 目标已 MinMax 缩放到 [0,1]，与 ReLU 输出匹配；
    # 偏置初值设为训练目标均值，避免 350 个 epoch 内输出层收敛过慢。
    model.add(layers.Dense(
        1, activation='relu',
        bias_initializer=keras.initializers.Constant(output_bias),
    ))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mean_squared_error',
        metrics=['mean_absolute_error'],
    )
    return model


def main() -> None:
    keras.utils.set_random_seed(RANDOM_STATE)

    # -------- 1. 特征提取与筛选 --------
    features = build_features(
        TRAIN_PATH, TEST_PATH,
        FeatureConfig(k_best=K_BEST, random_state=RANDOM_STATE),
    )

    # -------- 2. 特征与目标缩放 --------
    # 统计量只在训练集上拟合，测试集仅 transform。
    x_scaler = StandardScaler()
    y_scaler = MinMaxScaler()
    X_train = x_scaler.fit_transform(features.X_train)
    X_test = x_scaler.transform(features.X_test)
    y_train = y_scaler.fit_transform(features.y_train.reshape(-1, 1)).ravel()

    # -------- 3. 训练 --------
    model = build_dnn(X_train.shape[1], float(y_train.mean()))
    model.summary()

    print(f'\n开始训练 DNN：epochs={EPOCHS}, batch_size={BATCH_SIZE}')
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
    )

    # -------- 4. 预测并反归一化到 ℃ --------
    train_pred = y_scaler.inverse_transform(
        model.predict(X_train, verbose=0)
    ).ravel()
    test_pred = y_scaler.inverse_transform(
        model.predict(X_test, verbose=0)
    ).ravel()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_DIR / 'dnn_model.keras')

    history_frame = pd.DataFrame(history.history)
    history_frame.index.name = 'epoch'
    history_frame.to_csv(
        OUTPUT_DIR / 'training_history.csv', encoding='utf-8-sig'
    )

    finalize_model(
        'DNN', OUTPUT_DIR,
        features.y_train, train_pred, features.y_test, test_pred,
        features,
        extra_metadata={
            'architecture': [
                'Dense(256, relu)', 'Dropout(0.2)',
                'Dense(256, relu)', 'Dropout(0.1)',
                'Dense(256, relu)', 'Dropout(0.1)',
                'Dense(256, relu)', 'Dropout(0.1)',
                'Dense(1, relu)',
            ],
            'optimizer': 'Adam',
            'learning_rate': LEARNING_RATE,
            'loss': 'mean_squared_error',
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'x_scaler': 'StandardScaler',
            'y_scaler': 'MinMaxScaler([0,1])',
            'final_train_loss': float(history.history['loss'][-1]),
            'random_state': RANDOM_STATE,
        },
    )


if __name__ == '__main__':
    main()
