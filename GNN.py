#!/usr/bin/env python3
"""GNN（GINE 图神经网络）预测聚酰亚胺玻璃化转变温度（Tg）。

运行：
    /home/albert/anaconda3/envs/gnn/bin/python \
        /home/albert/MD_LAMMPS/GNN_OA/ML_Tg_prediction/gnn_model.py

流程：
    SMILES -> 分子图（51 维原子特征、14 维键特征）
    -> 训练集内分层划分 80%/20% 训练/验证
    -> GINE（带键特征的消息传递、残差 + LayerNorm）
    -> 验证集早停选择轮数
    -> 用全部 883 条训练集重训最佳轮数
    -> 测试集只做一次最终评估

默认超参数取自本工程既有 GNN 调参实验的最优结果
（hidden=128、3 层、dropout=0、mean 读出、lr≈1.45e-3、batch=16）。
"""
from __future__ import annotations

import argparse
import copy
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

from graph_features import (
    ATOM_FEATURE_DIM, ATOM_FEATURE_NAMES, BOND_FEATURE_DIM,
    BOND_FEATURE_NAMES, read_dataset,
)
from model_utils import finalize_model, regression_metrics


# ==========================================================
# 1. 路径与默认超参数
# ==========================================================
TRAIN_PATH = Path(
    '/home/albert/MD_LAMMPS/1_in_design/pre_picture/OA_data/Tg_train.xlsx'
)
TEST_PATH = Path(
    '/home/albert/MD_LAMMPS/1_in_design/pre_picture/OA_data/Tg_test.xlsx'
)
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'gnn'

HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.0
READOUT = 'mean'
LEARNING_RATE = 0.0014451963957419053
WEIGHT_DECAY = 1.5459737788156094e-06
BATCH_SIZE = 16
EPOCHS = 300
PATIENCE = 30
MIN_DELTA = 1e-4
VAL_FRACTION = 0.2
SEED = 42
NUM_THREADS = 4


class GINERegressor(nn.Module):
    """带键特征、残差连接和 LayerNorm 的图级回归模型。"""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        h = self.node_encoder(data.x)
        for conv, norm in zip(self.convs, self.norms):
            update = F.relu(conv(h, data.edge_index, data.edge_attr))
            h = norm(h + F.dropout(update, p=self.dropout, training=self.training))
        batch = data.batch
        pooled = global_mean_pool(h, batch, size=int(data.num_graphs))
        return self.head(pooled).view(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(graphs, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        graphs, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, generator=generator,
    )


def fit_label_scaler(labels: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.float64)
    return {
        'mean': float(labels.mean()),
        'std': float(labels.std(ddof=0)),
        'count': int(labels.size),
    }


def train_epoch(model, loader, optimizer, scaler, device) -> float:
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        target = (batch.y.view(-1) - scaler['mean']) / scaler['std']
        loss = F.mse_loss(model(batch), target)
        if not torch.isfinite(loss):
            raise FloatingPointError('训练损失出现 NaN/Inf，请降低学习率。')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
        count += batch.num_graphs
    return total / count


@torch.inference_mode()
def predict(model, graphs, scaler, batch_size: int, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truths, predictions = [], []
    for batch in make_loader(graphs, batch_size, False, SEED):
        batch = batch.to(device)
        output = model(batch) * scaler['std'] + scaler['mean']
        predictions.append(output.cpu().numpy().astype(np.float64))
        truths.append(batch.y.view(-1).cpu().numpy().astype(np.float64))
    return np.concatenate(truths), np.concatenate(predictions)


def split_train_validation(labels: np.ndarray, fraction: float, seed: int):
    """按 Tg 分位分箱分层，保证验证集覆盖整个温度范围。"""
    bins = pd.qcut(labels, q=10, labels=False, duplicates='drop')
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices, test_size=fraction, random_state=seed, stratify=bins,
    )
    return np.sort(train_idx), np.sort(val_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description='GINE 预测聚酰亚胺 Tg')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                        help='本机 WSL 的 CUDA 内核不兼容，默认 cpu')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    if args.epochs < 1 or args.patience < 1:
        raise ValueError('epochs 和 patience 必须为正整数。')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('当前环境没有可用的 CUDA。')

    torch.set_num_threads(NUM_THREADS)
    seed_everything(args.seed)
    device = torch.device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------- 1. 图特征 --------
    print(f'读取训练集并构建分子图：{TRAIN_PATH}')
    train_graphs, train_labels, _ = read_dataset(TRAIN_PATH)
    print(f'读取测试集并构建分子图：{TEST_PATH}')
    test_graphs, test_labels, _ = read_dataset(TEST_PATH)
    print(f'训练集 {len(train_graphs)} 个图，测试集 {len(test_graphs)} 个图；'
          f'原子特征 {ATOM_FEATURE_DIM} 维，键特征 {BOND_FEATURE_DIM} 维')

    train_idx, val_idx = split_train_validation(
        train_labels, VAL_FRACTION, args.seed
    )
    inner_graphs = [train_graphs[i] for i in train_idx]
    val_graphs = [train_graphs[i] for i in val_idx]
    inner_labels = train_labels[train_idx]
    print(f'训练 {len(inner_graphs)} / 验证 {len(val_graphs)}（分层抽样）')

    # -------- 2. 验证阶段：早停选择轮数 --------
    scaler = fit_label_scaler(inner_labels)
    seed_everything(args.seed)
    model = GINERegressor(ATOM_FEATURE_DIM, BOND_FEATURE_DIM).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    train_loader = make_loader(inner_graphs, BATCH_SIZE, True, args.seed)

    best_rmse = math.inf
    best_epoch = 0
    best_state = None
    stalled = 0
    history = []

    print(f'\n开始训练（最多 {args.epochs} 轮，验证早停 patience={args.patience}）')
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        val_truth, val_pred = predict(
            model, val_graphs, scaler, BATCH_SIZE, device
        )
        val_rmse = float(np.sqrt(np.mean((val_pred - val_truth) ** 2)))
        history.append({
            'epoch': epoch, 'train_mse_standardized': loss,
            'val_RMSE_C': val_rmse, 'lr': float(optimizer.param_groups[0]['lr']),
        })
        if val_rmse < best_rmse - MIN_DELTA:
            best_rmse, best_epoch = val_rmse, epoch
            best_state = copy.deepcopy(model.state_dict())
            stalled = 0
        else:
            stalled += 1
        if epoch == 1 or epoch % 10 == 0 or stalled == args.patience:
            print(f'  epoch {epoch:3d}  train MSE {loss:.5f}  '
                  f'val RMSE {val_rmse:.3f} ℃  best {best_rmse:.3f} ℃')
        if stalled >= args.patience:
            print(f'  验证 RMSE 连续 {args.patience} 轮未改善，早停。')
            break

    if best_state is None:
        raise RuntimeError('训练没有产生可用模型。')
    model.load_state_dict(best_state)
    val_truth, val_pred = predict(model, val_graphs, scaler, BATCH_SIZE, device)
    val_metrics = regression_metrics(val_truth, val_pred)
    print(f'最佳验证轮数 {best_epoch}，验证 R²={val_metrics["R2"]:.4f}，'
          f'RMSE={val_metrics["RMSE_C"]:.3f} ℃')
    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / 'search_history.csv', index=False, encoding='utf-8-sig'
    )

    # -------- 3. 用全部训练集重训最佳轮数 --------
    print(f'\n用全部 {len(train_graphs)} 条训练集重训 {best_epoch} 轮……')
    seed_everything(args.seed)
    final_scaler = fit_label_scaler(train_labels)
    final_model = GINERegressor(ATOM_FEATURE_DIM, BOND_FEATURE_DIM).to(device)
    optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    full_loader = make_loader(train_graphs, BATCH_SIZE, True, args.seed)
    final_history = []
    for epoch in range(1, best_epoch + 1):
        loss = train_epoch(final_model, full_loader, optimizer, final_scaler, device)
        scheduler.step()
        final_history.append({
            'epoch': epoch, 'train_mse_standardized': loss,
            'lr': float(optimizer.param_groups[0]['lr']),
        })
        if epoch == 1 or epoch % 20 == 0 or epoch == best_epoch:
            print(f'  epoch {epoch:3d}/{best_epoch}  train MSE {loss:.5f}')
    pd.DataFrame(final_history).to_csv(
        OUTPUT_DIR / 'final_history.csv', index=False, encoding='utf-8-sig'
    )

    # -------- 4. 预测与保存 --------
    train_truth, train_pred = predict(
        final_model, train_graphs, final_scaler, BATCH_SIZE, device
    )
    test_truth, test_pred = predict(
        final_model, test_graphs, final_scaler, BATCH_SIZE, device
    )

    torch.save({
        'model_class': 'GINERegressor',
        'model_state_dict': final_model.state_dict(),
        'model_kwargs': {
            'node_dim': ATOM_FEATURE_DIM,
            'edge_dim': BOND_FEATURE_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
        },
        'label_scaler': final_scaler,
        'atom_feature_names': ATOM_FEATURE_NAMES,
        'bond_feature_names': BOND_FEATURE_NAMES,
        'best_epoch': best_epoch,
        'seed': args.seed,
    }, OUTPUT_DIR / 'gnn_model.pt')

    finalize_model(
        'GNN', OUTPUT_DIR,
        train_truth, train_pred, test_truth, test_pred,
        features=None,
        extra_metadata={
            'architecture': 'GINE (GINEConv + residual + LayerNorm + mean pool)',
            'atom_feature_dim': ATOM_FEATURE_DIM,
            'bond_feature_dim': BOND_FEATURE_DIM,
            'hyperparameters': {
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT,
                'readout': READOUT,
                'learning_rate': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
                'batch_size': BATCH_SIZE,
            },
            'max_epochs': args.epochs,
            'patience': args.patience,
            'val_fraction': VAL_FRACTION,
            'best_epoch': best_epoch,
            'validation_metrics': val_metrics,
            'final_label_scaler': final_scaler,
            'test_used_for_optimization': False,
            'random_state': args.seed,
            'device': str(device),
        },
    )


if __name__ == '__main__':
    main()
