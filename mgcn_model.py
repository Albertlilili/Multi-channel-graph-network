#!/usr/bin/env python3
"""多通道 GCN（MGCN，带注意力机制）预测聚酰亚胺玻璃化转变温度（Tg）。

三个通道图
----------
1. **节点图**：原子为节点、化学键为边（51 维原子特征、14 维键特征）。
2. **化学键图**：化学键为节点，共享原子的键之间连边；节点特征为键特征
   + 两端原子特征的和/绝对差，边特征为共享原子特征。
3. **子结构图**：稠合环系与非环连接链作为节点，子结构之间的连接键为边。

注意力机制
----------
* 每个通道内部使用 **节点级注意力池化**（softmax 权重聚合节点表示）；
* 三个通道的图级表示通过 **通道注意力** 自适应加权融合，再回归 Tg。

训练流程与 `gnn_model.py` 一致：训练集内按 Tg 分位分层划出 20% 验证集，
验证 RMSE 早停选出最佳轮数，再用全部训练集重训该轮数，测试集只评估一次。

运行：
    /home/albert/anaconda3/envs/gnn/bin/python \
        /home/albert/MD_LAMMPS/GNN_OA/ML_Tg_prediction/mgcn_model.py
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
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter, softmax

from graph_features import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from model_utils import finalize_model, regression_metrics
from multichannel_features import (
    BOND_GRAPH_EDGE_DIM, BOND_NODE_FEATURE_DIM, SUBSTRUCTURE_EDGE_DIM,
    SUBSTRUCTURE_FEATURE_DIM, read_multichannel_dataset,
)


# ==========================================================
# 1. 路径与默认超参数
# ==========================================================
TRAIN_PATH = Path(
    '/home/albert/MD_LAMMPS/1_in_design/pre_picture/OA_data/Tg_train.xlsx'
)
TEST_PATH = Path(
    '/home/albert/MD_LAMMPS/1_in_design/pre_picture/OA_data/Tg_test.xlsx'
)
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'mgcn'

HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.1
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 16
EPOCHS = 300
PATIENCE = 30
MIN_DELTA = 1e-4
VAL_FRACTION = 0.2
SEED = 42
NUM_THREADS = 4
CHANNEL_NAMES = ('node', 'bond', 'substructure')


class AttentionPooling(nn.Module):
    """节点级注意力池化：每个图内对节点表示做 softmax 加权求和。"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor,
                num_graphs: int) -> torch.Tensor:
        score = self.gate(x).view(-1)
        weights = softmax(score, batch, num_nodes=x.size(0))
        return scatter(
            weights.unsqueeze(-1) * x, batch, dim=0,
            dim_size=num_graphs, reduce='sum',
        )


class ChannelEncoder(nn.Module):
    """单个通道：边门控 GCN + 残差/LayerNorm + 注意力池化。"""

    def __init__(
        self, node_dim: int, edge_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.input = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList()
        self.edge_gates = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.edge_gates.append(nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.pool = AttentionPooling(hidden_dim)

    def forward(self, data) -> torch.Tensor:
        h = self.input(data.x)
        for conv, gate, norm in zip(self.convs, self.edge_gates, self.norms):
            edge_weight = torch.sigmoid(gate(data.edge_attr)).view(-1)
            update = conv(h, data.edge_index, edge_weight=edge_weight)
            h = norm(h + F.dropout(
                F.relu(update), p=self.dropout, training=self.training
            ))
        return self.pool(h, data.batch, int(data.num_graphs))


class MultiChannelGCN(nn.Module):
    """三通道 GCN + 通道注意力融合。"""

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.node_encoder = ChannelEncoder(
            ATOM_FEATURE_DIM, BOND_FEATURE_DIM, hidden_dim, num_layers, dropout
        )
        self.bond_encoder = ChannelEncoder(
            BOND_NODE_FEATURE_DIM, BOND_GRAPH_EDGE_DIM,
            hidden_dim, num_layers, dropout,
        )
        self.sub_encoder = ChannelEncoder(
            SUBSTRUCTURE_FEATURE_DIM, SUBSTRUCTURE_EDGE_DIM,
            hidden_dim, num_layers, dropout,
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node, bond, sub) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.stack([
            self.node_encoder(node),
            self.bond_encoder(bond),
            self.sub_encoder(sub),
        ], dim=1)                                   # [B, 3, H]
        scores = self.channel_gate(embeddings).squeeze(-1)
        weights = torch.softmax(scores, dim=1)      # [B, 3]
        fused = (weights.unsqueeze(-1) * embeddings).sum(dim=1)
        fused = self.fusion_norm(fused)
        return self.head(fused).view(-1), weights


# ==========================================================
# 2. 数据加载与训练工具
# ==========================================================
def collate_molecules(items: list[dict]) -> dict:
    """分别 batch 三个通道图，标签堆叠成一个向量。"""
    return {
        'node': Batch.from_data_list([item['node'] for item in items]),
        'bond': Batch.from_data_list([item['bond'] for item in items]),
        'sub': Batch.from_data_list([item['sub'] for item in items]),
        'y': torch.cat([item['y'] for item in items]).view(-1),
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def make_loader(graphs, batch_size: int, shuffle: bool,
                seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        graphs, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        generator=generator, collate_fn=collate_molecules,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        target = (batch['y'] - scaler['mean']) / scaler['std']
        prediction, _ = model(batch['node'], batch['bond'], batch['sub'])
        loss = F.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError('训练损失出现 NaN/Inf，请降低学习率。')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * batch['y'].numel()
        count += batch['y'].numel()
    return total / count


@torch.inference_mode()
def predict(model, graphs, scaler, batch_size: int, device):
    """返回真实值、预测值（℃）和每个样本的通道注意力权重。"""
    model.eval()
    truths, predictions, weights = [], [], []
    for batch in make_loader(graphs, batch_size, False, SEED):
        batch = move_batch(batch, device)
        prediction, channel_weights = model(
            batch['node'], batch['bond'], batch['sub']
        )
        predictions.append(
            (prediction * scaler['std'] + scaler['mean']).cpu().numpy().astype(
                np.float64
            )
        )
        truths.append(batch['y'].cpu().numpy().astype(np.float64))
        weights.append(channel_weights.cpu().numpy().astype(np.float64))
    return (
        np.concatenate(truths),
        np.concatenate(predictions),
        np.concatenate(weights),
    )


def split_train_validation(labels: np.ndarray, fraction: float, seed: int):
    bins = pd.qcut(labels, q=10, labels=False, duplicates='drop')
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices, test_size=fraction, random_state=seed, stratify=bins,
    )
    return np.sort(train_idx), np.sort(val_idx)


def save_channel_attention(output_dir: Path, split: str, truths: np.ndarray,
                           predictions: np.ndarray, weights: np.ndarray) -> None:
    table = pd.DataFrame({
        'y_true_C': truths,
        'y_pred_C': predictions,
        'error_C': predictions - truths,
        'w_node': weights[:, 0],
        'w_bond': weights[:, 1],
        'w_substructure': weights[:, 2],
    })
    table.to_csv(
        output_dir / f'channel_attention_{split}.csv',
        index=False, encoding='utf-8-sig',
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='多通道 GCN（节点图/化学键图/子结构图 + 注意力）预测 Tg'
    )
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

    # -------- 1. 三通道图特征 --------
    print(f'读取训练集并构建三通道图：{TRAIN_PATH}')
    train_graphs, train_labels, _ = read_multichannel_dataset(TRAIN_PATH)
    print(f'读取测试集并构建三通道图：{TEST_PATH}')
    test_graphs, test_labels, _ = read_multichannel_dataset(TEST_PATH)
    print(f'训练集 {len(train_graphs)} 个分子，测试集 {len(test_graphs)} 个分子')
    print(f'  节点通道：原子 {ATOM_FEATURE_DIM} 维 / 键 {BOND_FEATURE_DIM} 维')
    print(f'  化学键通道：键节点 {BOND_NODE_FEATURE_DIM} 维 / 边 {BOND_GRAPH_EDGE_DIM} 维')
    print(f'  子结构通道：节点 {SUBSTRUCTURE_FEATURE_DIM} 维 / 边 {SUBSTRUCTURE_EDGE_DIM} 维')

    node_counts = [g['node'].num_nodes for g in train_graphs]
    bond_counts = [g['bond'].num_nodes for g in train_graphs]
    sub_counts = [g['sub'].num_nodes for g in train_graphs]
    print(f'  平均节点数：原子图 {np.mean(node_counts):.1f}，'
          f'化学键图 {np.mean(bond_counts):.1f}，子结构图 {np.mean(sub_counts):.1f}')

    train_idx, val_idx = split_train_validation(
        train_labels, VAL_FRACTION, args.seed
    )
    inner_graphs = [train_graphs[i] for i in train_idx]
    val_graphs = [train_graphs[i] for i in val_idx]
    inner_labels = train_labels[train_idx]
    print(f'训练 {len(inner_graphs)} / 验证 {len(val_graphs)}（按 Tg 分位分层）')

    # -------- 2. 验证阶段：早停选择轮数 --------
    scaler = fit_label_scaler(inner_labels)
    seed_everything(args.seed)
    model = MultiChannelGCN().to(device)
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
        val_truth, val_pred, _ = predict(
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
    val_truth, val_pred, val_weights = predict(
        model, val_graphs, scaler, BATCH_SIZE, device
    )
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
    final_model = MultiChannelGCN().to(device)
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
    train_truth, train_pred, train_weights = predict(
        final_model, train_graphs, final_scaler, BATCH_SIZE, device
    )
    test_truth, test_pred, test_weights = predict(
        final_model, test_graphs, final_scaler, BATCH_SIZE, device
    )
    save_channel_attention(OUTPUT_DIR, 'train', train_truth, train_pred, train_weights)
    save_channel_attention(OUTPUT_DIR, 'test', test_truth, test_pred, test_weights)

    mean_weights = {
        'train': dict(zip(CHANNEL_NAMES, train_weights.mean(axis=0).tolist())),
        'test': dict(zip(CHANNEL_NAMES, test_weights.mean(axis=0).tolist())),
    }
    print('平均通道注意力权重：'
          + '，'.join(f'{name}={mean_weights["test"][name]:.3f}'
                     for name in CHANNEL_NAMES))

    torch.save({
        'model_class': 'MultiChannelGCN',
        'model_state_dict': final_model.state_dict(),
        'model_kwargs': {
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
        },
        'label_scaler': final_scaler,
        'channels': {
            'node': {'node_dim': ATOM_FEATURE_DIM, 'edge_dim': BOND_FEATURE_DIM},
            'bond': {'node_dim': BOND_NODE_FEATURE_DIM,
                     'edge_dim': BOND_GRAPH_EDGE_DIM},
            'substructure': {'node_dim': SUBSTRUCTURE_FEATURE_DIM,
                             'edge_dim': SUBSTRUCTURE_EDGE_DIM},
        },
        'best_epoch': best_epoch,
        'seed': args.seed,
    }, OUTPUT_DIR / 'mgcn_model.pt')

    finalize_model(
        'MGCN', OUTPUT_DIR,
        train_truth, train_pred, test_truth, test_pred,
        features=None,
        extra_metadata={
            'architecture': (
                'multi-channel GCN: atom graph + chemical-bond line graph + '
                'substructure graph; edge-gated GCNConv, attention pooling, '
                'channel attention fusion'
            ),
            'channels': {
                'node': {'node_dim': ATOM_FEATURE_DIM,
                         'edge_dim': BOND_FEATURE_DIM,
                         'graph': 'atoms as nodes, bonds as edges'},
                'bond': {'node_dim': BOND_NODE_FEATURE_DIM,
                         'edge_dim': BOND_GRAPH_EDGE_DIM,
                         'graph': 'bonds as nodes, bonds sharing an atom are adjacent'},
                'substructure': {'node_dim': SUBSTRUCTURE_FEATURE_DIM,
                                 'edge_dim': SUBSTRUCTURE_EDGE_DIM,
                                 'graph': 'fused ring systems and acyclic linkers as nodes'},
            },
            'mean_channel_attention': mean_weights,
            'hyperparameters': {
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT,
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
