#!/usr/bin/env python3
"""多通道 GCN（MGCN，带注意力机制 + 5 折交叉验证遗传算法 GA 超参数优化）预测 Tg。

三个通道图（严格对齐 Table S1–S4）：
1. 原子图（Ga）：节点 51 维 / 边 14 维（质量与自由基电子数仅在训练折内标准化）。
2. 化学键图（Gb）：节点 116 维 / 边 51 维（同步使用折内标准化后的原子特征）。
3. 子结构图（Gs）：节点 21 维 / 边 35 维（Table S4 SMARTS + 未覆盖环/链 + 主链/边界特征）。

训练与评估流程（严格无数据泄漏）：
1. 在全部训练集上按 Tg 分位分层划分 5 折交叉验证。
2. 在训练集内部采用遗传算法（GA：种群大小 40、迭代代数 30、交叉率 0.8、变异率 0.1）
   优化超参数，以 5 折验证集平均 RMSE（℃）作为适应度指标（最小化）。
3. 每一折的原子连续特征缩放器与标签缩放器均仅在该折训练子集内拟合后再变换验证子集。
4. 确定最优超参数后，在 5 折上以完整训练轮数确认最佳轮数与 CV 指标，
   随后在全部训练集上重新拟合缩放器并重训最佳轮数，最后在独立测试集上一次性评估。

运行：
    /home/albert/anaconda3/envs/gnn/bin/python \
        /home/albert/MD_LAMMPS/GNN_OA/ML_Tg_prediction/mgcn_model.py
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import scatter, softmax

from graph_features import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from model_utils import finalize_model, regression_metrics
from multichannel_features import (
    BOND_GRAPH_EDGE_DIM,
    BOND_NODE_FEATURE_DIM,
    SUBSTRUCTURE_EDGE_DIM,
    SUBSTRUCTURE_FEATURE_DIM,
    apply_multichannel_scalers,
    fit_multichannel_scalers,
    read_multichannel_dataset,
)


# ==========================================================
# 1. 路径与默认超参数 / GA 配置
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / 'Tg_train.xlsx'
TEST_PATH = BASE_DIR / 'Tg_test.xlsx'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'mgcn'

CV_FOLDS = 5
GA_POP_SIZE = 40
GA_GENERATIONS = 30
GA_CROSSOVER_RATE = 0.8
GA_MUTATION_RATE = 0.1
GA_ELITE_COUNT = 2
GA_TOURNAMENT_K = 3

GA_EPOCHS = 60
GA_PATIENCE = 10
FINAL_MAX_EPOCHS = 300
FINAL_PATIENCE = 30
MIN_DELTA = 1e-4
SEED = 42
NUM_THREADS = 4
CHANNEL_NAMES = ('node', 'bond', 'substructure')

# GA 离散超参数搜索空间
GA_SEARCH_SPACE: dict[str, tuple] = {
    'hidden_dim': (64, 96, 128, 192),
    'num_layers': (2, 3, 4),
    'dropout': (0.0, 0.1, 0.2, 0.3),
    'learning_rate': (5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3),
    'weight_decay': (0.0, 1e-6, 1e-5, 1e-4),
    'batch_size': (64, 128, 256),
}
GENE_NAMES = tuple(GA_SEARCH_SPACE.keys())


# ==========================================================
# 2. 网络结构
# ==========================================================
class EdgeGatedGCNConv(nn.Module):
    """边门控图卷积层（带自环和对称度归一化），纯原生 Tensor 散射聚合。"""

    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        n = x.size(0)
        hx = self.lin(x)
        if edge_index.numel() == 0:
            return hx + self.bias
        src, dst = edge_index[0], edge_index[1]
        edge_w = torch.sigmoid(self.edge_gate(edge_attr)).view(-1)
        deg = torch.ones(n, dtype=x.dtype, device=x.device).scatter_add(0, dst, edge_w)
        deg_inv_sqrt = deg.clamp(min=1e-12).rsqrt()
        norm_w = edge_w * deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        out = hx * deg_inv_sqrt.pow(2).unsqueeze(-1)
        out = out.scatter_add(
            0, dst.unsqueeze(-1).expand(-1, hx.size(-1)), hx[src] * norm_w.unsqueeze(-1)
        )
        return out + self.bias


class AttentionPooling(nn.Module):
    """节点级注意力池化：每个图内对节点表示做 softmax 加权求和。"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        score = self.gate(x).view(-1)
        weights = softmax(score, batch, num_nodes=x.size(0))
        return scatter(
            weights.unsqueeze(-1) * x, batch, dim=0, dim_size=num_graphs, reduce='sum',
        )


class ChannelEncoder(nn.Module):
    """单个通道：边门控 GCN + 残差/LayerNorm + 节点注意力池化。"""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.input = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EdgeGatedGCNConv(hidden_dim, edge_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.pool = AttentionPooling(hidden_dim)

    def forward(self, data) -> torch.Tensor:
        h = self.input(data.x)
        for conv, norm in zip(self.convs, self.norms):
            update = conv(h, data.edge_index, data.edge_attr)
            h = norm(h + F.dropout(F.relu(update), p=self.dropout, training=self.training))
        return self.pool(h, data.batch, int(data.num_graphs))


class MultiChannelGCN(nn.Module):
    """三通道 GCN + 通道注意力融合。"""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
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
# 3. 批处理、缩放器与训练/预测工具
# ==========================================================
def collate_molecules(items: list[dict]) -> dict:
    return {
        'node': Batch.from_data_list([item['node'] for item in items]),
        'bond': Batch.from_data_list([item['bond'] for item in items]),
        'sub': Batch.from_data_list([item['sub'] for item in items]),
        'y': torch.cat([item['y'] for item in items]).view(-1),
    }


def build_prebatched_chunks(
    graphs: list[dict], batch_size: int, device: torch.device, seed: int | None = None,
) -> list[dict]:
    """预打包批处理到目标设备，消除每轮重复 collate 与 H2D 拷贝开销。"""
    items = list(graphs)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(items)
    batches = []
    for start in range(0, len(items), batch_size):
        batch = collate_molecules(items[start:start + batch_size])
        batches.append({k: v.to(device) for k, v in batch.items()})
    return batches


def make_loader(graphs, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
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
    std = float(labels.std(ddof=0))
    return {
        'mean': float(labels.mean()),
        'std': std if std > 1e-12 else 1.0,
        'count': int(labels.size),
    }


def train_epoch_batches(model, batches: list[dict], optimizer, scaler: dict[str, float],
                        rng: random.Random) -> float:
    model.train()
    order = list(range(len(batches)))
    rng.shuffle(order)
    device = batches[0]['y'].device
    total_loss = torch.zeros((), device=device)
    count = 0
    for idx in order:
        batch = batches[idx]
        optimizer.zero_grad(set_to_none=True)
        target = (batch['y'] - scaler['mean']) / scaler['std']
        prediction, _ = model(batch['node'], batch['bond'], batch['sub'])
        loss = F.mse_loss(prediction, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        n_items = batch['y'].numel()
        total_loss = total_loss + loss.detach() * n_items
        count += n_items
    avg_loss = float((total_loss / max(count, 1)).item())
    if not math.isfinite(avg_loss):
        raise FloatingPointError('训练损失出现 NaN/Inf。')
    return avg_loss


@torch.inference_mode()
def evaluate_batches_rmse(model, batches: list[dict], scaler: dict[str, float]) -> float:
    model.eval()
    device = batches[0]['y'].device
    sq_err = torch.zeros((), dtype=torch.float64, device=device)
    count = 0
    for batch in batches:
        prediction, _ = model(batch['node'], batch['bond'], batch['sub'])
        pred_c = prediction.to(torch.float64) * scaler['std'] + scaler['mean']
        diff = pred_c - batch['y'].to(torch.float64)
        sq_err = sq_err + (diff * diff).sum()
        count += batch['y'].numel()
    return float(torch.sqrt(sq_err / max(count, 1)).item())


@torch.inference_mode()
def predict_batches(model, batches: list[dict], scaler: dict[str, float]):
    model.eval()
    truths, predictions, weights = [], [], []
    for batch in batches:
        prediction, channel_weights = model(
            batch['node'], batch['bond'], batch['sub']
        )
        predictions.append(
            (prediction * scaler['std'] + scaler['mean']).cpu().numpy().astype(np.float64)
        )
        truths.append(batch['y'].cpu().numpy().astype(np.float64))
        weights.append(channel_weights.cpu().numpy().astype(np.float64))
    return (
        np.concatenate(truths),
        np.concatenate(predictions),
        np.concatenate(weights),
    )


def make_stratified_kfold_splits(
    labels: np.ndarray, n_splits: int, seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    bins = pd.qcut(labels, q=10, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(np.sort(tr), np.sort(va)) for tr, va in skf.split(np.zeros(len(labels)), bins)]


# ==========================================================
# 4. 单折训练与 5 折 CV 评估
# ==========================================================
def prepare_cv_folds(
    train_graphs_raw: list[dict],
    train_labels: np.ndarray,
    n_splits: int,
    seed: int,
    device: torch.device,
) -> list[dict]:
    """为每一折分别在训练子集上拟合原子与标签缩放器，避免任何验证/测试泄漏。"""
    splits = make_stratified_kfold_splits(train_labels, n_splits, seed)
    folds = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        tr_raw = [train_graphs_raw[i] for i in tr_idx]
        va_raw = [train_graphs_raw[i] for i in va_idx]
        atom_scaler = fit_multichannel_scalers(tr_raw)
        label_scaler = fit_label_scaler(train_labels[tr_idx])
        tr_scaled = apply_multichannel_scalers(tr_raw, atom_scaler)
        va_scaled = apply_multichannel_scalers(va_raw, atom_scaler)
        # 验证集不改变批次切分；可预打包到设备
        val_batches = build_prebatched_chunks(va_scaled, batch_size=256, device=device)
        folds.append({
            'fold': fold_idx,
            'train_graphs': tr_scaled,
            'val_graphs': va_scaled,
            'label_scaler': label_scaler,
            'val_batches': val_batches,
            'train_batch_cache': {},
        })
    return folds


def get_fold_train_batches(
    fold_data: dict, batch_size: int, device: torch.device, seed: int,
) -> list[dict]:
    if batch_size not in fold_data['train_batch_cache']:
        fold_data['train_batch_cache'][batch_size] = build_prebatched_chunks(
            fold_data['train_graphs'], batch_size=batch_size, device=device, seed=seed
        )
    return fold_data['train_batch_cache'][batch_size]


def train_single_fold(
    fold_data: dict,
    params: dict,
    max_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    return_predictions: bool = False,
) -> dict:
    fold_seed = seed + fold_data['fold'] * 101
    seed_everything(fold_seed)
    rng = random.Random(fold_seed)

    model = MultiChannelGCN(
        hidden_dim=int(params['hidden_dim']),
        num_layers=int(params['num_layers']),
        dropout=float(params['dropout']),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params['learning_rate']),
        weight_decay=float(params['weight_decay']),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    train_batches = get_fold_train_batches(
        fold_data, int(params['batch_size']), device, fold_seed
    )
    val_batches = fold_data['val_batches']
    scaler = fold_data['label_scaler']

    best_rmse = math.inf
    best_epoch = 1
    best_state = None
    stalled = 0

    for epoch in range(1, max_epochs + 1):
        try:
            train_epoch_batches(model, train_batches, optimizer, scaler, rng)
        except FloatingPointError:
            return {'best_rmse': 1e6, 'best_epoch': epoch, 'failed': True}
        scheduler.step()
        val_rmse = evaluate_batches_rmse(model, val_batches, scaler)
        if not math.isfinite(val_rmse):
            return {'best_rmse': 1e6, 'best_epoch': epoch, 'failed': True}
        if val_rmse < best_rmse - MIN_DELTA:
            best_rmse = val_rmse
            best_epoch = epoch
            if return_predictions:
                best_state = copy.deepcopy(model.state_dict())
            stalled = 0
        else:
            stalled += 1
        if stalled >= patience:
            break

    result = {'best_rmse': best_rmse, 'best_epoch': best_epoch, 'failed': False}
    if return_predictions and best_state is not None:
        model.load_state_dict(best_state)
        val_truth, val_pred, _ = predict_batches(model, val_batches, scaler)
        metrics = regression_metrics(val_truth, val_pred)
        result['metrics'] = metrics
    return result


def evaluate_chromosome_cv(
    chromosome: tuple[int, ...],
    folds: list[dict],
    max_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[float, list[float], list[int]]:
    params = decode_chromosome(chromosome)
    fold_rmses: list[float] = []
    fold_epochs: list[int] = []
    for fold_data in folds:
        res = train_single_fold(
            fold_data, params, max_epochs=max_epochs, patience=patience,
            seed=seed, device=device, return_predictions=False,
        )
        fold_rmses.append(res['best_rmse'])
        fold_epochs.append(res['best_epoch'])
    return float(np.mean(fold_rmses)), fold_rmses, fold_epochs


# ==========================================================
# 5. 遗传算法（GA）超参数优化
# ==========================================================
def decode_chromosome(chrom: tuple[int, ...]) -> dict:
    return {
        name: GA_SEARCH_SPACE[name][gene_idx]
        for name, gene_idx in zip(GENE_NAMES, chrom)
    }


def random_chromosome(rng: random.Random) -> tuple[int, ...]:
    return tuple(rng.randrange(len(GA_SEARCH_SPACE[name])) for name in GENE_NAMES)


def tournament_select(
    population: list[tuple[int, ...]], fitnesses: list[float],
    k: int, rng: random.Random,
) -> tuple[int, ...]:
    chosen = rng.sample(range(len(population)), k=min(k, len(population)))
    best_idx = min(chosen, key=lambda idx: fitnesses[idx])
    return population[best_idx]


def crossover_chromosomes(
    p1: tuple[int, ...], p2: tuple[int, ...],
    crossover_rate: float, rng: random.Random,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if rng.random() >= crossover_rate:
        return p1, p2
    c1, c2 = [], []
    for g1, g2 in zip(p1, p2):
        if rng.random() < 0.5:
            c1.append(g1)
            c2.append(g2)
        else:
            c1.append(g2)
            c2.append(g1)
    return tuple(c1), tuple(c2)


def mutate_chromosome(
    chrom: tuple[int, ...], mutation_rate: float, rng: random.Random,
) -> tuple[int, ...]:
    genes = list(chrom)
    if rng.random() < mutation_rate:
        pos = rng.randrange(len(genes))
        choices = [i for i in range(len(GA_SEARCH_SPACE[GENE_NAMES[pos]])) if i != genes[pos]]
        if choices:
            genes[pos] = rng.choice(choices)
    return tuple(genes)


def run_ga_search(
    folds: list[dict],
    output_dir: Path,
    pop_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    ga_epochs: int,
    ga_patience: int,
    seed: int,
    device: torch.device,
    resume: bool = False,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    ckpt_path = output_dir / 'ga_checkpoint.json'
    fitness_cache: dict[tuple[int, ...], dict] = {}
    history_rows: list[dict] = []
    start_gen = 0
    population: list[tuple[int, ...]] = []

    if resume and ckpt_path.is_file():
        ckpt = json.loads(ckpt_path.read_text(encoding='utf-8'))
        for item in ckpt['cache']:
            fitness_cache[tuple(item['chrom'])] = item['record']
        history_rows = ckpt['history']
        population = [tuple(c) for c in ckpt['population']]
        start_gen = int(ckpt['next_generation'])
        print(f'从断点恢复 GA：已完成 {start_gen}/{generations} 代，缓存 {len(fitness_cache)} 组超参数')

    if not population:
        # 初始化第 0 代种群，尽量保证初始多样性
        seen_init: set[tuple[int, ...]] = set()
        while len(population) < pop_size:
            c = random_chromosome(rng)
            if c not in seen_init or len(seen_init) >= 200:
                seen_init.add(c)
                population.append(c)

    def eval_pop(pop: list[tuple[int, ...]], gen_label: str) -> list[float]:
        fits: list[float] = []
        new_evals = 0
        for idx, chrom in enumerate(pop):
            if chrom not in fitness_cache:
                new_evals += 1
                t0 = time.perf_counter()
                mean_rmse, fold_rmses, fold_epochs = evaluate_chromosome_cv(
                    chrom, folds, ga_epochs, ga_patience, seed, device
                )
                dt = time.perf_counter() - t0
                params = decode_chromosome(chrom)
                fitness_cache[chrom] = {
                    'mean_cv_rmse': mean_rmse,
                    'fold_rmses': fold_rmses,
                    'fold_epochs': fold_epochs,
                    'eval_seconds': dt,
                    'params': params,
                }
                print(
                    f'  [{gen_label}] 个体 {idx+1:2d}/{len(pop)} '
                    f'(新评估 #{len(fitness_cache):3d}, {dt:4.1f}s) '
                    f'CV RMSE={mean_rmse:.3f} ℃ | {params}',
                    flush=True,
                )
            fits.append(fitness_cache[chrom]['mean_cv_rmse'])
        if new_evals == 0:
            print(f'  [{gen_label}] 所有个体均命中缓存，跳过重复训练。', flush=True)
        return fits

    if start_gen == 0:
        print(f'\n=== 遗传算法（GA）超参数寻优开始 ===')
        print(f'  种群大小={pop_size}，迭代代数={generations}，'
              f'交叉率={crossover_rate}，变异率={mutation_rate}，5折CV自适应度=平均RMSE')
        fitnesses = eval_pop(population, 'Gen 00')
        best_idx = int(np.argmin(fitnesses))
        history_rows.append({
            'generation': 0,
            'best_cv_rmse': float(fitnesses[best_idx]),
            'mean_cv_rmse': float(np.mean(fitnesses)),
            'worst_cv_rmse': float(np.max(fitnesses)),
            'unique_evaluated': len(fitness_cache),
            'best_params': json.dumps(decode_chromosome(population[best_idx]), ensure_ascii=False),
        })
        print(f'>> Gen 00 完成: 最优 5折 CV RMSE = {fitnesses[best_idx]:.4f} ℃ '
              f'(均值 {np.mean(fitnesses):.4f} ℃)', flush=True)
        start_gen = 1
        ckpt_payload = {
            'next_generation': 1,
            'population': [list(c) for c in population],
            'history': history_rows,
            'cache': [{'chrom': list(k), 'record': v} for k, v in fitness_cache.items()],
        }
        ckpt_path.write_text(json.dumps(ckpt_payload, ensure_ascii=False), encoding='utf-8')
    else:
        fitnesses = [fitness_cache[c]['mean_cv_rmse'] for c in population]

    for gen in range(start_gen, generations + 1):
        # 精英保留
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i])
        next_pop: list[tuple[int, ...]] = [population[i] for i in ranked[:GA_ELITE_COUNT]]

        while len(next_pop) < pop_size:
            p1 = tournament_select(population, fitnesses, GA_TOURNAMENT_K, rng)
            p2 = tournament_select(population, fitnesses, GA_TOURNAMENT_K, rng)
            c1, c2 = crossover_chromosomes(p1, p2, crossover_rate, rng)
            c1 = mutate_chromosome(c1, mutation_rate, rng)
            next_pop.append(c1)
            if len(next_pop) < pop_size:
                c2 = mutate_chromosome(c2, mutation_rate, rng)
                next_pop.append(c2)

        population = next_pop
        fitnesses = eval_pop(population, f'Gen {gen:02d}')
        best_idx = int(np.argmin(fitnesses))
        history_rows.append({
            'generation': gen,
            'best_cv_rmse': float(fitnesses[best_idx]),
            'mean_cv_rmse': float(np.mean(fitnesses)),
            'worst_cv_rmse': float(np.max(fitnesses)),
            'unique_evaluated': len(fitness_cache),
            'best_params': json.dumps(decode_chromosome(population[best_idx]), ensure_ascii=False),
        })
        print(
            f'>> Gen {gen:02d}/{generations:02d} 完成: '
            f'最优 5折 CV RMSE = {fitnesses[best_idx]:.4f} ℃ '
            f'(均值 {np.mean(fitnesses):.4f} ℃, 累计唯一配置 {len(fitness_cache)})',
            flush=True,
        )
        ckpt_payload = {
            'next_generation': gen + 1,
            'population': [list(c) for c in population],
            'history': history_rows,
            'cache': [{'chrom': list(k), 'record': v} for k, v in fitness_cache.items()],
        }
        ckpt_path.write_text(json.dumps(ckpt_payload, ensure_ascii=False), encoding='utf-8')

    best_chrom = min(fitness_cache.keys(), key=lambda k: fitness_cache[k]['mean_cv_rmse'])
    best_record = fitness_cache[best_chrom]

    ga_history_df = pd.DataFrame(history_rows)
    eval_rows = []
    for chrom, rec in sorted(fitness_cache.items(), key=lambda kv: kv[1]['mean_cv_rmse']):
        row = {**rec['params']}
        row['mean_cv_rmse_C'] = rec['mean_cv_rmse']
        row['fold_rmses_C'] = json.dumps([round(x, 4) for x in rec['fold_rmses']])
        row['fold_epochs'] = json.dumps(rec['fold_epochs'])
        row['eval_seconds'] = round(rec['eval_seconds'], 2)
        eval_rows.append(row)
    ga_evals_df = pd.DataFrame(eval_rows)

    return best_record['params'], ga_history_df, ga_evals_df


def save_channel_attention(
    output_dir: Path, split: str, truths: np.ndarray,
    predictions: np.ndarray, weights: np.ndarray,
) -> None:
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


# ==========================================================
# 6. 主流程
# ==========================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description='多通道 GCN（Table S1–S4 三通道 + 5折CV遗传算法GA寻优）预测 Tg'
    )
    parser.add_argument('--pop-size', type=int, default=GA_POP_SIZE)
    parser.add_argument('--generations', type=int, default=GA_GENERATIONS)
    parser.add_argument('--crossover-rate', type=float, default=GA_CROSSOVER_RATE)
    parser.add_argument('--mutation-rate', type=float, default=GA_MUTATION_RATE)
    parser.add_argument('--ga-epochs', type=int, default=GA_EPOCHS)
    parser.add_argument('--ga-patience', type=int, default=GA_PATIENCE)
    parser.add_argument('--final-max-epochs', type=int, default=FINAL_MAX_EPOCHS)
    parser.add_argument('--final-patience', type=int, default=FINAL_PATIENCE)
    parser.add_argument(
        '--device', choices=['cpu', 'cuda'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--resume', action='store_true', help='从 ga_checkpoint.json 恢复')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('当前环境没有可用的 CUDA。')

    torch.set_num_threads(NUM_THREADS)
    seed_everything(args.seed)
    device = torch.device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------- 1. 读取数据与构造原始三通道图（未标准化原子标量） --------
    print(f'使用设备：{device}')
    print(f'读取训练集并构建三通道图：{TRAIN_PATH}')
    train_graphs_raw, train_labels, _ = read_multichannel_dataset(TRAIN_PATH)
    print(f'读取测试集并构建三通道图：{TEST_PATH}')
    test_graphs_raw, test_labels, _ = read_multichannel_dataset(TEST_PATH)
    print(f'训练集 {len(train_graphs_raw)} 个分子，测试集 {len(test_graphs_raw)} 个分子')
    print(f'  原子图 Ga：节点 {ATOM_FEATURE_DIM} 维 / 边 {BOND_FEATURE_DIM} 维')
    print(f'  化学键图 Gb：节点 {BOND_NODE_FEATURE_DIM} 维 / 边 {BOND_GRAPH_EDGE_DIM} 维')
    print(f'  子结构图 Gs：节点 {SUBSTRUCTURE_FEATURE_DIM} 维 / 边 {SUBSTRUCTURE_EDGE_DIM} 维')

    # -------- 2. 训练集内 5 折交叉验证准备（折内独立拟合缩放器） --------
    folds = prepare_cv_folds(
        train_graphs_raw, train_labels, CV_FOLDS, args.seed, device
    )

    # -------- 3. 遗传算法（GA）寻优超参数 --------
    best_params, ga_history_df, ga_evals_df = run_ga_search(
        folds=folds,
        output_dir=OUTPUT_DIR,
        pop_size=args.pop_size,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        ga_epochs=args.ga_epochs,
        ga_patience=args.ga_patience,
        seed=args.seed,
        device=device,
        resume=args.resume,
    )
    ga_history_df.to_csv(OUTPUT_DIR / 'ga_history.csv', index=False, encoding='utf-8-sig')
    ga_evals_df.to_csv(OUTPUT_DIR / 'ga_evaluations.csv', index=False, encoding='utf-8-sig')
    print(f'\nGA 最优超参数：{json.dumps(best_params, ensure_ascii=False)}')

    # -------- 4. 用完整轮数在 5 折上确认最优超参数的 CV 指标与最佳训练轮数 --------
    print(f'\n使用最优超参数进行 5 折完整评估（max_epochs={args.final_max_epochs}, patience={args.final_patience}）……')
    cv_rows = []
    for fold_data in folds:
        res = train_single_fold(
            fold_data, best_params,
            max_epochs=args.final_max_epochs,
            patience=args.final_patience,
            seed=args.seed,
            device=device,
            return_predictions=True,
        )
        m = res['metrics']
        cv_rows.append({
            'fold': fold_data['fold'] + 1,
            'best_epoch': res['best_epoch'],
            'val_R2': m['R2'],
            'val_RMSE_C': m['RMSE_C'],
            'val_MAE_C': m['MAE_C'],
        })
        print(f"  Fold {fold_data['fold']+1}: best_epoch={res['best_epoch']:3d}  "
              f"R²={m['R2']:.4f}  RMSE={m['RMSE_C']:.3f} ℃  MAE={m['MAE_C']:.3f} ℃")
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(OUTPUT_DIR / 'cv_best_fold_metrics.csv', index=False, encoding='utf-8-sig')
    cv_mean_rmse = float(cv_df['val_RMSE_C'].mean())
    cv_mean_r2 = float(cv_df['val_R2'].mean())
    retrain_epochs = max(1, int(round(float(cv_df['best_epoch'].mean()))))
    print(f'5 折平均：R²={cv_mean_r2:.4f}，RMSE={cv_mean_rmse:.3f} ℃，'
          f'重训轮数={retrain_epochs}')

    # -------- 5. 在完整训练集上重新拟合预处理缩放器并重训模型 --------
    print(f'\n在全部 {len(train_graphs_raw)} 个训练结构上重新拟合缩放器并重训 {retrain_epochs} 轮……')
    final_atom_scaler = fit_multichannel_scalers(train_graphs_raw)
    final_label_scaler = fit_label_scaler(train_labels)
    train_graphs_scaled = apply_multichannel_scalers(train_graphs_raw, final_atom_scaler)
    test_graphs_scaled = apply_multichannel_scalers(test_graphs_raw, final_atom_scaler)

    seed_everything(args.seed)
    rng = random.Random(args.seed)
    final_model = MultiChannelGCN(
        hidden_dim=int(best_params['hidden_dim']),
        num_layers=int(best_params['num_layers']),
        dropout=float(best_params['dropout']),
    ).to(device)
    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=float(best_params['learning_rate']),
        weight_decay=float(best_params['weight_decay']),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(retrain_epochs, 1)
    )
    full_train_batches = build_prebatched_chunks(
        train_graphs_scaled, batch_size=int(best_params['batch_size']),
        device=device, seed=args.seed,
    )

    final_history = []
    for epoch in range(1, retrain_epochs + 1):
        loss = train_epoch_batches(
            final_model, full_train_batches, optimizer, final_label_scaler, rng
        )
        scheduler.step()
        final_history.append({
            'epoch': epoch,
            'train_mse_standardized': loss,
            'lr': float(optimizer.param_groups[0]['lr']),
        })
        if epoch == 1 or epoch % 20 == 0 or epoch == retrain_epochs:
            print(f'  epoch {epoch:3d}/{retrain_epochs}  train MSE {loss:.5f}')
    pd.DataFrame(final_history).to_csv(
        OUTPUT_DIR / 'final_history.csv', index=False, encoding='utf-8-sig'
    )

    # -------- 6. 独立测试集一次性评估与保存 --------
    train_eval_batches = build_prebatched_chunks(train_graphs_scaled, batch_size=256, device=device)
    test_eval_batches = build_prebatched_chunks(test_graphs_scaled, batch_size=256, device=device)
    train_truth, train_pred, train_weights = predict_batches(
        final_model, train_eval_batches, final_label_scaler
    )
    test_truth, test_pred, test_weights = predict_batches(
        final_model, test_eval_batches, final_label_scaler
    )
    save_channel_attention(OUTPUT_DIR, 'train', train_truth, train_pred, train_weights)
    save_channel_attention(OUTPUT_DIR, 'test', test_truth, test_pred, test_weights)

    mean_weights = {
        'train': dict(zip(CHANNEL_NAMES, train_weights.mean(axis=0).tolist())),
        'test': dict(zip(CHANNEL_NAMES, test_weights.mean(axis=0).tolist())),
    }
    print('平均通道注意力权重（测试集）：'
          + '，'.join(f'{name}={mean_weights["test"][name]:.3f}' for name in CHANNEL_NAMES))

    torch.save({
        'model_class': 'MultiChannelGCN',
        'model_state_dict': final_model.state_dict(),
        'model_kwargs': {
            'hidden_dim': int(best_params['hidden_dim']),
            'num_layers': int(best_params['num_layers']),
            'dropout': float(best_params['dropout']),
        },
        'best_hyperparameters': best_params,
        'atom_scaler': final_atom_scaler,
        'label_scaler': final_label_scaler,
        'channels': {
            'node': {'node_dim': ATOM_FEATURE_DIM, 'edge_dim': BOND_FEATURE_DIM},
            'bond': {'node_dim': BOND_NODE_FEATURE_DIM, 'edge_dim': BOND_GRAPH_EDGE_DIM},
            'substructure': {'node_dim': SUBSTRUCTURE_FEATURE_DIM, 'edge_dim': SUBSTRUCTURE_EDGE_DIM},
        },
        'retrain_epochs': retrain_epochs,
        'seed': args.seed,
    }, OUTPUT_DIR / 'mgcn_model.pt')

    finalize_model(
        'MGCN', OUTPUT_DIR,
        train_truth, train_pred, test_truth, test_pred,
        features=None,
        extra_metadata={
            'architecture': (
                'multi-channel GCN (Table S1-S4): atomic graph (51/14) + '
                'chemical bond graph (116/51) + substructure graph (21/35); '
                'edge-gated GCN, node attention pooling, channel attention fusion'
            ),
            'channels': {
                'node': {'node_dim': ATOM_FEATURE_DIM, 'edge_dim': BOND_FEATURE_DIM},
                'bond': {'node_dim': BOND_NODE_FEATURE_DIM, 'edge_dim': BOND_GRAPH_EDGE_DIM},
                'substructure': {'node_dim': SUBSTRUCTURE_FEATURE_DIM, 'edge_dim': SUBSTRUCTURE_EDGE_DIM},
            },
            'ga_config': {
                'cv_folds': CV_FOLDS,
                'population_size': args.pop_size,
                'generations': args.generations,
                'crossover_rate': args.crossover_rate,
                'mutation_rate': args.mutation_rate,
                'fitness_metric': '5-fold CV mean RMSE (C)',
                'ga_epochs': args.ga_epochs,
                'ga_patience': args.ga_patience,
                'unique_configurations_evaluated': int(len(ga_evals_df)),
            },
            'hyperparameters': best_params,
            'cv_5fold_summary': {
                'mean_R2': cv_mean_r2,
                'mean_RMSE_C': cv_mean_rmse,
                'mean_MAE_C': float(cv_df['val_MAE_C'].mean()),
                'folds': cv_rows,
            },
            'retrain_epochs': retrain_epochs,
            'final_atom_scaler': final_atom_scaler,
            'final_label_scaler': final_label_scaler,
            'mean_channel_attention': mean_weights,
            'test_used_for_optimization': False,
            'random_state': args.seed,
            'device': str(device),
        },
    )


if __name__ == '__main__':
    main()
