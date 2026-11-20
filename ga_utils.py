#!/usr/bin/env python3
"""遗传算法（GA）超参数寻优与 5 折防泄漏特征折分共享工具。"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from feature_pipeline import (
    FeatureConfig,
    FeatureMatrices,
    RawFeatureData,
    fit_transform_feature_fold,
)

CV_FOLDS = 5
GA_POP_SIZE = 40
GA_GENERATIONS = 30
GA_CROSSOVER_RATE = 0.8
GA_MUTATION_RATE = 0.1
GA_ELITE_COUNT = 2
GA_TOURNAMENT_K = 3


def make_stratified_kfold_splits(
    labels: np.ndarray, n_splits: int = CV_FOLDS, seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """按 Tg 分位分层划分 n_splits 折，与图模型保持完全一致。"""
    bins = pd.qcut(labels, q=10, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(np.sort(tr), np.sort(va)) for tr, va in skf.split(np.zeros(len(labels)), bins)]


def prepare_descriptor_cv_folds(
    raw: RawFeatureData,
    config: FeatureConfig,
    n_splits: int = CV_FOLDS,
    seed: int = 42,
) -> list[dict]:
    """在每个 CV 训练折内部独立完成缺失值填补、相关性去重、指纹筛选与 SelectKBest。"""
    splits = make_stratified_kfold_splits(raw.y_train, n_splits=n_splits, seed=seed)
    folds = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        fm: FeatureMatrices = fit_transform_feature_fold(
            raw, train_idx=tr_idx, eval_idx=va_idx, config=config,
            use_test=False, verbose=(fold_idx == 0),
        )
        folds.append({
            'fold': fold_idx,
            'X_train': fm.X_train,
            'y_train': fm.y_train,
            'X_val': fm.X_test,
            'y_val': fm.y_test,
            'feature_names': fm.feature_names,
        })
    return folds


def run_discrete_ga(
    model_tag: str,
    search_space: dict[str, tuple],
    eval_fn: Callable[[dict], dict],
    output_dir: Path,
    pop_size: int = GA_POP_SIZE,
    generations: int = GA_GENERATIONS,
    crossover_rate: float = GA_CROSSOVER_RATE,
    mutation_rate: float = GA_MUTATION_RATE,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """通用离散超参数遗传算法（最小化 eval_fn 返回的 mean_cv_rmse）。"""
    rng = random.Random(seed)
    gene_names = tuple(search_space.keys())
    fitness_cache: dict[tuple[int, ...], dict] = {}
    history_rows: list[dict] = []

    def decode(chrom: tuple[int, ...]) -> dict:
        return {name: search_space[name][idx] for name, idx in zip(gene_names, chrom)}

    def rand_chrom() -> tuple[int, ...]:
        return tuple(rng.randrange(len(search_space[name])) for name in gene_names)

    def tournament(pop: list[tuple[int, ...]], fits: list[float]) -> tuple[int, ...]:
        chosen = rng.sample(range(len(pop)), k=min(GA_TOURNAMENT_K, len(pop)))
        return pop[min(chosen, key=lambda i: fits[i])]

    def crossover(
        p1: tuple[int, ...], p2: tuple[int, ...],
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

    def mutate(chrom: tuple[int, ...]) -> tuple[int, ...]:
        genes = list(chrom)
        if rng.random() < mutation_rate:
            pos = rng.randrange(len(genes))
            choices = [i for i in range(len(search_space[gene_names[pos]])) if i != genes[pos]]
            if choices:
                genes[pos] = rng.choice(choices)
        return tuple(genes)

    def eval_pop(pop: list[tuple[int, ...]], gen_label: str) -> list[float]:
        fits: list[float] = []
        new_evals = 0
        for idx, chrom in enumerate(pop):
            if chrom not in fitness_cache:
                new_evals += 1
                params = decode(chrom)
                t0 = time.perf_counter()
                res = eval_fn(params)
                dt = time.perf_counter() - t0
                fitness_cache[chrom] = {
                    'mean_cv_rmse': float(res['mean_cv_rmse']),
                    'fold_rmses': [float(x) for x in res['fold_rmses']],
                    'extra': res.get('extra', {}),
                    'eval_seconds': dt,
                    'params': params,
                }
                print(
                    f'  [{model_tag} | {gen_label}] 个体 {idx+1:2d}/{len(pop)} '
                    f'(新评估 #{len(fitness_cache):3d}, {dt:4.2f}s) '
                    f'CV RMSE={res["mean_cv_rmse"]:.3f} ℃ | {params}',
                    flush=True,
                )
            fits.append(fitness_cache[chrom]['mean_cv_rmse'])
        if new_evals == 0:
            print(f'  [{model_tag} | {gen_label}] 所有个体均命中缓存。', flush=True)
        return fits

    population: list[tuple[int, ...]] = []
    seen_init: set[tuple[int, ...]] = set()
    while len(population) < pop_size:
        c = rand_chrom()
        if c not in seen_init or len(seen_init) >= 200:
            seen_init.add(c)
            population.append(c)

    print(f'\n=== [{model_tag}] 遗传算法（GA）超参数寻优开始 ===')
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
        'best_params': json.dumps(decode(population[best_idx]), ensure_ascii=False),
    })
    print(f'>> [{model_tag}] Gen 00 完成: 最优 5折 CV RMSE = {fitnesses[best_idx]:.4f} ℃',
          flush=True)

    for gen in range(1, generations + 1):
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i])
        next_pop: list[tuple[int, ...]] = [population[i] for i in ranked[:GA_ELITE_COUNT]]

        while len(next_pop) < pop_size:
            p1 = tournament(population, fitnesses)
            p2 = tournament(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            next_pop.append(mutate(c1))
            if len(next_pop) < pop_size:
                next_pop.append(mutate(c2))

        population = next_pop
        fitnesses = eval_pop(population, f'Gen {gen:02d}')
        best_idx = int(np.argmin(fitnesses))
        history_rows.append({
            'generation': gen,
            'best_cv_rmse': float(fitnesses[best_idx]),
            'mean_cv_rmse': float(np.mean(fitnesses)),
            'worst_cv_rmse': float(np.max(fitnesses)),
            'unique_evaluated': len(fitness_cache),
            'best_params': json.dumps(decode(population[best_idx]), ensure_ascii=False),
        })
        print(
            f'>> [{model_tag}] Gen {gen:02d}/{generations:02d} 完成: '
            f'最优 5折 CV RMSE = {fitnesses[best_idx]:.4f} ℃ '
            f'(均值 {np.mean(fitnesses):.4f} ℃, 累计唯一配置 {len(fitness_cache)})',
            flush=True,
        )

    best_chrom = min(fitness_cache.keys(), key=lambda k: fitness_cache[k]['mean_cv_rmse'])
    best_rec = fitness_cache[best_chrom]

    eval_rows = []
    for _, rec in sorted(fitness_cache.items(), key=lambda kv: kv[1]['mean_cv_rmse']):
        row = {**rec['params']}
        row['mean_cv_rmse_C'] = rec['mean_cv_rmse']
        row['fold_rmses_C'] = json.dumps([round(x, 4) for x in rec['fold_rmses']])
        if rec['extra']:
            row['extra'] = json.dumps(rec['extra'], ensure_ascii=False)
        row['eval_seconds'] = round(rec['eval_seconds'], 3)
        eval_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history_rows)
    evals_df = pd.DataFrame(eval_rows)
    history_df.to_csv(output_dir / 'ga_history.csv', index=False, encoding='utf-8-sig')
    evals_df.to_csv(output_dir / 'ga_evaluations.csv', index=False, encoding='utf-8-sig')

    return best_rec['params'], history_df, evals_df
